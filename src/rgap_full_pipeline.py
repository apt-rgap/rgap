"""
Last version: Feb 2026.
RGAP: Rule-Gated Attention Priors for Unsupervised APT Detection
================================================================
Single-file implementation integrating:
  - Walky-G MRI mining for true Minimal Rare Itemsets (Definition 1)
  - Pairwise FP-Growth-style frequent rule mining
  - 4-node-type, 9-edge-type heterogeneous graph (flattened)
  - RGAP gated attention: e_ij = e_base + gamma_ij * b_ij
  - Self-supervised training: FAM masking + symmetric InfoNCE
  - Composite anomaly scoring + gate-based explainability

Fixes applied vs. previous version:
  [C1] Rule MLP final bias zero-initialized (Prop 1 correctness)
  [C2] InfoNCE made symmetric
  [C3] FAM uses XOR-style masking (not zero-masking)
  [D1] Walky-G used for rare rules (true MRI, not pairwise approximation)
  [D2] String-based antecedent/consequent (multi-item support)
  [D3] Scoring weights: 0.6 recon + 0.4 emb (paper spec)
  [M1] Explainability reports max gate per process (not sum)
  [M2] Best-F1 uses precision_recall_curve (O(N log N))
"""

import os
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import softmax


# ============================================================
# 0) Reproducibility + metrics
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def safe_ap(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, y_score)


def best_f1_from_scores(y_true, y_score):
    """[M2] O(N log N) via precision_recall_curve instead of O(N^2) loop."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(thresholds[-1])
    return float(f1s[best_idx]), best_thr


def compute_ndcg_all(y_true_binary, y_scores, k=None):
    y_true = np.asarray(y_true_binary, dtype=float).reshape(1, -1)
    y_pred = np.asarray(y_scores, dtype=float).reshape(1, -1)
    return ndcg_score(y_true, y_pred, k=k)


def minmax(x):
    x = np.asarray(x, dtype=float)
    if np.max(x) - np.min(x) < 1e-12:
        return np.zeros_like(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# ============================================================
# 1) Data loading
# ============================================================

def load_binary_matrix_csv(input_csv, index_col=None):
    df = pd.read_csv(input_csv)
    if index_col is not None:
        df = pd.read_csv(input_csv, index_col=index_col)
    else:
        for cand in ["process_id", "uuid", "Object_ID", "subject_uuid", "pid"]:
            if cand in df.columns:
                df = df.set_index(cand)
                break
        else:
            df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = (df > 0).astype(np.float32)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def load_parent_matrix_csv(parent_csv, index_col=None):
    df = pd.read_csv(parent_csv)
    if index_col is not None:
        df = pd.read_csv(parent_csv, index_col=index_col)
    else:
        for cand in ["process_id", "uuid", "Object_ID", "subject_uuid", "pid"]:
            if cand in df.columns:
                df = df.set_index(cand)
                break
        else:
            df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = (df > 0).astype(np.float32)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def load_apt_ids(gt_csv):
    gt_df = pd.read_csv(gt_csv)
    if "label" in gt_df.columns and "uuid" in gt_df.columns:
        apt_ids = gt_df[
            gt_df["label"].astype(str).str.contains("AdmSubject::Node", na=False)
        ]["uuid"].astype(str).tolist()
    else:
        apt_ids = gt_df.iloc[:, 0].astype(str).tolist()
    return set(apt_ids)


# ============================================================
# 2) Rule dataclass  (string-based antecedent/consequent)
# ============================================================

@dataclass
class Rule:
    rid: str
    antecedent: Tuple[str, ...]   # column names (supports multi-item)
    consequent: Tuple[str, ...]   # column names (supports multi-item)
    support_count: int
    confidence: float
    lift: float
    rule_type: str                # "rare" or "freq"

    @property
    def all_items(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.antecedent) | set(self.consequent)))

    def __len__(self):
        return len(self.antecedent) + len(self.consequent)


# ============================================================
# 3) Walky-G: Minimal Rare Itemset Mining
#    (Szathmary et al., CIMCA 2007 / ICTAI 2007)
# ============================================================

class _ITNode:
    """Internal IT-tree node for Walky-G."""
    __slots__ = ("itemset", "tidset", "children", "parent")

    def __init__(self, itemset, tidset, parent=None):
        self.itemset = frozenset(itemset)
        self.tidset = set(tidset)
        self.children: List["_ITNode"] = []
        self.parent = parent

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def support(self):
        return len(self.tidset)


class WalkyG:
    """
    Depth-first vertical mining of Minimal Rare Itemsets.

    After .run(transactions), provides:
      self.minimal_rare_itemsets  : List[(frozenset, support_int, tidset)]
      self.frequent_generators    : List[(frozenset, support_int, tidset)]
      self.fgMap                  : Dict[frozenset, support_int]
    """

    def __init__(self, min_support: int):
        self.min_support = min_support
        self.fgMap: Dict[frozenset, int] = {}
        self.minimal_rare_itemsets: List[Tuple] = []
        self.frequent_generators: List[Tuple] = []

    def run(self, transactions: List[Set[str]]) -> List[Tuple]:
        vertical_db = self._build_vertical_db(transactions)
        num_objects = len(transactions)

        root = _ITNode(frozenset(), set(range(num_objects)))
        self.fgMap = {frozenset(): num_objects}
        self.minimal_rare_itemsets = []
        self.frequent_generators = []

        root_children = []
        for item, tidset in vertical_db.items():
            supp = len(tidset)
            if supp == 0:
                continue
            if self.min_support <= supp < num_objects:
                root_children.append(_ITNode(frozenset([item]), tidset, parent=root))
            elif 0 < supp < self.min_support:
                self._save_mri(frozenset([item]), supp, tidset)

        root_children.sort(key=lambda n: sorted(n.itemset))
        root.children = root_children

        for child in reversed(root.children):
            self._save_fg(child)
            self._extend(child)

        return self.minimal_rare_itemsets

    def _build_vertical_db(self, transactions):
        vertical = defaultdict(set)
        for tid, trans in enumerate(transactions):
            for it in trans:
                vertical[it].add(tid)
        return vertical

    def _save_fg(self, node):
        self.fgMap[node.itemset] = node.support()
        self.frequent_generators.append((node.itemset, node.support(), set(node.tidset)))

    def _save_mri(self, itemset, supp, tidset):
        entry = (frozenset(itemset), supp, set(tidset))
        if entry not in self.minimal_rare_itemsets:
            self.minimal_rare_itemsets.append(entry)

    def _extend(self, curr_node):
        parent = curr_node.parent
        if parent is None:
            right_siblings = []
        else:
            siblings = parent.children
            idx = siblings.index(curr_node)
            right_siblings = siblings[idx + 1:]

        for other in right_siblings:
            gen_node = self._get_next_generator(curr_node, other)
            if gen_node is not None:
                curr_node.add_child(gen_node)

        curr_node.children.sort(key=lambda n: sorted(n.itemset))
        for child in reversed(curr_node.children):
            self._save_fg(child)
            self._extend(child)

    def _get_next_generator(self, node_a, node_b):
        cand_itemset = node_a.itemset | node_b.itemset
        if len(cand_itemset) == len(node_a.itemset) or len(cand_itemset) == len(node_b.itemset):
            return None

        cand_tidset = node_a.tidset & node_b.tidset
        cand_support = len(cand_tidset)

        if cand_support < self.min_support:
            if self._all_subsets_are_fgs(cand_itemset):
                self._save_mri(cand_itemset, cand_support, cand_tidset)
            return None

        if cand_tidset == node_a.tidset or cand_tidset == node_b.tidset:
            return None

        if self._subsumes_fg(cand_itemset, cand_support, [node_a.itemset, node_b.itemset]):
            return None

        return _ITNode(cand_itemset, cand_tidset)

    def _all_subsets_are_fgs(self, itemset):
        k = len(itemset)
        if k == 0:
            return False
        for subset in combinations(itemset, k - 1):
            if frozenset(subset) not in self.fgMap:
                return False
        return True

    def _subsumes_fg(self, cand_itemset, cand_support, parents):
        k = len(cand_itemset)
        parent_sets = set(parents)
        for subset in combinations(cand_itemset, k - 1):
            ss = frozenset(subset)
            if ss in parent_sets:
                continue
            if ss in self.fgMap:
                if self.fgMap[ss] == cand_support:
                    return True
            else:
                return True
        return False


def _df_to_transactions(df_binary: pd.DataFrame) -> List[Set[str]]:
    """Convert binary DataFrame to vertical transaction list."""
    transactions = []
    for _, row in df_binary.iterrows():
        transactions.append(set(row.index[row.astype(bool)]))
    return transactions


# ============================================================
# 4) Rule Mining: dual-strategy rare rules + pairwise frequent rules
# ============================================================

def mine_rules(
    X_bin: np.ndarray,
    feature_names: List[str],
    min_rare_support_pct: float = 2.0,    # % of N  (e.g. 2.0 = 2%, 0.01 = 0.01%)
    min_freq_support_pct: float = 10.0,   # % of N
    rare_conf: float = 0.3,
    rare_lift: float = 1.2,
    freq_conf: float = 0.8,
    max_rules_per_type: int = 3000,
    rare_low_lift_max: float = 0.85,
) -> Tuple[List[Rule], List[Rule]]:
    """
    RARE rules: two complementary strategies, whichever yields more rules:

    Strategy A — WalkyG / absolute rarity (sparse datasets, large N, many actions):
        Mines Minimal Rare Itemsets where supp(A∪B) < sigma_r * N.
        Works when N is large and co-occurrences are truly sparse.

    Strategy B — Low-lift / conditional rarity (dense datasets, small M, e.g. bovia):
        Flags pairs of *frequent* items that co-occur LESS than statistical
        independence predicts: lift(A→B) < rare_low_lift_max.
        Captures 'expected-but-absent' patterns — a strong APT signal.
        (A process that has both EXECUTE and WRITE individually but rarely
        together is anomalous relative to the normal population.)

    The strategy with more rules is kept; both are attempted.

    FREQUENT rules: efficient pairwise mining.
    """
    n, d = X_bin.shape
    X_bin = X_bin.astype(np.float32)
    item_counts = X_bin.sum(axis=0).astype(int)   # (d,)
    cooc        = (X_bin.T @ X_bin).astype(int)   # (d, d)

    # Strategy A — WalkyG absolute rarity: supp(A∪B) < (min_rare_support_pct / 100) * N
    min_rare_supp = max(2, int(math.ceil((min_rare_support_pct / 100.0) * n)))
    df_tmp = pd.DataFrame(X_bin, columns=feature_names)
    transactions = _df_to_transactions(df_tmp)

    miner = WalkyG(min_support=min_rare_supp)
    mris  = miner.run(transactions)

    mri_sizes = [len(iset) for iset, _, _ in mris]
    n_binary  = sum(1 for s in mri_sizes if s >= 2)
    print(f"[MRI-A] WalkyG  min_support={min_rare_supp} ({min_rare_support_pct:.4g}% of {n})  "
          f"total MRIs={len(mris)}  size≥2={n_binary}")

    supp_lookup: Dict[frozenset, int] = dict(miner.fgMap)
    for itemset, supp, _ in mris:
        supp_lookup[itemset] = supp

    def _get_supp(fs: frozenset) -> int:
        if fs in supp_lookup:
            return supp_lookup[fs]
        if not fs:
            return n
        cols = [feature_names.index(it) for it in fs if it in feature_names]
        return int(X_bin[:, cols].all(axis=1).sum()) if cols else 0

    rules_A: List[Rule] = []
    for full_itemset, supp_full, _ in mris:
        items = list(full_itemset)
        if len(items) < 2:
            continue
        for r in range(1, len(items)):
            for A_tup in combinations(items, r):
                A = frozenset(A_tup)
                B = full_itemset - A
                if not B:
                    continue
                supp_A = _get_supp(A)
                if supp_A == 0:
                    continue
                conf  = supp_full / supp_A
                supp_B = _get_supp(B)
                lift  = conf / max(supp_B / n, 1e-12)
                if conf < rare_conf or lift < rare_lift:
                    continue
                rules_A.append(Rule(
                    rid=f"mriA_{len(rules_A)}",
                    antecedent=tuple(sorted(A)),
                    consequent=tuple(sorted(B)),
                    support_count=supp_full,
                    confidence=conf, lift=lift,
                    rule_type="rare",
                ))
    print(f"[MRI-A] Rules after conf≥{rare_conf} lift≥{rare_lift}: {len(rules_A)}")

    # ── Strategy B: low-lift / conditional rarity (all action pairs) ────────
    # Evaluate ALL ordered pairs (a→b) — not just frequent ones.
    # A pair is "rare" when a and b co-occur LESS than statistical
    # independence predicts: lift(a→b) < rare_low_lift_max.
    # No upper support gate is applied here: even uncommon actions can
    # exhibit the "expected-but-absent" pattern that flags APT behaviour.
    # Only require at least 2 co-occurrences (sab >= 2) to exclude noise.
    rules_B: List[Rule] = []
    for a in range(d):
        sa = item_counts[a]
        if sa < 2:
            continue
        for b in range(d):
            if a == b:
                continue
            sb = item_counts[b]
            if sb < 2:
                continue
            sab = cooc[a, b]
            if sab < 2:
                continue
            conf = sab / sa
            lift = conf / max(sb / n, 1e-12)
            # rare: observed conf << expected → lift < rare_low_lift_max
            if lift >= rare_low_lift_max:
                continue
            if conf < rare_conf:
                continue
            rules_B.append(Rule(
                rid=f"mriB_{len(rules_B)}",
                antecedent=(feature_names[a],),
                consequent=(feature_names[b],),
                support_count=int(sab),
                confidence=float(conf),
                lift=float(lift),
                rule_type="rare",
            ))

    print(f"[MRI-B] Low-lift rules (all pairs, lift<{rare_low_lift_max}): {len(rules_B)}")

    # ── Pick the strategy that yields more rules ────────────────────────────
    rare_rules = rules_A if len(rules_A) >= len(rules_B) else rules_B
    strategy   = "A (WalkyG)" if len(rules_A) >= len(rules_B) else "B (low-lift)"
    rare_rules.sort(key=lambda r: (r.lift, r.confidence, -r.support_count), reverse=True)
    rare_rules = rare_rules[:max_rules_per_type]
    print(f"[MRI] Using strategy {strategy}  →  {len(rare_rules)} rare rules")

    # ---- FREQUENT rules via pairwise mining ----
    freq_rules: List[Rule] = []
    min_freq_supp = max(1, int(math.ceil((min_freq_support_pct / 100.0) * n)))
    item_counts = X_bin.sum(axis=0).astype(int)
    cooc = (X_bin.T @ X_bin).astype(int)

    for a in range(d):
        supp_a = int(item_counts[a])
        if supp_a < min_freq_supp:
            continue
        for b in range(d):
            if a == b:
                continue
            supp_b = int(item_counts[b])
            supp_ab = int(cooc[a, b])
            if supp_b == 0 or supp_ab == 0:
                continue
            conf = supp_ab / max(supp_a, 1)
            if conf < freq_conf:
                continue
            lift = conf / max(supp_b / n, 1e-12)
            freq_rules.append(Rule(
                rid=f"freq_{a}_{b}",
                antecedent=(feature_names[a],),
                consequent=(feature_names[b],),
                support_count=supp_ab,
                confidence=conf,
                lift=lift,
                rule_type="freq",
            ))

    freq_rules.sort(
        key=lambda r: (r.confidence, r.lift, r.support_count), reverse=True
    )
    freq_rules = freq_rules[:max_rules_per_type]

    return rare_rules, freq_rules


# ============================================================
# 5) Rule tagging  (multi-item, string-based)
# ============================================================

def build_rule_membership(
    X_bin: np.ndarray,
    feature_names: List[str],
    rare_rules: List[Rule],
    freq_rules: List[Rule],
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    rules_rare[p] = list of rare rule indices satisfied by process p
                    (all items in ant ∪ con are active)
    rules_viol[p] = list of freq rule indices violated by process p
                    (antecedent holds but consequent missing)
    """
    n = X_bin.shape[0]
    col_to_idx = {name: i for i, name in enumerate(feature_names)}

    rules_rare: List[List[int]] = [[] for _ in range(n)]
    rules_viol: List[List[int]] = [[] for _ in range(n)]

    # Rare: full itemset (ant ∪ con) must all be active
    for ridx, r in enumerate(rare_rules):
        full_items = set(r.antecedent) | set(r.consequent)
        cols = [col_to_idx[it] for it in full_items if it in col_to_idx]
        if not cols:
            continue
        sat = np.where(X_bin[:, cols].all(axis=1))[0]
        for p in sat:
            rules_rare[p].append(ridx)

    # Freq violation: antecedent ⊆ acts(p) AND consequent ⊄ acts(p)
    for fidx, r in enumerate(freq_rules):
        ant_cols = [col_to_idx[it] for it in r.antecedent if it in col_to_idx]
        con_cols = [col_to_idx[it] for it in r.consequent if it in col_to_idx]
        if not ant_cols:
            continue
        ant_sat = X_bin[:, ant_cols].all(axis=1)
        con_sat = X_bin[:, con_cols].all(axis=1) if con_cols else np.ones(n, dtype=bool)
        for p in np.where(ant_sat & ~con_sat)[0]:
            rules_viol[p].append(fidx)

    return rules_rare, rules_viol


# ============================================================
# 6) Parent mappings (unchanged)
# ============================================================

def build_parent_child_dicts(df_parent, process_ids):
    process_set = set(process_ids)
    parent_of = {}
    children_of = {pid: [] for pid in process_ids}
    if df_parent is None:
        return parent_of, children_of
    common_children = [pid for pid in df_parent.index if pid in process_set]
    for child in common_children:
        row = df_parent.loc[child]
        active_parents = [p for p in row[row == 1].index.tolist() if p in process_set]
        if active_parents:
            parent = active_parents[0]
            parent_of[child] = parent
            children_of[parent].append(child)
    return parent_of, children_of


# ============================================================
# 7) Graph builder  (updated for string-based rules)
# ============================================================

@dataclass
class RGAPGraph:
    process_x: torch.Tensor
    action_x: torch.Tensor
    rare_rule_x: torch.Tensor
    freq_rule_x: torch.Tensor
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_rule_feat: torch.Tensor
    process_ids: List[str]
    action_names: List[str]
    rare_rules: List[Rule]
    freq_rules: List[Rule]
    nP: int
    nA: int
    nRr: int
    nRf: int
    process_offset: int
    action_offset: int
    rare_offset: int
    freq_offset: int


def build_rgap_graph(
    df_binary: pd.DataFrame,
    df_parent: Optional[pd.DataFrame],
    rare_rules: List[Rule],
    freq_rules: List[Rule],
    rules_rare: List[List[int]],
    rules_viol: List[List[int]],
    k_similarity: int = 5,
) -> RGAPGraph:
    process_ids = df_binary.index.tolist()
    action_names = df_binary.columns.tolist()
    col_to_idx = {name: i for i, name in enumerate(action_names)}

    X = df_binary.values.astype(np.float32)
    nP, d = X.shape
    nA = d
    nRr = len(rare_rules)
    nRf = len(freq_rules)

    # Node features
    process_x = torch.tensor(X, dtype=torch.float32)
    action_x = torch.eye(nA, dtype=torch.float32)

    rare_rule_feats = [[
        1.0 / max(r.support_count, 1), r.confidence, float(len(r))
    ] for r in rare_rules]
    rare_rule_x = (
        torch.tensor(rare_rule_feats, dtype=torch.float32) if nRr > 0
        else torch.zeros((0, 3), dtype=torch.float32)
    )

    freq_rule_feats = [[
        1.0 / max(r.support_count, 1), r.confidence, float(len(r)), 1.0
    ] for r in freq_rules]
    freq_rule_x = (
        torch.tensor(freq_rule_feats, dtype=torch.float32) if nRf > 0
        else torch.zeros((0, 4), dtype=torch.float32)
    )

    # Node offsets
    process_offset = 0
    action_offset  = nP
    rare_offset    = nP + nA
    freq_offset    = nP + nA + nRr

    # Node type IDs: 0=process, 1=action, 2=rare_rule, 3=freq_rule
    node_type = torch.zeros(nP + nA + nRr + nRf, dtype=torch.long)
    node_type[action_offset:rare_offset] = 1
    node_type[rare_offset:freq_offset]   = 2
    node_type[freq_offset:]              = 3

    # Edge type IDs
    ETYPE = {
        "performed":        0,
        "performed_rev":    1,
        "spawn":            2,
        "spawn_rev":        3,
        "similarity":       4,
        "satisfy_rare":     5,
        "satisfy_rare_rev": 6,
        "violate_freq":     7,
        "violate_freq_rev": 8,
    }
    # Rule edge types to preserve during augmentation
    RULE_ETYPES = {5, 6, 7, 8}

    src_list, dst_list, etype_list, feat_list = [], [], [], []

    def add_edge(u, v, etype_name, feat):
        src_list.append(u); dst_list.append(v)
        etype_list.append(ETYPE[etype_name]); feat_list.append(feat)

    ZERO4 = [0.0, 0.0, 0.0, 0.0]

    # --- Performed edges ---
    rows, cols = np.nonzero(X)
    for p_idx, a_idx in zip(rows, cols):
        u, v = process_offset + p_idx, action_offset + a_idx
        add_edge(u, v, "performed",     ZERO4)
        add_edge(v, u, "performed_rev", ZERO4)

    # --- Spawn edges ---
    parent_of, _ = build_parent_child_dicts(df_parent, process_ids)
    pid_to_idx = {pid: i for i, pid in enumerate(process_ids)}

    for child_pid, parent_pid in parent_of.items():
        c = pid_to_idx[child_pid]
        p = pid_to_idx[parent_pid]
        u, v = process_offset + p, process_offset + c
        shared = set(rules_rare[p]) & set(rules_rare[c])
        if shared:
            rr = [rare_rules[ri] for ri in shared]
            feat = [float(len(shared)),
                    float(sum(1.0 / max(r.support_count, 1) for r in rr)),
                    float(max(r.confidence for r in rr)),
                    float(max(len(r) for r in rr))]
        else:
            feat = ZERO4
        add_edge(u, v, "spawn",     feat)
        add_edge(v, u, "spawn_rev", feat)

    # --- Similarity edges ---
    if nP > 1:
        k_eff = min(k_similarity + 1, nP)
        nbrs = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
        nbrs.fit(X)
        _, indices = nbrs.kneighbors(X)
        seen = set()
        for i in range(nP):
            for j in indices[i][1:]:
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in seen:
                    continue
                seen.add(pair)
                shared = set(rules_rare[i]) & set(rules_rare[j])
                if shared:
                    rr = [rare_rules[ri] for ri in shared]
                    feat = [float(len(shared)),
                            float(sum(1.0 / max(r.support_count, 1) for r in rr)),
                            float(max(r.confidence for r in rr)),
                            float(max(len(r) for r in rr))]
                else:
                    feat = ZERO4
                u, v = process_offset + i, process_offset + j
                add_edge(u, v, "similarity", feat)
                add_edge(v, u, "similarity", feat)

    # --- Satisfy-rare edges ---
    for p_idx, rr_list in enumerate(rules_rare):
        for ridx in rr_list:
            r = rare_rules[ridx]
            u = process_offset + p_idx
            v = rare_offset + ridx
            feat = [1.0 / max(r.support_count, 1), r.confidence, float(len(r)), 0.0]
            add_edge(u, v, "satisfy_rare",     feat)
            add_edge(v, u, "satisfy_rare_rev", feat)

    # --- Violate-freq edges ---
    for p_idx, fr_list in enumerate(rules_viol):
        # Compute active action names for this process
        active_names: Set[str] = {
            action_names[j] for j in np.where(X[p_idx] == 1)[0]
        }
        for fidx in fr_list:
            r = freq_rules[fidx]
            viol_margin = float(len(set(r.consequent) - active_names))
            u = process_offset + p_idx
            v = freq_offset + fidx
            feat = [1.0 / max(r.support_count, 1), r.confidence, float(len(r)), viol_margin]
            add_edge(u, v, "violate_freq",     feat)
            add_edge(v, u, "violate_freq_rev", feat)

    edge_index    = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type_t   = torch.tensor(etype_list, dtype=torch.long)
    edge_rule_feat = torch.tensor(feat_list, dtype=torch.float32)

    return RGAPGraph(
        process_x=process_x, action_x=action_x,
        rare_rule_x=rare_rule_x, freq_rule_x=freq_rule_x,
        node_type=node_type,
        edge_index=edge_index, edge_type=edge_type_t, edge_rule_feat=edge_rule_feat,
        process_ids=process_ids, action_names=action_names,
        rare_rules=rare_rules, freq_rules=freq_rules,
        nP=nP, nA=nA, nRr=nRr, nRf=nRf,
        process_offset=process_offset, action_offset=action_offset,
        rare_offset=rare_offset, freq_offset=freq_offset,
    )


# ============================================================
# 8) RGAP convolution  [C1] zero-init rule MLP final bias
# ============================================================

class RGAPConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_edge_types, rule_feat_dim=4, dropout=0.1):
        super().__init__()
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.rel_emb = nn.Embedding(num_edge_types, out_dim)

        self.attn_vec = nn.Parameter(torch.randn(out_dim * 2 + out_dim))

        # [C1] Zero-init final bias so b_ij = 0 when x^rule = 0 (Proposition 1)
        _rule_final = nn.Linear(out_dim, 1)
        nn.init.zeros_(_rule_final.bias)
        self.rule_mlp = nn.Sequential(
            nn.Linear(rule_feat_dim, out_dim),
            nn.ReLU(),
            _rule_final,
        )

        self.gate_rule   = nn.Linear(rule_feat_dim, 1)
        self.gate_neural = nn.Linear(out_dim * 2, 1)
        self.msg_lin     = nn.Linear(out_dim, out_dim)
        self.dropout     = dropout

    def forward(self, x, edge_index, edge_type, edge_rule_feat, total_nodes=None):
        src, dst = edge_index[0], edge_index[1]

        h_src = self.W_k(x[src])
        h_dst = self.W_q(x[dst])
        rel   = self.rel_emb(edge_type)

        # Base GAT logit (relation-conditioned)
        base_in = torch.cat([h_src, h_dst, rel], dim=1)
        e_base  = F.leaky_relu((base_in * self.attn_vec).sum(dim=1), negative_slope=0.2)

        # Rule bias b_ij
        b_ij = self.rule_mlp(edge_rule_feat).squeeze(-1)

        # Gate gamma_ij
        gamma_ij = torch.sigmoid(
            self.gate_rule(edge_rule_feat).squeeze(-1) +
            self.gate_neural(torch.cat([h_src, h_dst], dim=1)).squeeze(-1)
        )

        e = e_base + gamma_ij * b_ij

        # Safe softmax with explicit total_nodes
        n_nodes = total_nodes if total_nodes is not None else int(dst.max().item()) + 1
        alpha = softmax(e, dst, num_nodes=n_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = self.msg_lin(self.W_v(x[src]) + rel) * alpha.unsqueeze(-1)

        out = torch.zeros(n_nodes, msg.shape[-1], device=x.device)
        out.index_add_(0, dst, msg)
        return out, alpha, gamma_ij, b_ij


# ============================================================
# 9) Full RGAP model
# ============================================================

class RGAPModel(nn.Module):
    def __init__(self, proc_in_dim, action_in_dim, rare_rule_in_dim,
                 freq_rule_in_dim, hidden_dim=128, num_edge_types=9,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.proc_proj   = nn.Linear(proc_in_dim, hidden_dim)
        self.action_proj = nn.Linear(action_in_dim, hidden_dim)
        self.rare_proj   = nn.Linear(rare_rule_in_dim, hidden_dim) if rare_rule_in_dim > 0 else None
        self.freq_proj   = nn.Linear(freq_rule_in_dim, hidden_dim) if freq_rule_in_dim > 0 else None

        self.layers = nn.ModuleList([
            RGAPConv(hidden_dim, hidden_dim, num_edge_types, rule_feat_dim=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = dropout

    def _total_nodes(self, graph: RGAPGraph) -> int:
        return graph.nP + graph.nA + graph.nRr + graph.nRf

    def encode_initial(self, graph: RGAPGraph, masked_process_x=None):
        px = masked_process_x if masked_process_x is not None else graph.process_x
        hs = [self.proc_proj(px), self.action_proj(graph.action_x)]
        if graph.nRr > 0 and self.rare_proj is not None:
            hs.append(self.rare_proj(graph.rare_rule_x))
        if graph.nRf > 0 and self.freq_proj is not None:
            hs.append(self.freq_proj(graph.freq_rule_x))
        return torch.cat(hs, dim=0)

    def forward(self, graph: RGAPGraph, masked_process_x=None):
        x = self.encode_initial(graph, masked_process_x)
        total_nodes = self._total_nodes(graph)
        all_gates, all_priors = [], []

        for conv in self.layers:
            x_new, alpha, gamma_ij, b_ij = conv(
                x, graph.edge_index, graph.edge_type, graph.edge_rule_feat,
                total_nodes=total_nodes,
            )
            x = F.dropout(F.relu(x_new), p=self.dropout, training=self.training)
            all_gates.append(gamma_ij)
            all_priors.append(b_ij)

        z_proc   = x[graph.process_offset:graph.action_offset]
        z_action = x[graph.action_offset:graph.rare_offset]
        x_hat    = torch.sigmoid(z_proc @ z_action.t())
        return x_hat, x, all_gates, all_priors


# ============================================================
# 10) FAM masking + graph augmentations
# ============================================================

def compute_feature_mask_probs(df_binary, support_pct_thresh=2.0, p_mask=0.2):
    """support_pct_thresh: percentage of processes that must execute an action
    for it to be considered 'frequent' (e.g. 2.0 = 2%)."""
    freq = df_binary.mean(axis=0).values.astype(np.float32)
    frequent = (freq >= support_pct_thresh / 100.0).astype(np.float32)
    return p_mask * frequent


def apply_fam_to_process_x(process_x, mask_probs, device):
    """[C3] XOR-style flip (both 0→1 and 1→0), not zero-masking."""
    p = torch.tensor(mask_probs, dtype=torch.float32, device=device)
    M = torch.bernoulli(p.unsqueeze(0).expand(process_x.size(0), -1))
    x_masked = (process_x - M).abs()   # binary XOR
    return x_masked, M


def augment_graph_for_contrastive(graph: RGAPGraph, edge_drop_p=0.2):
    """Drop non-rule edges; preserve satisfy_rare and violate_freq edges."""
    RULE_ETYPES = {5, 6, 7, 8}
    et_np = graph.edge_type.cpu().numpy()
    keep = np.array([
        True if et in RULE_ETYPES else (random.random() >= edge_drop_p)
        for et in et_np
    ])
    keep_t = torch.tensor(keep, dtype=torch.bool)
    return RGAPGraph(
        process_x=graph.process_x, action_x=graph.action_x,
        rare_rule_x=graph.rare_rule_x, freq_rule_x=graph.freq_rule_x,
        node_type=graph.node_type,
        edge_index=graph.edge_index[:, keep_t],
        edge_type=graph.edge_type[keep_t],
        edge_rule_feat=graph.edge_rule_feat[keep_t],
        process_ids=graph.process_ids, action_names=graph.action_names,
        rare_rules=graph.rare_rules, freq_rules=graph.freq_rules,
        nP=graph.nP, nA=graph.nA, nRr=graph.nRr, nRf=graph.nRf,
        process_offset=graph.process_offset, action_offset=graph.action_offset,
        rare_offset=graph.rare_offset, freq_offset=graph.freq_offset,
    )


def info_nce_loss(z1, z2, tau=0.5):
    """[C2] Symmetric InfoNCE."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


# ============================================================
# 11) Train RGAP
# ============================================================

def train_rgap(
    graph: RGAPGraph,
    df_binary: pd.DataFrame,
    apt_ids: Optional[set],
    hidden_dim=128,
    num_layers=2,
    lr=1e-3,
    epochs=50,
    lambda_contrast=0.1,
    p_mask=0.2,
    support_pct_thresh=2.0,
    edge_drop_p=0.2,
    tau=0.5,
    device="cpu",
):
    model = RGAPModel(
        proc_in_dim=graph.process_x.size(1),
        action_in_dim=graph.action_x.size(1),
        rare_rule_in_dim=graph.rare_rule_x.size(1) if graph.nRr > 0 else 0,
        freq_rule_in_dim=graph.freq_rule_x.size(1) if graph.nRf > 0 else 0,
        hidden_dim=hidden_dim,
        num_edge_types=int(graph.edge_type.max().item()) + 1,
        num_layers=num_layers,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for attr in ("process_x", "action_x", "rare_rule_x", "freq_rule_x",
                 "node_type", "edge_index", "edge_type", "edge_rule_feat"):
        setattr(graph, attr, getattr(graph, attr).to(device))

    X_true = torch.tensor(
        df_binary.values.astype(np.float32), dtype=torch.float32, device=device
    )

    # FAM: compute frequency from benign reference distribution
    if apt_ids is not None:
        benign_mask = np.array([pid not in apt_ids for pid in graph.process_ids], dtype=bool)
        df_ref = df_binary.iloc[benign_mask] if benign_mask.sum() > 0 else df_binary
    else:
        df_ref = df_binary
    mask_probs = compute_feature_mask_probs(
        df_ref, support_pct_thresh=support_pct_thresh, p_mask=p_mask
    )

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Reconstruction view
        proc_x_masked, _ = apply_fam_to_process_x(graph.process_x, mask_probs, device)
        x_hat, _, _, _   = model(graph, masked_process_x=proc_x_masked)
        loss_recon = F.binary_cross_entropy(x_hat, X_true)

        # Contrastive views
        g1 = augment_graph_for_contrastive(graph, edge_drop_p)
        g2 = augment_graph_for_contrastive(graph, edge_drop_p)
        px1, _ = apply_fam_to_process_x(g1.process_x, mask_probs, device)
        px2, _ = apply_fam_to_process_x(g2.process_x, mask_probs, device)
        _, z1, _, _ = model(g1, masked_process_x=px1)
        _, z2, _, _ = model(g2, masked_process_x=px2)
        z1p = z1[g1.process_offset:g1.action_offset]
        z2p = z2[g2.process_offset:g2.action_offset]
        loss_contrast = info_nce_loss(z1p, z2p, tau=tau)

        loss = loss_recon + lambda_contrast * loss_contrast
        loss.backward()
        optimizer.step()

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"[RGAP] epoch {ep:03d}/{epochs}  "
                  f"recon={loss_recon.item():.6f}  "
                  f"contrast={loss_contrast.item():.6f}  "
                  f"total={loss.item():.6f}")

    return model, mask_probs


# ============================================================
# 12) Score RGAP
# ============================================================

@torch.no_grad()
def score_rgap(model, graph: RGAPGraph, df_binary: pd.DataFrame, device="cpu"):
    model.eval()
    X_true = torch.tensor(
        df_binary.values.astype(np.float32), dtype=torch.float32, device=device
    )
    x_hat, z, all_gates, _ = model(graph, masked_process_x=graph.process_x)
    z_proc = z[graph.process_offset:graph.action_offset]

    # Reconstruction error per process
    recon_err = F.binary_cross_entropy(
        x_hat, X_true, reduction="none"
    ).mean(dim=1).cpu().numpy()

    # Centroid of 20% lowest-error processes
    nP = len(recon_err)
    k = max(1, int(0.2 * nP))
    benign_idx = np.argsort(recon_err)[:k]
    centroid = z_proc[benign_idx].mean(dim=0, keepdim=True)
    emb_dist = torch.norm(z_proc - centroid, dim=1).cpu().numpy()

    recon_n = minmax(recon_err)
    emb_n   = minmax(emb_dist)

    # [D3] Paper-specified weights: 0.6 recon + 0.4 emb
    final_score = 0.6 * recon_n + 0.4 * emb_n

    # [M1] Explainability: MAX gate per process from incoming rule edges
    src_np    = graph.edge_index[0].cpu().numpy()
    dst_np    = graph.edge_index[1].cpu().numpy()
    et_np     = graph.edge_type.cpu().numpy()
    gate_np   = all_gates[-1].detach().cpu().numpy()

    # Incoming rule edges: satisfy_rare_rev (6) and violate_freq_rev (8)
    proc_max_gate = np.zeros(nP, dtype=np.float32)
    for eidx in range(len(et_np)):
        d = int(dst_np[eidx])
        if d < nP and et_np[eidx] in (6, 8):
            proc_max_gate[d] = max(proc_max_gate[d], float(gate_np[eidx]))

    explain = {
        "recon_err":       recon_err,
        "emb_dist":        emb_dist,
        "proc_max_gate":   proc_max_gate,   # renamed from proc_rule_gate
        "z_proc":          z_proc.cpu().numpy(),
        "x_hat":           x_hat.cpu().numpy(),
    }
    return final_score, explain


# ============================================================
# 13) Full pipeline
# ============================================================

def run_rgap_pipeline(
    input_csv,
    gt_csv,
    parent_csv=None,
    index_col=None,
    min_rare_support_pct=2.0,    # e.g. 2.0 = 2%, 0.01 = 0.01%
    min_freq_support_pct=10.0,   # e.g. 10.0 = 10%
    rare_conf=0.3,
    rare_lift=1.2,
    freq_conf=0.8,
    max_rules_per_type=3000,
    k_similarity=5,
    hidden_dim=128,
    num_layers=2,
    lr=1e-3,
    epochs=50,
    lambda_contrast=0.1,
    p_mask=0.2,
    support_pct_thresh=2.0,
    edge_drop_p=0.2,
    tau=0.5,
    seed=42,
    device=None,
    ndcg_ks=(100, 500, 1000),
):
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    df_binary = load_binary_matrix_csv(input_csv, index_col=index_col)
    df_parent = None
    if parent_csv and os.path.exists(parent_csv):
        df_parent = load_parent_matrix_csv(parent_csv, index_col=index_col)

    apt_ids     = load_apt_ids(gt_csv)
    process_ids = df_binary.index.tolist()
    y_true      = np.array([1 if pid in apt_ids else 0 for pid in process_ids], dtype=int)

    print(f"[INFO] processes={df_binary.shape[0]}, actions={df_binary.shape[1]}")
    print(f"[INFO] positives={int(y_true.sum())}, negatives={int((y_true == 0).sum())}")

    # Rule mining: Walky-G MRI rare rules + pairwise frequent rules
    feature_names = df_binary.columns.tolist()
    rare_rules, freq_rules = mine_rules(
        X_bin=df_binary.values.astype(np.float32),
        feature_names=feature_names,
        min_rare_support_pct=min_rare_support_pct,
        min_freq_support_pct=min_freq_support_pct,
        rare_conf=rare_conf,
        rare_lift=rare_lift,
        freq_conf=freq_conf,
        max_rules_per_type=max_rules_per_type,
    )
    print(f"[INFO] rare_rules={len(rare_rules)}, freq_rules={len(freq_rules)}")

    # Rule tagging (string-based, multi-item)
    rules_rare, rules_viol = build_rule_membership(
        df_binary.values.astype(np.float32),
        feature_names,
        rare_rules,
        freq_rules,
    )

    # Graph
    graph = build_rgap_graph(
        df_binary=df_binary, df_parent=df_parent,
        rare_rules=rare_rules, freq_rules=freq_rules,
        rules_rare=rules_rare, rules_viol=rules_viol,
        k_similarity=k_similarity,
    )
    print(f"[INFO] graph nodes={graph.nP + graph.nA + graph.nRr + graph.nRf}")
    print(f"[INFO] graph edges={graph.edge_index.size(1)}")

    # Train
    model, mask_probs = train_rgap(
        graph=graph, df_binary=df_binary, apt_ids=apt_ids,
        hidden_dim=hidden_dim, num_layers=num_layers, lr=lr, epochs=epochs,
        lambda_contrast=lambda_contrast, p_mask=p_mask,
        support_pct_thresh=support_pct_thresh, edge_drop_p=edge_drop_p,
        tau=tau, device=device,
    )

    # Score
    scores, explain = score_rgap(model, graph, df_binary, device=device)

    # Metrics
    auc          = safe_auc(y_true, scores)
    ap           = safe_ap(y_true, scores)
    best_f1, thr = best_f1_from_scores(y_true, scores)
    ndcg_all     = compute_ndcg_all(y_true, scores, k=None)

    print(f"\n[RGAP] AUC      = {auc:.6f}")
    print(f"[RGAP] AP       = {ap:.6f}")
    print(f"[RGAP] Best F1  = {best_f1:.6f}  (threshold={thr:.6f})")
    print(f"[RGAP] nDCG@all = {ndcg_all:.6f}")
    for k in ndcg_ks:
        k_eff = min(int(k), len(y_true))
        print(f"[RGAP] nDCG@{k_eff} = {compute_ndcg_all(y_true, scores, k=k_eff):.6f}")

    df_rank = pd.DataFrame({
        "process_id":    process_ids,
        "rgap_score":    scores,
        "recon_err":     explain["recon_err"],
        "emb_dist":      explain["emb_dist"],
        "max_rule_gate": explain["proc_max_gate"],
        "label_is_apt":  y_true,
    }).sort_values("rgap_score", ascending=False).reset_index(drop=True)

    metrics = {
        "auc": auc, "ap": ap, "best_f1": best_f1,
        "best_f1_threshold": thr, "ndcg_all": ndcg_all,
        "num_rare_rules": len(rare_rules), "num_freq_rules": len(freq_rules),
    }
    artifacts = {
        "model": model, "mask_probs": mask_probs, "graph": graph,
        "rare_rules": rare_rules, "freq_rules": freq_rules, "explain": explain,
    }
    return df_rank, metrics, artifacts


# ============================================================
# 14) Example usage (same input structure as before)
# ============================================================

if __name__ == "__main__":
    set_seed(42)

    input_csv  = "../5dir/ProcessEvent.csv"

    gt_csv     ="../5dir/5dir_bovia_simple.csv"
 
  
    parent_csv ="../5dir/ProcessParent.csv"
 
    

    df_rank, metrics, artifacts = run_rgap_pipeline(
        input_csv=input_csv,
        gt_csv=gt_csv,
        parent_csv=parent_csv,
        index_col=None,
        min_rare_support_pct=0.051,   # 40% — tune down if still 0 rare rules
        min_freq_support_pct=50.0,   # 10%
        rare_conf=0.2,
        rare_lift=1.2,
        freq_conf=0.4,
        max_rules_per_type=1500,
        k_similarity=5,
        hidden_dim=8,
        num_layers=2,
        lr=1e-3,
        epochs=60,
        lambda_contrast=0.1,
        p_mask=0.2,
        support_pct_thresh=2.0,
        edge_drop_p=0.2,
        tau=0.5,
        seed=42,
        ndcg_ks=(100, 500, 1000),
    )

    print("\nTop 20 ranked processes:")
    print(df_rank.head(20))
    print("\nMetrics:")
    print(metrics)
