# RGAP: Rule-Gated Attention Priors for Unsupervised APT Detection

> **IEEE ICDM 2026** | [Paper](#) | [Code](https://github.com/apt-rgap/rgap)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

RGAP is a **neuro-symbolic graph attention model** for unsupervised Advanced Persistent Threat (APT) detection on provenance graphs. It bridges symbolic rule mining and graph neural networks by injecting association-rule evidence directly into the GNN attention mechanism through a learnable per-edge gate — without requiring any attack labels during training.

![RGAP](https://github.com/apt-rgap/rgap/blob/main/figures/rgapPipeline.png)

---

## How It Works

### 1. Two complementary symbolic signals

RGAP mines two types of association rules from binary process–action matrices:

| Signal | What it captures | APT relevance |
|---|---|---|
| **Rare-rule satisfaction** | Action co-occurrences that almost never appear in normal processes | Direct fingerprint of APT behaviour |
| **Frequent-rule violation** | Processes that initiate a common action sequence but suppress the expected consequent | Detects APT actors who evade frequency-based detectors |

Together they capture both *what attackers do* and *what they fail to do*.

### 2. Four-node heterogeneous provenance graph

Each scenario is represented as a heterogeneous graph with four node types and five directed edge types:

```
Nodes:  process (P) · action (A) · rare_rule (Rr) · freq_rule (Rf)
Edges:  performed · spawn · similarity · satisfy_rare · violate_freq
```

The `spawn` edges are extracted from the process-parent matrix **P**, encoding the OS process hierarchy — a relational signal absent from all prior GNN-based APT detectors.

### 3. Rule-gated attention mechanism

For each directed edge *(i → j)*, RGAP computes:

```
e_ij = e_base_ij  +  γ_ij · b_ij
```

- **`e_base_ij`** — standard GAT attention logit (neural evidence)
- **`b_ij`** — rule-evidence bias computed by a small MLP from symbolic edge features
- **`γ_ij ∈ [0,1]`** — learnable gate conditioned on both symbolic features and neural embeddings

The gate learns *per-edge* when to trust rule evidence and when to defer to neural context. Setting `γ=0` recovers vanilla GAT; setting `w_n=0` yields a fixed symbolic prior. RGAP strictly generalises both.

### 4. Self-supervised training (no attack labels)

| Objective | Description |
|---|---|
| **Frequency-Aware Masking (FAM)** | Only frequent feature columns are masked; rare columns are deliberately left uncorrupted so the bottleneck encoder must learn to represent the rare signal |
| **Contrastive augmentation** | Rule-supported edges are preserved across augmented views, implicitly rewarding the gate to open on symbolic signals |

### 5. Anomaly scoring & explainability

At inference, each process receives a composite score:

```
s_final(p) = λ_recon · s̃_recon(p)  +  λ_emb · s̃_emb(p)
```

Top-ranked processes are explained by reporting the **highest-gate rule edges** with corresponding rule patterns and optional MITRE ATT&CK mappings.

---

## Results (DARPA TC — nDCG | AUC-ROC)


![Results](https://github.com/apt-rgap/rgap/blob/main/figures/rgapResults.png)

![Parameters](https://github.com/apt-rgap/rgap/blob/main/figures/grapSensitivity.png)



---

## Installation

```bash
git clone https://github.com/apt-rgap/rgap.git
cd rgap
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 2.0+, PyTorch Geometric, scikit-learn, pandas, mlxtend

---

## Dataset Setup

RGAP is evaluated on the [DARPA Transparent Computing (TC)](https://www.darpa.mil/program/transparent-computing) dataset. Raw logs must be obtained from DARPA directly.

Link to download the data:

```bash
https://gitlab.com/adaptdata
```

This produces:
- `ProcessEvent.csv` — binary process–action matrix
- `ProcessParent.csv` — process-parent matrix
- `GroundTruth.csv`  — ground-truth labels (used only for evaluation)

---

## Quick Start [Check out the main pipeline file here!](https://github.com/apt-rgap/rgap/blob/main/src/rgap_full_pipeline.py)


```python

input_csv  = "../5dir/ProcessEvent.csv"
gt_csv     ="../5dir/5dir_bovia_simple.csv"
parent_csv ="../5dir/ProcessParent.csv"

print("\nTop 20 ranked processes:")
print(df_rank.head(20))
print("\nMetrics:")
print(metrics)
```

### Key hyperparameters

![Sensitivity](https://github.com/apt-rgap/rgap/blob/main/figures/rgapParameters.png)


---

## Repository Structure

```
src/
├── rgap.py               # Main single-file implementation
├── rgap_rule_miner.py    # Walky-G MRI mining + frequent rule mining
├── rgap_graph_builder.py # Heterogeneous graph construction
├── rgap_conv.py          # RGAP gated attention convolution layer
├── rgap_trainer.py       # Self-supervised training (FAM + InfoNCE)
├── rgap_scorer.py        # Anomaly scoring + explainability
├── rgap_pipeline.py      # End-to-end pipeline
├── preprocess/
│   └── darpa_tc_to_matrix.py  # CDM log → X, P matrices
├── data/
│   └── README.md         # Dataset download instructions
└── requirements.txt
```

---

## Explainability Example

For each flagged process, RGAP reports the rule evidence that most influenced the decision.

```
Process: nginx:4821   score: 0.91
  satisfy_rare  γ=0.91  READ×/var/www/html + EXEC×/bin/sh + WRITE×/tmp   → T1059.004

Process: python:9244  score: 0.87
  violate_freq  γ=0.82  CONNECT×remote_IP ⇏ CLOSE_SOCKET                 → T1046
```

Gate values directly indicate which symbolic patterns the model trusted — providing audit-trail-ready explanations mappable to MITRE ATT&CK.

---

## Citation

```bibtex
@inproceedings{rgap2026,
  title     = {{RGAP}: Rule-Gated Attention Priors for Unsupervised {APT} Detection on Provenance Graphs},
  booktitle = {IEEE ICDM},
  year      = {2026},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

