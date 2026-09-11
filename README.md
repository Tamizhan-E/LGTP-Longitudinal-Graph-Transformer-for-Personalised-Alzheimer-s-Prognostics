**LONGITUDINAL GRAPH-TRANSFORMER NETWORK FOR PERSONALISED PREDICTION OF MILD COGNITIVE IMPAIRMENT TO ALZHEIMER'S DISEASE CONVERSION**

> **MSc Dissertation Project — University of Liverpool (COMP702, 2025/26)**  
> Predicting *when* a patient with Mild Cognitive Impairment will convert to Alzheimer's Disease using longitudinal clinical data, population graph reasoning, and survival analysis.

---

## 🏆 Results at a Glance

| Metric | Value | Interpretation |
|---|---|---|
| C-index | **0.865** | Excellent ranking quality |
| AUC @ 24 months | **0.905** | Exceeds ADMV-Net (0.830) |
| AUC @ 36 months | **0.930** | Matches best ADNI literature |
| Brier Score @ 24mo | **0.135** | Well-calibrated predictions |
| Expected Calibration Error | **0.018** | Trustworthy confidence estimates |

> Achieved using **only clinical cognitive CSV files** — no PET imaging, no CSF biomarkers, no raw MRI.

---

## 🎯 What Problem Does This Solve?

Existing AI systems for Alzheimer's research answer the wrong question.

They ask: *"Does this patient have MCI or AD right now?"*

LGTP asks: ***"When will this patient convert from MCI to Alzheimer's Disease?"***

This shift — from static diagnosis to dynamic prognosis — is what clinicians actually need for treatment planning, clinical trial recruitment, and patient counselling.

---

## 🔬 The Three Innovations

```
Patient visits          Transformer          Population Graph        Survival Output
over time         →    (trajectory)    →    (similar patients)  →   (risk curve)
(B, T, 14)            (B, 128)              (N, 128)                P(convert by t)
```

### 1. Temporal Transformer Encoder
- Encodes each patient's **sequence of clinical visits** over time — not just the latest snapshot
- Uses **temporal positional encoding** based on actual month values — handles irregular ADNI visit spacing
- **CLS token** aggregates the full visit sequence into a single 128-dimensional patient trajectory embedding
- **MC Dropout** kept active at inference for uncertainty quantification

### 2. Population Graph (GATv2)
- Builds a **patient similarity graph** — each patient connected to k=10 most similar patients by cosine similarity
- **Graph Attention Network v2** (Brody et al. 2022) enriches each patient's embedding using population-level knowledge
- GATv2 chosen over original GAT for provably superior expressiveness (dynamic attention)

### 3. Survival Analysis Head (DeepSurv + Cox-PH)
- Outputs **personalised risk curves** at 12, 24, and 36 months — not a binary yes/no label
- **Cox partial likelihood loss** handles right-censored patients correctly
- **Temperature scaling** for calibrated probability estimates

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LGTP Pipeline                            │
├─────────────┬──────────────────┬──────────────┬────────────────┤
│  8 ADNI     │  Temporal        │  Population  │  Survival      │
│  CSV files  │  Transformer     │  Graph GAT   │  Head          │
│             │  (3 layers,      │  (2 layers,  │  (DeepSurv)    │
│  → merge    │   4 heads,       │   4 heads,   │                │
│  → 14 feats │   d=128)         │   d=128)     │  P(T≤t|X)     │
│  → 1053 pts │                  │              │  @ 12/24/36mo  │
└─────────────┴──────────────────┴──────────────┴────────────────┘
                      ↓                               ↓
              MC Dropout (50 passes)           SHAP Attribution
              Epistemic uncertainty            Feature importance
```

---

## 📁 Repository Structure

```
lgtp/
├── src/
│   ├── data/
│   │   ├── adni_loader.py       # Merges 8 ADNI CSVs, assigns survival labels
│   │   ├── preprocessing.py     # Subject-level split, normalisation, padding
│   │   └── dataset.py           # PyTorch Dataset + population graph construction
│   ├── models/
│   │   ├── temporal_encoder.py  # Transformer with temporal positional encoding
│   │   ├── graph_network.py     # GATv2 population graph module
│   │   ├── survival_head.py     # DeepSurv head + Cox partial likelihood loss
│   │   └── lgtp_model.py        # End-to-end LGTP model
│   ├── training/
│   │   └── trainer.py           # Full-graph training strategy + early stopping
│   ├── evaluation/
│   │   ├── metrics.py           # C-index, time-AUC, Brier, calibration
│   │   └── evaluator.py         # Figure generation pipeline
│   └── explainability/
│       ├── shap_explainer.py    # KernelSHAP biomarker attribution
│       └── attention_viz.py     # Visit-level attention heatmap
├── notebooks/
│   └── LGTP_Full_Pipeline.ipynb # Complete Google Colab pipeline
├── tests/
│   └── test_model.py            # pytest unit tests
└── requirements.txt
```

---

## 🚀 Quick Start

### Option A — Google Colab (Recommended)
1. Open `notebooks/LGTP_Full_Pipeline.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run cells in order — upload ADNI files when prompted in Cell 2

### Option B — Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/lgtp.git
cd lgtp
pip install -r requirements.txt

# Place your 8 ADNI CSV files in data/raw/
python scripts/02_filter_cohort.py    # preprocess + split
python scripts/03_train.py            # train LGTP
python scripts/04_evaluate.py         # metrics + figures
python scripts/05_explain.py          # SHAP + attention maps
```

---

## 📂 Data Requirements

This project uses the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** dataset.

**Access:** Apply free at [ida.loni.usc.edu](https://ida.loni.usc.edu)

**8 files needed from LONI Study Files:**

| File | Section on LONI | Purpose |
|---|---|---|
| `DXSUM_*.csv` | Diagnostics | Diagnosis labels per visit (CN/MCI/AD) |
| `MMSE_*.csv` | Assessments | Mini Mental State Exam scores |
| `CDR_*.csv` | Assessments | Clinical Dementia Rating |
| `FAQ_*.csv` | Assessments | Functional Activities Questionnaire |
| `ADAS_*.csv` | Assessments | ADAS-Cog cognitive scores |
| `NEUROBAT_*.csv` | Assessments | Neuropsychological battery |
| `PTDEMOG_*.csv` | Enrollment | Demographics (age, sex, education) |
| `REGISTRY_*.csv` | Enrollment | Visit dates for temporal alignment |

Place all files in `data/raw/` — the loader handles date suffixes like `DXSUM_20Aug2026.csv` automatically.

---

## 📈 Key Results

### Ablation Study — Contribution of Each Component

| Model Variant | C-index | AUC 24mo | AUC 36mo | Int. Brier |
|---|---|---|---|---|
| **Full LGTP (proposed)** | **0.865** | **0.905** | **0.930** | **0.154** |
| No-graph variant | 0.864 | 0.904 | 0.930 | 0.158 |
| Baseline-only (single visit) | ~0.810 | ~0.850 | ~0.880 | ~0.195 |

**Key finding:** Longitudinal modelling contributes +0.054 C-index over single-visit baseline — confirming that temporal cognitive trajectory is the primary predictive signal.

### SHAP Feature Importance (Top 5)

| Rank | Feature | Mean \|SHAP\| | Clinical Meaning |
|---|---|---|---|
| 1 | CDR-SB | 0.201 | Primary AD staging instrument (gold standard) |
| 2 | MMSE | 0.178 | Standard cognitive screening |
| 3 | FAQ Total | 0.132 | Functional assessment |
| 4 | ADAS-Cog | 0.124 | Cognitive assessment scale |
| 5 | Logical Memory Delayed | 0.122 | Episodic memory test |

CDR-SB ranking first is consistent with its role as the primary endpoint in Alzheimer's clinical trials — the model learned this without any clinical prior knowledge.

### Calibration
- **Expected Calibration Error: 0.018** — predicted probabilities closely match observed conversion rates
- Correctly classified patients show **3.6× lower epistemic uncertainty** than misclassified ones
- The model reliably signals when it is uncertain — critical for safe clinical use

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3-orange?style=flat-square)
![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-yellow?style=flat-square&logo=googlecolab)

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.0+ | Core deep learning framework |
| PyTorch Geometric | 2.3+ | GATv2 graph neural network |
| scikit-learn | 1.3+ | Preprocessing, calibration |
| lifelines | 0.27+ | Concordance index computation |
| shap | 0.43+ | KernelSHAP attribution |
| pandas | 2.0+ | CSV loading and merging |
| matplotlib | 3.7+ | All visualisations |

---

## ⚠️ Ethical Note

ADNI data is used under an approved Data Use Agreement for non-commercial academic research. No participant re-identification was attempted. All data is handled in compliance with the ADNI Data Use Agreement.

LGTP is designed as a **clinical decision support tool** — predictions must always be interpreted by qualified clinicians and should never be used as the sole basis for a clinical decision.

---

## 👤 Author

**Tamizhan Elango**  
MSc Advanced Data Science and Artificial Intelligence — University of Liverpool (2025/26)  
🔗 [LinkedIn](https://www.linkedin.com/in/tamizhan-e-cse/)  
📧 tamizhan2603@gmail.com

*Supervised by Dr Blaine Keetch — School of Computer Science and Informatics, University of Liverpool*

---

*Data source: Alzheimer's Disease Neuroimaging Initiative (ADNI) — [adni.loni.usc.edu](https://adni.loni.usc.edu)*
