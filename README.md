# P10-Hybrid_Risk_Scoring_Model

> **Goal:** Show how classic tabular ML and modern large‑language‑model (LLM) signals can work **together** to improve small‑business credit‑risk assessment.
>
> ---

## 1 · Why It Matters
- Alternative lending platforms (e.g., **Credibly**) need faster, more flexible risk tools than rule‑based scorecards.
- Traditional models ignore the borrower’s own narrative. Recent LLM advances let us **quantify that text** and fold it back into underwriting.
- We deliver a **hybrid pipeline**—XGBoost on structured data **+** LLM‑derived features from loan descriptions—so analysts can audit both the numeric and the narrative sides of risk.

---

## 2 · Objectives
1. **Baseline** Train a solid XGBoost classifier on cleaned Lending‑Club data.  
2. **Enhance** Generate and score borrower narratives with open-source large language models, then feed those scores back into the tabular model.  
3. **Explore** Document where AI helps (or doesn’t) and outline real‑world deployment trade‑offs.

---   

## 3 · Data at a Glance
| Item                              | Value |
|-----------------------------------|-------|
| Period                            | 2007 – Q3 2020 |
| Observations                      | **2.9 M** loans |
| Raw columns                       | 141 |
| Kept after missing‑value pruning  | 106 |
| Outcome classes (re‑mapped)       | **good**, **bad**, **other** |

---

## 4 · Method Overview
| Stage                    | What We Do (poster summary) |
|--------------------------|-----------------------------|
| **Pre‑processing**       | Drop > 50 %‑null columns, binary‑encode Y/N flags, one‑hot rare categoricals, map sub‑grades to ordinal risk buckets. |
| **Baseline Model**       | XGBoost with early stopping & class weighting. Random train–test split outperforms strict time splits on this dataset. |
| **LLM Add‑on**           | 1. Generate realistic borrower *descriptions* from structured fields.<br>2. Use GPT‑4o‑mini to craft the few‑shot examples, then embed those shots in the TinyLlama prompt so TinyLlama returns a 0 – 1 risk score.<br>3. Feed that score back into the tabular feature set. |
| **Evaluation**           | Track accuracy, recall, and SHAP importance; compare baseline vs. hybrid. |

---

## 5 · Key Findings
- **Hybrid lift exists** – traditional features still rule, but narrative‑based risk adds a measurable signal (confirmed by SHAP).  
- **Model split choice matters** – random splits inflate metrics; temporal splits paint a harsher, more production‑like picture.  
- **LLM trade‑offs** – GPT gives the most accurate text‑risk scores but costs more, while open‑source models are cheaper yet less accurate.

---

## Repo Structure

- **main/** – all core work include:
  -  exploratory EDA  
  -  baseline XGBoost model  
  -  TinyLlama hybrid  
  -  other research & explorations  
- **scripts/** – quick one‑off or scratch scripts created during the project
