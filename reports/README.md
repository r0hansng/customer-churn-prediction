# Reports Directory

This directory contains reports, evaluations, and analysis outputs for the
Customer Churn Prediction & Retention Strategy project.

## Main Technical Report

The full technical report covering both milestones is available as:

- **`report_tex.tex`** (project root) — LaTeX source for the Final Technical Report
- **`report_pdf.pdf`** (project root) — Compiled Final Technical Report
- **Overleaf link:** https://www.overleaf.com/read/xnyxtzmjfzjy#67cbfb

## Report Contents

| Section | What's Covered |
|---------|---------------|
| Problem Statement | Business case for churn prediction |
| Data Description | IBM Telco dataset (7,043 records, 26.5% churn) |
| EDA | Churn rates by contract, internet service, payment method |
| Methodology | ImbPipeline, SMOTE, feature engineering |
| Evaluation | 5 models × metrics at threshold 0.50 and 0.35 |
| Optimisation | RandomizedSearchCV + ROC-AUC, best hyperparameters |
| Milestone 2 | LangGraph + RAG retention engine architecture |
| Batch Collective Retention | Aggregate-then-generate strategy (1 API call for any batch size) |
| Conclusion | Final results summary |

## Batch Collective Retention Strategy

Rather than calling the LLM once per predicted churner (expensive and slow at scale),
the batch engine follows an **aggregate-then-generate** pattern:

1. Compute **15 population-level statistics** across all at-risk customers
   (avg tenure, avg charges, dominant contract, internet service, payment method, etc.)
2. Retrieve **top-3 policy chunks** from FAISS using the aggregate profile as the query
3. Make **exactly one Gemini 2.5 Flash call** → returns a structured retention programme:
   - Root Cause Analysis
   - Recommended Retention Initiatives (with policy citations)
   - Estimated Impact
   - Sub-Segment Targeting (seniors, high-bill, no-partner customers)

> Cost: **1 Gemini API call** regardless of batch size (tested on 7,043-row full dataset).

## Model Performance Summary (test set, threshold = 0.50)

| Model | Recall | F1 | ROC-AUC |
|-------|--------|----|---------|
| Logistic Regression | **78.88%** | 61.91% | **84.37%** |
| Decision Tree | 70.05% | 60.09% | 80.35% |
| MLP (Neural Net) | 65.78% | 56.81% | 79.39% |
| **Random Forest** | 71.93% | **62.05%** | 83.88% |
| Gradient Boosting | 68.45% | 60.95% | 84.18% |

> At threshold 0.35, Logistic Regression achieves **89.57% Recall** on the Churn class.
