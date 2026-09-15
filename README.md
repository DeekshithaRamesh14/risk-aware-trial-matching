# Risk-Aware Patient–Trial Matching

A 3-stage AI pipeline that retrieves semantically relevant clinical trials,
reranks by precision, and penalises by predicted failure risk.

## Pipeline
- Stage 1: PubMedBERT bi-encoder + FAISS retrieval (11,601 → 100)
- Stage 2: MiniLM cross-encoder reranking (100 → 20)
- Stage 3: XGBoost risk classifier with SHAP (ROC-AUC 0.8312)

## Results
- CV ROC-AUC: 0.8859 ± 0.0077
- Test ROC-AUC: 0.8312
- Test Accuracy: 80%

## Stack
Python · PubMedBERT · FAISS · XGBoost · SHAP · FastAPI

## Authors
Deekshitha R · Twinkle Sahu · se25mbds002
