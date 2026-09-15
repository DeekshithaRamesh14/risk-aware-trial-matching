import numpy as np
import pickle
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import shap

OUT_DIR = '/dgxa_home/se25mbds002/trial_match/data/processed'
MODEL_DIR = '/dgxa_home/se25mbds002/trial_match/models'
import os
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X_train = np.load(f'{OUT_DIR}/X_train.npy')
y_train = np.load(f'{OUT_DIR}/y_train.npy')
X_valid = np.load(f'{OUT_DIR}/X_valid.npy')
y_valid = np.load(f'{OUT_DIR}/y_valid.npy')
X_test  = np.load(f'{OUT_DIR}/X_test.npy')
y_test  = np.load(f'{OUT_DIR}/y_test.npy')

with open(f'{OUT_DIR}/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

print(f'Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}')
print(f'Features: {feature_cols}')

# Combine train+valid for final training (use valid only for CV)
X_trainval = np.vstack([X_train, X_valid])
y_trainval = np.concatenate([y_train, y_valid])

# Pipeline: SMOTE inside CV folds
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=4
    ))
])

# 5-fold cross-validation on train+valid
print('\nRunning 5-fold CV...')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    pipeline, X_trainval, y_trainval,
    cv=cv, scoring='roc_auc', n_jobs=1
)
print(f'CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
print(f'Per-fold:   {[round(s,4) for s in cv_scores]}')

# Train final model on full train+valid
print('\nTraining final model...')
pipeline.fit(X_trainval, y_trainval)

# Evaluate on held-out test set
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)
test_auc = roc_auc_score(y_test, y_prob)
print(f'\nTest ROC-AUC: {test_auc:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Save model
with open(f'{MODEL_DIR}/xgb_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print(f'Model saved to {MODEL_DIR}/xgb_pipeline.pkl')

# SHAP values on test set
print('\nComputing SHAP values...')
explainer = shap.TreeExplainer(pipeline.named_steps['clf'])
X_test_transformed = pipeline.named_steps['smote']
# SHAP on raw test features (no SMOTE at inference)
shap_values = explainer.shap_values(X_test)
print('SHAP mean absolute values per feature:')
for fname, val in sorted(
    zip(feature_cols, np.abs(shap_values).mean(axis=0)),
    key=lambda x: -x[1]
):
    print(f'  {fname:30s}: {val:.4f}')

# Save explainer
with open(f'{MODEL_DIR}/shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)
print('\nSHAP explainer saved.')
print('\nDone.')
