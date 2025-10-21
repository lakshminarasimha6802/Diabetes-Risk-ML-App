from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import joblib
from datetime import datetime

def train_and_save():
    # Load regression target and convert to binary risk (top 30% = 1)
    data = load_diabetes()
    X = data.data                  # (442, 10)
    y_reg = data.target            # continuous
    threshold = np.percentile(y_reg, 70)
    y = (y_reg >= threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    meta = {
        "problem": "diabetes_risk_binary",
        "classes": ["low_risk", "high_risk"],
        "algorithm": "StandardScaler + LogisticRegression",
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(auc), 4),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "threshold_percentile": 70,
        "features": data.feature_names,   # ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
        "note": "Educational model; not for medical decisions."
    }

    joblib.dump({"model": pipe, "meta": meta}, "model.pkl")
    print("Saved model.pkl with meta:", meta)

if __name__ == "__main__":
    train_and_save()
