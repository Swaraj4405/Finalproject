"""First baselines under the zero-day (leave-attack-class-out) protocol.

For each experiment, some attack classes are hidden from training and appear
only at test time. Scaling uses statistics from the training data only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SEED = 42
DATA = "data/botiot_sample.csv"
DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS", "L4_SRC_PORT", "Label", "Attack"]
EXPERIMENTS = {
    "hold DoS+DDoS": ["DoS", "DDoS"],
    "hold Reconnaissance": ["Reconnaissance"],
    "hold Theft": ["Theft"],
}
MAX_UNSEEN = 50000      # cap unseen test flows so every model is scored on the same set
OCSVM_TRAIN = 20000     # One-Class SVM is slow, so train it on a benign subsample

df = pd.read_csv(DATA)
df = df.replace([np.inf, -np.inf], np.nan)
print("loaded", df.shape)
print(df["Attack"].value_counts().to_string(), "\n")


def prepare(train_df, test_dfs):
    """Fill gaps, log-scale and standardise using training statistics only."""
    feats = [c for c in train_df.columns if c not in DROP]
    Xtr = train_df[feats]
    med = Xtr.median()
    nonneg = [c for c in feats if (Xtr[c].fillna(0) >= 0).all()]

    def tf(X):
        X = X.fillna(med).copy()
        X[nonneg] = np.log1p(X[nonneg].clip(lower=0))
        return X

    scaler = StandardScaler().fit(tf(Xtr))
    return scaler.transform(tf(Xtr)), [scaler.transform(tf(d[feats])) for d in test_dfs]


def score(name, pred_benign, pred_known, pred_unseen):
    """pred_* are 0/1 arrays (1 = flagged as attack)."""
    y_true = np.r_[np.zeros(len(pred_benign)), np.ones(len(pred_known)), np.ones(len(pred_unseen))]
    y_pred = np.r_[pred_benign, pred_known, pred_unseen]
    return {"model": name,
            "recall_unseen": round(float(pred_unseen.mean()), 4),
            "recall_known": round(float(pred_known.mean()), 4),
            "FPR": round(float(pred_benign.mean()), 4),
            "F1": round(float(f1_score(y_true, y_pred)), 4),
            "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4)}


rows = []
for exp, hold in EXPERIMENTS.items():
    print("=== Experiment:", exp, "===")
    seen = df[~df["Attack"].isin(hold)]
    unseen = df[df["Attack"].isin(hold)]
    if len(unseen) > MAX_UNSEEN:
        unseen = unseen.sample(MAX_UNSEEN, random_state=SEED)

    train, test = train_test_split(seen, test_size=0.2, stratify=seen["Attack"], random_state=SEED)
    test_benign = test[test["Attack"] == "Benign"]
    test_known = test[test["Attack"] != "Benign"]

    Xtr, (Xb, Xk, Xu) = prepare(train, [test_benign, test_known, unseen])
    ytr = train["Label"].values

    # Supervised baseline: trained on benign + known attacks
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr)
    r = score("RandomForest", rf.predict(Xb), rf.predict(Xk), rf.predict(Xu))
    r["experiment"] = exp
    rows.append(r)
    print(r)

    # Anomaly baseline: trained on benign traffic only
    benign_tr = Xtr[ytr == 0]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(benign_tr), size=min(OCSVM_TRAIN, len(benign_tr)), replace=False)
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(benign_tr[idx])
    to01 = lambda X: (oc.predict(X) == -1).astype(int)   # -1 means outlier = attack
    r = score("OneClassSVM", to01(Xb), to01(Xk), to01(Xu))
    r["experiment"] = exp
    rows.append(r)
    print(r, "\n")

res = pd.DataFrame(rows)[["experiment", "model", "recall_unseen", "recall_known", "FPR", "F1", "MCC"]]
res.to_csv("results/baselines.csv", index=False)
print(res.to_string(index=False))
print("\nsaved results/baselines.csv")