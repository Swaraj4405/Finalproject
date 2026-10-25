"""Autoencoder vs One-Class SVM vs Random Forest under the zero-day protocol.

Anomaly models learn what normal looks like, so the scaler and the detection
threshold are fitted on BENIGN training flows only. Hidden attack classes never
influence scaling, training or thresholds.
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tensorflow import keras

SEED = 42
keras.utils.set_random_seed(SEED)
DATA = "data/botiot_sample.csv"
DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS", "L4_SRC_PORT", "Label", "Attack"]
EXPERIMENTS = {
    "hold DoS+DDoS": ["DoS", "DDoS"],
    "hold Reconnaissance": ["Reconnaissance"],
    "hold Theft": ["Theft"],
}
MAX_UNSEEN = 50000
OCSVM_TRAIN = 20000
PERCENTILES = [95, 99]   # threshold = this percentile of benign validation error

df = pd.read_csv(DATA).replace([np.inf, -np.inf], np.nan)
print("loaded", df.shape, "\n")


def prepare(train_df, test_dfs):
    """Median fill, log-scale and standardise using BENIGN training flows only."""
    feats = [c for c in train_df.columns if c not in DROP]
    ben = train_df.loc[train_df["Attack"] == "Benign", feats]
    med = ben.median().fillna(0)
    nonneg = [c for c in feats if (ben[c].fillna(0) >= 0).all()]

    def tf(X):
        X = X.fillna(med).copy()
        X[nonneg] = np.log1p(X[nonneg].clip(lower=0))
        return X

    scaler = StandardScaler().fit(tf(ben))
    out = lambda d: scaler.transform(tf(d[feats])).astype("float32")
    return out(train_df), [out(d) for d in test_dfs]


def build_ae(d):
    inp = keras.Input(shape=(d,))
    x = keras.layers.Dense(32, activation="relu")(inp)
    x = keras.layers.Dense(16, activation="relu")(x)
    z = keras.layers.Dense(8, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(z)
    x = keras.layers.Dense(32, activation="relu")(x)
    model = keras.Model(inp, keras.layers.Dense(d)(x))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def recon_error(model, X):
    return np.mean((X - model.predict(X, batch_size=4096, verbose=0)) ** 2, axis=1)


def score(exp, name, pb, pk, pu):
    y_true = np.r_[np.zeros(len(pb)), np.ones(len(pk)), np.ones(len(pu))]
    y_pred = np.r_[pb, pk, pu]
    return {"experiment": exp, "model": name,
            "recall_unseen": round(float(pu.mean()), 4),
            "recall_known": round(float(pk.mean()), 4),
            "FPR": round(float(pb.mean()), 4),
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
    tb, tk = test[test["Attack"] == "Benign"], test[test["Attack"] != "Benign"]
    Xtr, (Xb, Xk, Xu) = prepare(train, [tb, tk, unseen])
    ytr = train["Label"].values
    benign = Xtr[ytr == 0]

    # 1) Random Forest (supervised, sees known attacks)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr)
    rows.append(score(exp, "RandomForest", rf.predict(Xb), rf.predict(Xk), rf.predict(Xu)))
    print(rows[-1])

    # 2) One-Class SVM (benign only)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(benign), size=min(OCSVM_TRAIN, len(benign)), replace=False)
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(benign[idx])
    to01 = lambda X: (oc.predict(X) == -1).astype(int)
    rows.append(score(exp, "OneClassSVM", to01(Xb), to01(Xk), to01(Xu)))
    print(rows[-1])

    # 3) Autoencoder (benign only); threshold from benign validation flows
    fit, val = train_test_split(benign, test_size=0.15, random_state=SEED)
    ae = build_ae(benign.shape[1])
    ae.fit(fit, fit, validation_data=(val, val), epochs=60, batch_size=512, shuffle=True, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    val_err = recon_error(ae, val)
    eb, ek, eu = recon_error(ae, Xb), recon_error(ae, Xk), recon_error(ae, Xu)
    for p in PERCENTILES:
        th = np.percentile(val_err, p)
        rows.append(score(exp, f"Autoencoder p{p}", (eb > th).astype(int), (ek > th).astype(int), (eu > th).astype(int)))
        print(rows[-1], "threshold =", round(float(th), 4))
    print()

res = pd.DataFrame(rows)[["experiment", "model", "recall_unseen", "recall_known", "FPR", "F1", "MCC"]]
res.to_csv("results/autoencoder.csv", index=False)
print(res.to_string(index=False))
print("\nsaved results/autoencoder.csv")