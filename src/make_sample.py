import pandas as pd

SRC = "data/NF-BoT-IoT-v3.csv"
OUT = "data/botiot_sample.csv"

# Share of each class to keep. Benign and Theft are rare, so keep all of them.
FRAC = {"Benign": 1.0, "Theft": 1.0, "DDoS": 0.014, "DoS": 0.0125, "Reconnaissance": 0.059}

first = True
for i, chunk in enumerate(pd.read_csv(SRC, chunksize=500000)):
    parts = []
    for name, frac in FRAC.items():
        sub = chunk[chunk["Attack"] == name]
        if frac < 1.0:
            sub = sub.sample(frac=frac, random_state=42 + i)
        parts.append(sub)
    out = pd.concat(parts)
    out.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    print("chunk", i, "kept", len(out), flush=True)