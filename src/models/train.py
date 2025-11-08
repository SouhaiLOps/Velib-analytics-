# src/models/train.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

DATA = Path("data/raw/sample.csv")

def load_data():
    if not DATA.exists():
        # dataset jouet plus grand (bruit léger)
        x = np.arange(1, 51)  # 50 points
        y = 2 * x + np.random.normal(0, 0.5, size=x.shape)
        df = pd.DataFrame({"x": x, "y": y})
        DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA, index=False)
    return pd.read_csv(DATA)

def main():
    df = load_data()
    X = df[["x"]]
    y = df["y"]

    # assure au moins 2 points en test
    test_size = max(0.2, 2 / len(df))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

    model = LinearRegression().fit(Xtr, ytr)
    ypred = model.predict(Xte)

    mae = mean_absolute_error(yte, ypred)
    r2 = r2_score(yte, ypred) if len(yte) >= 2 else float("nan")

    print(f"n_train={len(Xtr)}, n_test={len(Xte)}")
    print(f"MAE test: {mae:.4f}")
    print(f"R2 test: {r2:.4f}" if len(Xte) >= 2 else "R2 test: nan (test set < 2)")

if __name__ == "__main__":
    main() 
