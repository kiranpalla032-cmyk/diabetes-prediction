import argparse
import joblib
import numpy as np

from utils import FEATURES

def main(args):
    model = joblib.load(args.model)
    if args.input:
        # Expect a comma-separated string of 8 values in correct order
        values = [float(x) for x in args.input.split(",")]
        if len(values) != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} values in --input, got {len(values)}")
        X = np.array(values).reshape(1, -1)
        pred = model.predict(X)[0]
        proba = None
        try:
            proba = model.predict_proba(X)[0,1]
        except Exception:
            pass
        print("Prediction (1 = diabetes likely, 0 = not likely):", int(pred))
        if proba is not None:
            print("Probability of diabetes:", float(proba))
    else:
        print("Please provide --input with 8 comma-separated values in the following order:")
        print(",".join(FEATURES))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.joblib", help="Path to trained model file")
    parser.add_argument("--input", type=str, help="Comma-separated feature values in order of FEATURES")
    args = parser.parse_args()
    main(args)
