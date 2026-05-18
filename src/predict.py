import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Please run: python src/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def predict_comment(comment: str):
    model = load_model()
    prediction = int(model.predict([comment])[0])
    decision_score = float(model.decision_function([comment])[0])
    label = "Cyberbullying" if prediction == 1 else "Not Cyberbullying"
    confidence = min(99.0, max(50.0, 50.0 + abs(decision_score) * 45.0))
    return label, decision_score, confidence


def main():
    print("Cyberbullying Detection Using Hyperbolic Tangent Kernel")
    print("Only classifier used: SVC(kernel='sigmoid')")
    print("-" * 60)

    while True:
        comment = input("\nEnter a comment, or type exit: ").strip()

        if comment.lower() in {"exit", "quit"}:
            print("Program closed.")
            break

        if not comment:
            print("Please enter a valid comment.")
            continue

        label, score, confidence = predict_comment(comment)

        print(f"Prediction: {label}")
        print(f"Decision Score: {score:.4f}")
        print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
