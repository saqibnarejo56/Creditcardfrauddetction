import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    DEFAULT_DATA_PATH,
    LABEL_COLUMN,
    MODEL_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
    RESULTS_PATH,
    TEXT_COLUMN,
    TOXICITY_COLUMNS,
)


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = [col.strip() for col in df.columns]
    return df


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset must contain a text column named '{TEXT_COLUMN}'. "
            f"Available columns: {list(df.columns)}"
        )

    if LABEL_COLUMN in df.columns:
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
        return df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

    available_toxicity_columns = [col for col in TOXICITY_COLUMNS if col in df.columns]

    if available_toxicity_columns:
        df[LABEL_COLUMN] = df[available_toxicity_columns].max(axis=1).astype(int)
        return df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

    raise ValueError(
        "Dataset must contain either a binary 'label' column "
        "or toxicity columns such as toxic, severe_toxic, obscene, threat, insult, identity_hate."
    )


def build_hyperbolic_tangent_kernel_model() -> Pipeline:
    """
    Only classifier used:

    SVC(kernel="sigmoid")

    This represents Hyperbolic Tangent Kernel.
    Formula:
    K(x, y) = tanh(gamma * x.T * y + coef0)
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=15000,
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                ),
            ),
            (
                "hyperbolic_tangent_kernel_svm",
                SVC(
                    kernel="sigmoid",
                    gamma="scale",
                    coef0=0.0,
                    C=2.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train(data_path: Path) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(data_path)
    df = prepare_target(df)

    X = df[TEXT_COLUMN].astype(str)
    y = df[LABEL_COLUMN].astype(int)

    if y.nunique() != 2:
        raise ValueError("The target label must contain both classes: 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training model using ONLY Hyperbolic Tangent Kernel...")
    model = build_hyperbolic_tangent_kernel_model()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    results = f"""
Cyberbullying Detection Using Hyperbolic Tangent Kernel

STRICT RULE:
Only Hyperbolic Tangent Kernel SVM is used.
No other kernel is used.
No other classifier is used.

CLASSIFIER:
SVC(kernel="sigmoid", gamma="scale", coef0=0.0, C=2.0, class_weight="balanced")

KERNEL FORMULA:
K(x, y) = tanh(gamma * x.T * y + coef0)

DATASET:
{data_path}

TOTAL_RECORDS:
{len(df)}

TRAINING_RECORDS:
{len(X_train)}

TESTING_RECORDS:
{len(X_test)}

ACCURACY:
{accuracy:.4f}

PRECISION:
{precision:.4f}

RECALL:
{recall:.4f}

F1_SCORE:
{f1:.4f}

CLASSIFICATION_REPORT:
{report}

CONFUSION_MATRIX:
{matrix}
""".strip()

    RESULTS_PATH.write_text(results, encoding="utf-8")
    joblib.dump(model, MODEL_PATH)

    print("Training completed successfully.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved results: {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Train cyberbullying detection using only Hyperbolic Tangent Kernel SVM."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to CSV dataset. Default: data/cyberbullying_sample.csv",
    )
    args = parser.parse_args()

    train(Path(args.data))


if __name__ == "__main__":
    main()
