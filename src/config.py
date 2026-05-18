from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

DEFAULT_DATA_PATH = DATA_DIR / "cyberbullying_sample.csv"
MODEL_PATH = MODEL_DIR / "hyperbolic_tangent_kernel_svm.pkl"
RESULTS_PATH = OUTPUT_DIR / "training_results.txt"

TEXT_COLUMN = "comment_text"
LABEL_COLUMN = "label"

TOXICITY_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
