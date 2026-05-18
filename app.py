from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hyperbolic_tangent_kernel_svm.pkl"
DATA_PATH = BASE_DIR / "data" / "cyberbullying_sample.csv"


st.set_page_config(
    page_title="Cyberbullying Detection",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["comment_text", "label"])
    return pd.read_csv(DATA_PATH)


def confidence_from_decision_score(score: float) -> float:
    """
    Confidence-style score calculated from SVM decision score.
    The prediction still comes only from Hyperbolic Tangent Kernel SVM.
    """
    return min(99.0, max(50.0, 50.0 + abs(score) * 45.0))


def get_label_name(value: int) -> str:
    return "Cyberbullying" if int(value) == 1 else "Not Cyberbullying"


def get_similar_dataset_comments(model, user_comment: str, dataset: pd.DataFrame, top_n: int = 5):
    if dataset.empty or "comment_text" not in dataset.columns:
        return pd.DataFrame()

    tfidf = model.named_steps["tfidf"]
    dataset_texts = dataset["comment_text"].astype(str).tolist()

    user_vector = tfidf.transform([user_comment])
    dataset_vectors = tfidf.transform(dataset_texts)

    similarity_scores = cosine_similarity(user_vector, dataset_vectors)[0]
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    rows = []
    for index in top_indices:
        label_value = int(dataset.iloc[index]["label"]) if "label" in dataset.columns else -1
        rows.append(
            {
                "Similar Dataset Comment": dataset.iloc[index]["comment_text"],
                "Dataset Label": get_label_name(label_value) if label_value in [0, 1] else "Unknown",
                "Similarity": f"{similarity_scores[index] * 100:.2f}%",
            }
        )

    return pd.DataFrame(rows)


model = load_model()
dataset = load_dataset()

st.title("🛡️ Cyberbullying Detection System")

if model is None:
    st.error("Model file not found. Train the model first.")
    st.code("python src/train_model.py", language="bash")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

left_col, right_col = st.columns([1.05, 1])

with left_col:
    st.subheader("Enter Comment")

    comment = st.text_area(
        "Write any comment here:",
        placeholder="Example: You are stupid and useless.",
        height=180,
    )

    predict_btn = st.button("Predict Comment", use_container_width=True)

with right_col:
    st.subheader("Prediction Result")

    if predict_btn:
        if not comment.strip():
            st.warning("Please enter a comment first.")
        else:
            prediction = int(model.predict([comment])[0])
            decision_score = float(model.decision_function([comment])[0])
            predicted_label = get_label_name(prediction)
            confidence = confidence_from_decision_score(decision_score)

            if predicted_label == "Cyberbullying":
                st.error("Prediction: Cyberbullying")
                explanation = "The comment is closer to harmful or insulting language patterns learned from the dataset."
            else:
                st.success("Prediction: Not Cyberbullying")
                explanation = "The comment is closer to safe or respectful language patterns learned from the dataset."

            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{confidence:.2f}%")
            m2.metric("Decision Score", f"{decision_score:.4f}")

            st.write(explanation)

            cyber_score = confidence if predicted_label == "Cyberbullying" else 100 - confidence
            safe_score = confidence if predicted_label == "Not Cyberbullying" else 100 - confidence

            graph_df = pd.DataFrame(
                {
                    "Prediction Score": [safe_score, cyber_score],
                },
                index=["Not Cyberbullying", "Cyberbullying"],
            )

            st.write("Real-Time Prediction Graph")
            st.bar_chart(graph_df)

            st.session_state.history.append(
                {
                    "Comment": comment[:80] + ("..." if len(comment) > 80 else ""),
                    "Prediction": predicted_label,
                    "Confidence": f"{confidence:.2f}%",
                    "Decision Score": round(decision_score, 4),
                }
            )

            st.subheader("Comment Comparison With Dataset")
            similar_df = get_similar_dataset_comments(model, comment, dataset, top_n=5)

            if not similar_df.empty:
                st.dataframe(similar_df, use_container_width=True, hide_index=True)
            else:
                st.info("Dataset comparison not available.")

st.divider()

st.subheader("Recent Predictions")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    score_history = []
    for item in st.session_state.history:
        raw_value = str(item["Confidence"]).replace("%", "")
        try:
            score_history.append(float(raw_value))
        except ValueError:
            score_history.append(0)

    chart_df = pd.DataFrame(
        {"Confidence": score_history},
        index=[f"Comment {i + 1}" for i in range(len(score_history))],
    )

    st.write("Confidence History")
    st.line_chart(chart_df)
else:
    st.info("No prediction yet. Enter a comment and click Predict.")
