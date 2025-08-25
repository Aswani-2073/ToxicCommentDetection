import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="🛡️ Comment Toxicity Detection", layout="wide", page_icon="🛡️")

MODEL_NAME = "unitary/toxic-bert"

# --------- Load model once and cache ---------
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,   # ✅ force float32 so no meta tensors
        device_map=None              # ✅ ensure model loads on CPU
    )
    model.eval()
    return tokenizer, model

# --------- Predict function ---------
def predict_toxicity(texts):
    tokenizer, model = load_model()
    inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits).detach().cpu().numpy()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return scores, labels

# --------- Utility for downloads ---------
def to_downloadable_csv(df: pd.DataFrame, fname: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=fname, mime="text/csv")

# --------- UI ---------
st.title("🛡️ Comment Toxicity Detection App")
st.write("Detect toxic, offensive, or hateful comments in real-time using Deep Learning.")

st.sidebar.header("Options")
mode = st.sidebar.radio("Choose Mode", ["Single Comment", "Bulk Prediction (CSV)"])

if mode == "Single Comment":
    user_input = st.text_area("✍️ Enter a comment:")
    if st.button("Predict"):
        if user_input.strip():
            scores, labels = predict_toxicity([user_input])
            results = dict(zip(labels, scores[0]))
            st.subheader("Prediction Results")
            df = pd.DataFrame(results.items(), columns=["Label", "Score"]).sort_values("Score", ascending=False)
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index("Label"))
        else:
            st.warning("Please enter a comment.")

else:
    uploaded_file = st.file_uploader("📂 Upload a CSV file with a 'comment' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "comment" in df.columns:
            scores, labels = predict_toxicity(df["comment"].astype(str).tolist())
            score_df = pd.DataFrame(scores, columns=labels)
            result_df = pd.concat([df, score_df], axis=1)
            st.write("### Bulk Prediction Results")
            st.dataframe(result_df.head(), use_container_width=True)
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv, "toxic_predictions.csv", "text/csv")
        else:
            st.error("CSV must contain a 'comment' column.")

st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit & Transformers")
