# 🛡️ Toxic Comment Detection

This project detects **toxic, offensive, or hateful comments** in real-time using **Machine Learning and Deep Learning** models.  
It includes a **Streamlit web application** where users can enter a comment or upload a CSV file to get predictions.

---

## 📌 Features
- ✅ **Text Preprocessing**: cleaning, tokenization, TF-IDF features  
- ✅ **Baseline Model**: Logistic Regression with TF-IDF  
- ✅ **Deep Learning Model**: BiLSTM for contextual understanding  
- ✅ **Model Comparison**: Accuracy, Precision, Recall, F1-score  
- ✅ **Streamlit Web App**:
  - Single comment prediction  
  - Bulk prediction via CSV upload  
  - Downloadable results  
  - Charts & insights  

---

## 📂 Project Structure
ToxicCommentDetection/
│── app.py # Streamlit app
│── README.md # Project documentation (this file)
│── models/ # Saved models
│ ├── logreg_toxicity.joblib
│ ├── tfidf_vectorizer.joblib
│ ├── bilstm_model.h5
│ ├── bilstm_tokenizer.json
│── sample_comments.csv # Example CSV for bulk testing
│── train.csv # Training dataset (not included in repo)


---

## ⚡ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ToxicCommentDetection.git
   cd ToxicCommentDetection


(Optional) Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate


Install the required dependencies:
pip install streamlit transformers torch torchvision torchaudio pandas numpy scikit-learn tensorflow matplotlib


Run the Streamlit app:
streamlit run app.py

📊 Model Performance
Model	             Accuracy	    Precision	Recall	F1-Score
Logistic Regression	0.83	    0.79	        0.75	0.77
BiLSTM            	0.87	    0.84	        0.82	0.83
