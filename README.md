# sentiment_analysis

# 🎭 Text Emotion Detection App

A modern, interactive web application built with **Streamlit** and **Deep Learning (TensorFlow/Keras)** that classifies text into six distinct emotions: **Sadness, Anger, Love, Surprise, Fear, and Joy**. It supports both real-time single text analysis with confidence breakdown charts and batch processing via CSV/TXT file uploads.

---

## 🚀 Features

* **✍️ Single Text Analysis:** Type or paste any sentence, or use quick-sample buttons to instantly predict the underlying emotion.
* **📊 Detailed Probabilities:** View confidence percentages for all 6 emotions visualized through an interactive bar chart and metric cards.
* **📁 Batch File Processing:** Upload CSV or TXT files to analyze multiple texts at once and download the prediction results as a new CSV file.
* **👨‍💻 Developer Sidebar:** Clean sidebar layout showcasing developer credentials and profile links.
* **⚡ Optimized Performance:** Uses `@st.cache_resource` to load the heavy LSTM model and tokenizer efficiently without lag.

---

## 🛠️ Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Deep Learning:** [TensorFlow / Keras](https://www.tensorflow.org/) (LSTM Model)
* **Data Manipulation & Preprocessing:** Pandas, NumPy, Scikit-learn (Joblib tokenizer)

---

## 📂 Project Directory Structure

```text
📦 text-emotion-detection-app/
├── 📜 app.py                  # Main Streamlit application script
├── 🤖 analysis_model.keras    # Trained LSTM/Deep Learning model
├── 📦 tokenizer.pkl           # Saved Scikit-learn/Keras tokenizer
└── 📖 README.md               # Project documentation

```

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally on your system:

### 1. Clone the Repository or Download Files

Make sure all your project files (`app.py`, `analysis_model.keras`, and `tokenizer.pkl`) are in the same folder.

### 2. Install Dependencies

Open your terminal/command prompt and install the required Python packages:

```bash
pip install streamlit tensorflow joblib numpy pandas

```

### 3. Run the Streamlit App

Execute the following command in your terminal:

```bash
streamlit run app.py

```

This will automatically open the web application in your default web browser (usually at `http://localhost:8501`).

---

## 📝 How to Use

1. **Sidebar:** View developer contact and LinkedIn profile information.
2. **Tab 1 (Single Text):**
* Enter a sentence or click one of the sample emotion buttons.
* Click **🔍 Analyze Text** to see the predicted emotion card, confidence score, and probability distribution chart.


3. **Tab 2 (File Upload):**
* Upload a `.csv` file (and select the text column) or a `.txt` file (one sentence per line).
* Click **🚀 Run Batch Prediction** to process the entire file and download the output results instantly.



---

## 👨‍💻 Developer

* **Name:** Ankit Kumar Ojha
* **Email:** [ankitojha1184@gmail.com](https://www.google.com/search?q=mailto%3Aankitojha1184%40gmail.com)
* **LinkedIn:** [Ankit Kumar Ojha Profile](https://www.linkedin.com/in/ankit-kumar-ojha-94835b323?utm_source=share_via&utm_content=profile&utm_medium=member_android)
