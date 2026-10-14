import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(
    page_title="Text Emotion Detection", 
    page_icon="🎭",
    layout="centered"
)

# Emotion Mapping
EMOTION_MAP = {
    0: ("Sadness 😢", "#1E88E5"),
    1: ("Anger 😡", "#E53935"),
    2: ("Love ❤️", "#D81B60"),
    3: ("Surprise 😲", "#FB8C00"),
    4: ("Fear 😨", "#8E24AA"),
    5: ("Joy 😊", "#4CAF50"),
}

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("analysis_model.keras")
    tokenizer = joblib.load("tokenizer.pkl")
    return model, tokenizer

model, tokenizer = load_assets()

# --- Sidebar ---
with st.sidebar:
    st.header("👨‍💻 Developer Info")
    st.write("**Name:** Ankit Kumar Ojha")
    st.write("**Email:** [ankitojha1184@gmail.com](mailto:ankitojha1184@gmail.com)")
    st.markdown("[🔗 LinkedIn Profile](https://www.linkedin.com/in/ankit-kumar-ojha-94835b323?utm_source=share_via&utm_content=profile&utm_medium=member_android)")
    st.divider()
    st.caption("Built with TensorFlow & Streamlit")

# --- Main UI ---
st.title("🎭 Text Emotion Detection")
st.write("Analyze emotions from a single text or upload a file for batch prediction.")

# Navigation Tabs: Single Text vs File Upload
tab1, tab2 = st.tabs(["✍️ Single Text", "📁 File Upload (Batch)"])

# --- TAB 1: Single Text Analysis ---
with tab1:
    st.subheader("💡 Try Sample Inputs:")
    col1, col2, col3 = st.columns(3)

    sample_text = ""
    if col1.button("😊 Joy Sample"):
        sample_text = "I am so glad and thankful for this wonderful day!"
    if col2.button("😢 Sadness Sample"):
        sample_text = "I feel completely heartbroken and alone."
    if col3.button("😡 Anger Sample"):
        sample_text = "This is totally unacceptable and makes me extremely furious!"

    user_text = st.text_area(
        "Your Text:", 
        value=sample_text if sample_text else "", 
        placeholder="e.g., I feel so happy and excited today!",
        height=120
    )

    if st.button("🔍 Analyze Text", type="primary"):
        if user_text.strip() == "":
            st.warning("Please enter or select some text first.")
        else:
            cleaned_text = user_text.lower()
            max_length = 100
            seq = tokenizer.texts_to_sequences([cleaned_text])
            padded_seq = pad_sequences(seq, maxlen=max_length, padding="pre")

            prediction = model.predict(padded_seq)[0]
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            emotion_label, color_code = EMOTION_MAP.get(predicted_class, ("Unknown", "#757575"))

            st.divider()

            # Display Result Card
            st.markdown(
                f"""
                <div style="background-color: {color_code}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
                    <h2 style="margin: 0; color: white;">Predicted Emotion: {emotion_label}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("") 
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Top Emotion", emotion_label)
            m_col2.metric("Confidence Score", f"{confidence * 100:.2f}%")

            st.subheader("📊 Emotion Breakdown")
            labels = [EMOTION_MAP[i][0] for i in range(len(EMOTION_MAP))]
            df_probs = pd.DataFrame({
                "Emotion": labels,
                "Probability (%)": prediction * 100
            }).set_index("Emotion")

            st.bar_chart(df_probs)

# --- TAB 2: File Upload (CSV/TXT) ---
with tab2:
    st.subheader("📂 Upload CSV or Text File")
    st.info("Aapka CSV file hoga toh usme ek column hona chahiye jisme sentences/text ho.")
    
    uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                # Assume first text column or user can select
                text_col = st.selectbox("Select the column containing text:", df.columns)
                texts_to_predict = df[text_col].dropna().astype(str).tolist()
            else:
                # Text file: each line is a sentence
                stringio = uploaded_file.getvalue().decode("utf-8")
                texts_to_predict = [line.strip() for line in stringio.split("\n") if line.strip()]

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing predictions..."):
                    cleaned_texts = [t.lower() for t in texts_to_predict]
                    seqs = tokenizer.texts_to_sequences(cleaned_texts)
                    padded_seqs = pad_sequences(seqs, maxlen=100, padding="pre")

                    predictions = model.predict(padded_seqs)
                    predicted_classes = np.argmax(predictions, axis=1)
                    confidences = np.max(predictions, axis=1)

                    results = []
                    for t, cls, conf in zip(texts_to_predict, predicted_classes, confidences):
                        label, _ = EMOTION_MAP.get(cls, ("Unknown", "gray"))
                        results.append({
                            "Text": t,
                            "Predicted Emotion": label,
                            "Confidence (%)": round(conf * 100, 2)
                        })

                    result_df = pd.DataFrame(results)
                    st.success("Prediction Completed!")
                    st.dataframe(result_df, use_container_width=True)

                    # Download button for results
                    csv_data = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="emotion_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")