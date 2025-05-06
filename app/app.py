import streamlit as st
import requests

st.set_page_config(page_title="Transformer Machine Translation", layout="centered")

st.title("🧠 Transformer Machine Translation")
st.markdown("Masukkan kalimat dalam bahasa sumber, lalu klik tombol untuk menerjemahkan menggunakan model Transformer buatan sendiri.")

input_text = st.text_area("Kalimat sumber", height=150)

if st.button("Terjemahkan"):
    if not input_text.strip():
        st.warning("Tolong masukkan kalimat terlebih dahulu.")
    else:
        try:
            response = requests.post(
                "http://flask-api:5000/translate",
                json={"text": input_text}
            )

            if response.status_code == 200:
                translated_text = response.json()["translation"]
                st.success("Hasil Terjemahan:")
                st.write(translated_text)
            else:
                st.error(f"Terjadi kesalahan dari API: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Gagal terhubung ke API. Pastikan Flask server berjalan di Docker container.")
