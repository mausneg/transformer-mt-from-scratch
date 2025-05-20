# Transformer-based Machine Translation from Scratch

Proyek ini membangun model penerjemahan mesin berbasis Transformer dari nol menggunakan TensorFlow, dengan REST API (Flask), antarmuka pengguna (Streamlit), dan deployment menggunakan Docker.

## 📌 Daftar Isi
1. [Pendahuluan](#pendahuluan)
2. [Struktur Proyek](#struktur-proyek)
3. [Arsitektur Model](#arsitektur-model)
4. [Alur Training](#alur-training)
5. [API & Antarmuka](#api--antarmuka)
6. [Docker & Deployment](#docker--deployment)
7. [Cara Menjalankan Proyek](#cara-menjalankan-proyek)
8. [Kesimpulan dan Pengembangan Lanjutan](#kesimpulan-dan-pengembangan-lanjutan)
9. [Referensi](#referensi)

---

## 📖 Pendahuluan

Proyek ini dibuat untuk memahami dan membangun model Transformer untuk penerjemahan mesin (machine translation) dari nol. Proyek ini juga mengintegrasikan teknologi modern untuk penyajian model dalam bentuk API dan antarmuka pengguna. Tujuan utamanya adalah membangun pipeline penerjemahan dari Bahasa Indonesia ke Bahasa Inggris menggunakan implementasi Transformer tanpa bantuan library model pre-trained.

**Teknologi yang digunakan:**
- TensorFlow (Model Transformer)
- Flask (REST API)
- Streamlit (User Interface)
- Docker (Deployment)

---

## 📁 Struktur Proyek

```
transformer-mt-from-scratch/
│
├── src/                 # Kode utama model dan utilitas
│   ├── model/           # Implementasi Transformer
│   └── ...
│
├── api/                 # REST API Flask
├── app/                 # Streamlit UI
├── data/                # Dataset mentah dan hasil preprocessing
├── saved_models/        # Model dan tokenizer yang sudah dilatih
├── docker-compose.yml   # Konfigurasi multi-container Docker
├── Dockerfile.api       # Dockerfile untuk API
├── Dockerfile.app       # Dockerfile untuk Streamlit
└── README.md
```

---

## 🧠 Arsitektur Model

Model dibangun mengikuti arsitektur **Transformer encoder-decoder**. Beberapa komponen penting yang diimplementasikan dari nol meliputi:
- Embedding dan positional encoding
- Multi-head self-attention
- Feed-forward networks
- Layer normalization dan masking

Model terdiri dari:
- 4 encoder dan decoder layer
- 8 attention head
- Dimensi model (d_model): 128
- Feed-forward dimensi: 512
- Dropout rate: 0.1

---

## 🔁 Alur Training

Dataset yang digunakan adalah pasangan kalimat Indonesia–Inggris dari [Tatoeba Project](https://tatoeba.org).

**Langkah-langkah:**
1. Preprocessing: lowercase, strip karakter non-alfabet, perluasan kontraksi.
2. Tokenisasi dan padding.
3. Pembuatan dataset encoder-decoder.
4. Pelatihan menggunakan TensorFlow dengan `CustomSchedule` untuk learning rate.

Evaluasi menggunakan:
- **Masked Accuracy**: ~0.68–0.69
- **BLEU Score**: 0.2917

---

## 🌐 API & Antarmuka

### REST API (Flask)
- Endpoint: `POST /translate`
- Format input: JSON (`{"sentence": "apa kabar?"}`)
- Output: JSON hasil terjemahan (`{"translation": "how are you?"}`)

### Streamlit UI
- Input teks Bahasa Indonesia
- Output hasil terjemahan Bahasa Inggris
- Antarmuka sederhana dan intuitif

---

## 🚢 Docker & Deployment

Proyek ini didesain agar mudah dijalankan menggunakan **Docker**:

- `Dockerfile.api`: membangun image untuk Flask API
- `Dockerfile.app`: membangun image untuk Streamlit UI
- `docker-compose.yml`: menghubungkan kedua service tersebut

Untuk menjalankan:

```bash
docker-compose up --build
```

Aplikasi tersedia di:
- Streamlit UI: `http://localhost:8501`
- Flask API: `http://localhost:5000/translate`

---

## ▶️ Cara Menjalankan Proyek

### Persiapan awal
1. Install Git, Docker, dan Git LFS:
   ```bash
   git lfs install
   ```

2. Clone repository dan unduh file besar:
   ```bash
   git clone https://github.com/mausneg/transformer-mt-from-scratch.git
   cd transformer-mt-from-scratch
   git lfs pull
   ```

3. Jalankan aplikasi:
   ```bash
   docker-compose up --build
   ```

### Kebutuhan sistem:
- RAM minimal 4 GB
- Python 3.8+ (jika tidak pakai Docker)
- GPU (opsional untuk training)

---

## ✅ Kesimpulan dan Pengembangan Lanjutan

Proyek ini menunjukkan bahwa membangun model penerjemahan berbasis Transformer dari nol dapat menghasilkan hasil yang cukup baik. Dengan BLEU score ~0.29, sistem dapat menghasilkan terjemahan dasar yang dapat dimanfaatkan sebagai proof of concept.

### Rencana pengembangan:
- Meningkatkan dataset dan pelatihan
- Menggunakan transfer learning (pre-trained encoder/decoder)
- Menambah metrik evaluasi dan validasi lebih lanjut

---

## 🔗 Referensi

- Vaswani et al., “Attention is All You Need” (2017)
- TensorFlow Tutorials: Transformer
- Tatoeba Project (dataset)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Docker Documentation](https://docs.docker.com)
- [Git LFS](https://git-lfs.com)