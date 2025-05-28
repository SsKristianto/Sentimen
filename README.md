# 🎯 Sistem Analisis Sentimen Indonesia

Sistem analisis sentimen otomatis untuk teks berbahasa Indonesia menggunakan Machine Learning dan Deep Learning.

## ✨ Fitur Utama

- **📊 Preprocessing Data Komprehensif**: Pembersihan teks, normalisasi bahasa Indonesia, tokenisasi, dan stemming
- **🤖 Multiple Models**: Naive Bayes dan LSTM untuk perbandingan performa
- **🔧 Feature Engineering**: TF-IDF, Bag of Words, Word2Vec, dan FastText
- **📈 Evaluasi Lengkap**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, dan Cross-validation
- **💻 Web Interface**: Aplikasi web interaktif dengan Streamlit
- **📝 Real-time Analysis**: Analisis sentimen teks secara langsung
- **📊 Visualisasi**: Grafik dan chart untuk analisis data

## 🚀 Quick Start

### 1. Persiapan Environment

```bash
# Clone atau download project
git clone <repository-url>
cd sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup NLTK Data (PENTING!)

**Pilihan A - Setup Otomatis (Recommended):**
```bash
python setup_nltk.py
```

**Pilihan B - Setup Manual:**
```python
import nltk
import ssl

# Handle SSL issues (jika ada)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### 3. Persiapan Dataset

Pastikan file `Dataset.csv` tersedia dengan struktur kolom:
- `userName`: Nama pengguna
- `content`: Konten ulasan/komentar
- `score`: Skor rating (1-5)
- `at`: Tanggal ulasan
- `appVersion`: Versi aplikasi

### 4. Menjalankan Aplikasi

**Pilihan A - Startup Script (Recommended):**
```bash
python run_app.py
```

**Pilihan B - Direct Streamlit:**
```bash
streamlit run streamlit_app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

## 🛠️ Troubleshooting

### Masalah NLTK Data

Jika Anda mendapat error seperti:
```
Resource punkt_tab not found. Please use the NLTK Downloader...
```

**Solusi:**
1. Jalankan `python setup_nltk.py`
2. Atau jalankan setup manual di atas
3. Restart aplikasi

### Masalah SSL Certificate

Jika download NLTK gagal karena SSL:
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### Masalah Return Statement

Jika Anda mendapat error:
```
"return" can be used only within a function
```

**Solusi:**
1. Jalankan fix script: `python fix_app.py`
2. Atau ganti `return` dengan `st.stop()` dalam Streamlit
3. Restructure code dengan if-else blocks

### Masalah Button Duplicate ID

Jika ada error duplicate button ID:
```
StreamlitDuplicateElementId: Multiple button elements with same ID
```

**Solusi:**
Tambahkan unique `key` parameter:
```python
st.button("Text", key="unique_key_name")
```

## 📁 Struktur Project

```
sentiment/
├── Dataset.csv                     # Dataset mentah
├── sentiment_analysis_system.py    # Core system class
├── streamlit_app.py               # Web application
├── requirements.txt               # Dependencies
├── README.md                     # Documentation ini
├── docs/
│   └── documentation.md          # Dokumentasi detail
├── models/                       # Model yang sudah ditraining
│   ├── naive_bayes_tfidf.pkl
│   ├── lstm_model.h5
│   ├── tokenizer.pkl
│   ├── tfidf_vectorizer.pkl
│   └── ...
└── dataset_sudah/               # Dataset yang sudah diproses
    ├── processed_dataset_1000.csv
    └── ...
```

## 🎮 Cara Penggunaan

### 1. Preprocessing Data

1. Buka menu **"⚙️ Preprocessing Data"**
2. Pilih jumlah data yang ingin diproses (500, 1000, 5000, 10000, atau semua)
3. Klik **"🚀 Mulai Preprocessing"**
4. Lihat hasil statistik dan visualisasi preprocessing
5. Data yang sudah diproses akan disimpan di folder `dataset_sudah/`

### 2. Training Model

1. Pastikan data sudah diproses terlebih dahulu
2. Buka menu **"🤖 Train Model"**
3. Pilih model yang ingin ditraining:
   - ✅ Naive Bayes (dengan TF-IDF atau Bag of Words)
   - ✅ LSTM
4. Atur parameter training (test size, CV folds, random state)
5. Klik **"🚀 Mulai Training"**
6. Lihat hasil evaluasi model (accuracy, precision, recall, F1-score)
7. Model akan disimpan di folder `models/`

### 3. Visualisasi Dataset

1. Buka menu **"📊 Visualisasi Dataset"**
2. Lihat berbagai visualisasi:
   - Distribusi sentimen
   - Analisis panjang teks
   - Word cloud per sentimen
   - Trend sentimen dari waktu ke waktu
   - Top words analysis

### 4. Analisis Teks Real-time

1. Buka menu **"📝 Text Analysis"**
2. Pilih model untuk prediksi (Naive Bayes atau LSTM)
3. Masukkan teks yang ingin dianalisis atau gunakan sample teks
4. Klik **"🔍 Analisis Sentimen"**
5. Lihat hasil:
   - Prediksi sentimen (Positif/Negatif/Netral)
   - Confidence score
   - Distribusi probabilitas
   - Kata-kata penting (untuk Naive Bayes)
   - Teks setelah preprocessing

### 5. Analisis Batch

1. Di menu **"📝 Text Analysis"**, scroll ke bagian **"📊 Analisis Batch"**
2. Upload file CSV dengan kolom `content`
3. Klik **"🚀 Analisis Semua"**
4. Download hasil analisis dalam format CSV

## 🔧 Technical Details

### Preprocessing Pipeline

1. **Text Cleaning**: 
   - Remove URLs, email, mentions, hashtags
   - Remove special characters and numbers
   - Convert to lowercase

2. **Normalization**: 
   - Convert Indonesian slang words to formal words
   - Handle common abbreviations

3. **Tokenization & Stopwords Removal**:
   - Using NLTK and Sastrawi
   - Remove Indonesian stopwords

4. **Stemming**:
   - Using Sastrawi stemmer
   - Convert words to root form

### Feature Engineering

1. **TF-IDF**: Term Frequency-Inverse Document Frequency
2. **Bag of Words**: Word frequency representation
3. **Word2Vec**: Dense word embeddings (100 dimensions)
4. **FastText**: Subword-based embeddings (100 dimensions)

### Models

1. **Naive Bayes**:
   - Multinomial Naive Bayes
   - Works with TF-IDF or Bag of Words features
   - Fast training and prediction

2. **LSTM**:
   - Sequential neural network
   - Embedding layer (128 dimensions)
   - LSTM layer (64 units)
   - Dense layers with dropout
   - Categorical crossentropy loss

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positive / (True positive + False positive)
- **Recall**: True positive / (True positive + False negative)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results
- **Cross-Validation**: 5-fold stratified cross-validation

## 📊 Performance

Sistem ini telah dioptimalkan untuk:
- ✅ High accuracy pada teks bahasa Indonesia
- ✅ Robust preprocessing untuk berbagai jenis teks
- ✅ Scalable untuk dataset besar (tested up to 50K+ samples)
- ✅ Fast inference untuk real-time analysis
- ✅ User-friendly interface

## 🛠️ Troubleshooting

### Common Issues

1. **Import Error**: 
   ```bash
   pip install -r requirements.txt
   ```

2. **NLTK Data Missing**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

3. **Memory Error (Large Dataset)**:
   - Reduce sample size in preprocessing
   - Use smaller batch size for LSTM training

4. **Model Not Found**:
   - Ensure models are trained first
   - Check `models/` directory exists

### Performance Optimization

1. **For Large Datasets**:
   - Process data in batches
   - Use sampling for initial experiments
   - Consider using more powerful hardware

2. **For Faster Training**:
   - Reduce max_features in vectorizers
   - Use smaller embedding dimensions
   - Reduce LSTM units

## 🔮 Future Enhancements

- [ ] Support untuk model transformer (BERT, RoBERTa)
- [ ] Multi-label sentiment classification
- [ ] Active learning untuk improvement
- [ ] REST API untuk integrasi
- [ ] Real-time monitoring dashboard
- [ ] Automated model retraining
- [ ] Support untuk bahasa daerah Indonesia

## 📚 Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **tensorflow**: Deep learning

### Text Processing
- **nltk**: Natural language toolkit
- **Sastrawi**: Indonesian language library
- **gensim**: Word embeddings

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive charts
- **wordcloud**: Word cloud generation

## 📞 Support

Untuk pertanyaan atau masalah:
1. Check dokumentasi di `docs/documentation.md`
2. Review common issues di README ini
3. Create issue di repository GitHub
4. Contact tim pengembang

## 📄 License

Project ini dilisensikan di bawah MIT License - lihat file LICENSE untuk detail.

## 🤝 Contributing

Kontribusi sangat diterima! Silakan:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push ke branch
5. Create Pull Request

## 📈 Changelog

### v1.0.0 (Current)
- ✅ Initial release
- ✅ Complete preprocessing pipeline
- ✅ Naive Bayes dan LSTM models
- ✅ Web interface dengan Streamlit
- ✅ Comprehensive evaluation metrics
- ✅ Real-time dan batch analysis
- ✅ Visualization dashboard

---

**🎯 Happy Analyzing! 🚀**