import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import string
import os
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Word Embeddings
from gensim.models import Word2Vec, FastText
import gensim.downloader as api

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Page configuration
st.set_page_config(
    page_title="Analisis Sentimen Indonesia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'vectorizers' not in st.session_state:
    st.session_state.vectorizers = {}

# Create necessary directories
os.makedirs('dataset_sudah', exist_ok=True)
os.makedirs('models', exist_ok=True)

class TextPreprocessor:
    def __init__(self):
        # Initialize Sastrawi components
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = stopword_factory.create_stop_word_remover()
        
        # Load normalization dictionary
        self.normalization_dict = self.load_normalization_dict()
    
    def load_normalization_dict(self):
        """Load Indonesian text normalization dictionary"""
        return {
            'yg': 'yang', 'dgn': 'dengan', 'dr': 'dari', 'utk': 'untuk',
            'krn': 'karena', 'sdh': 'sudah', 'blm': 'belum', 'hrs': 'harus',
            'sy': 'saya', 'km': 'kamu', 'dia': 'dia', 'mrk': 'mereka',
            'bgm': 'bagaimana', 'gmn': 'gimana', 'knp': 'kenapa', 'kpn': 'kapan',
            'dmn': 'dimana', 'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak',
            'bs': 'bisa', 'tp': 'tapi', 'dpt': 'dapat', 'jd': 'jadi',
            'bgd': 'banget', 'bgt': 'banget', 'bener': 'benar', 'bnr': 'benar',
            'ud': 'sudah', 'dah': 'sudah', 'emg': 'memang', 'memng': 'memang',
            'skrg': 'sekarang', 'skg': 'sekarang', 'trs': 'terus', 'trs': 'terus',
            'sm': 'sama', 'aja': 'saja', 'aj': 'saja', 'org': 'orang',
            'orng': 'orang', 'byk': 'banyak', 'bnyk': 'banyak', 'lg': 'lagi',
            'lgi': 'lagi', 'wkt': 'waktu', 'wktu': 'waktu', 'smpe': 'sampai',
            'smp': 'sampai', 'hbis': 'habis', 'abis': 'habis', 'mo': 'mau',
            'mw': 'mau', 'tau': 'tahu', 'g': 'tidak', 'ngga': 'tidak',
            'nggak': 'tidak', 'enggak': 'tidak', 'ndak': 'tidak'
        }
    
    def clean_text(self, text):
        """Clean text from special characters, URLs, and emojis"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    def normalize_text(self, text):
        """Normalize Indonesian slang words"""
        if pd.isna(text) or text == "":
            return ""
        
        words = text.split()
        normalized_words = []
        
        for word in words:
            if word in self.normalization_dict:
                normalized_words.append(self.normalization_dict[word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def tokenize_and_remove_stopwords(self, text):
        """Tokenize and remove stopwords"""
        if pd.isna(text) or text == "":
            return ""
        
        # Remove stopwords using Sastrawi
        text = self.stopword_remover.remove(text)
        return text
    
    def stem_text(self, text):
        """Apply stemming using Sastrawi"""
        if pd.isna(text) or text == "":
            return ""
        
        return self.stemmer.stem(text)
    
    def preprocess_text(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.normalize_text(text)
        text = self.tokenize_and_remove_stopwords(text)
        text = self.stem_text(text)
        return text

class SentimentLabeler:
    def __init__(self):
        # Simple keyword-based labeling for demo
        self.positive_keywords = [
            'bagus', 'baik', 'suka', 'senang', 'puas', 'mantap', 'hebat',
            'keren', 'oke', 'recommended', 'memuaskan', 'excellent', 'good',
            'nice', 'amazing', 'wonderful', 'perfect', 'love', 'like',
            'happy', 'satisfied', 'great', 'awesome', 'fantastic'
        ]
        
        self.negative_keywords = [
            'buruk', 'jelek', 'tidak', 'benci', 'kecewa', 'marah', 'sedih',
            'bad', 'terrible', 'awful', 'hate', 'disappointed', 'angry',
            'sad', 'poor', 'worst', 'horrible', 'disgusting', 'annoying',
            'boring', 'useless', 'waste', 'regret', 'problem', 'error'
        ]
    
    def label_sentiment(self, text):
        """Label sentiment based on keywords"""
        if pd.isna(text) or text == "":
            return 'neutral'
        
        text = str(text).lower()
        
        positive_count = sum(1 for word in self.positive_keywords if word in text)
        negative_count = sum(1 for word in self.negative_keywords if word in text)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Processing Data", "Train Model", "Visualisasi Dataset", "Text Analysis", "Tentang"]
)

# Main Title
st.markdown('<h1 class="main-header">🎯 Sistem Analisis Sentimen Bahasa Indonesia</h1>', unsafe_allow_html=True)

# PAGE 1: PROCESSING DATA
if page == "Processing Data":
    st.markdown('<h2 class="section-header">📝 Processing Data</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.write("**Dataset Overview:**")
        st.write(f"Shape: {df.shape}")
        st.write(df.head())
        
        # Data selection
        st.markdown("### Pilih Jumlah Data untuk Diproses")
        data_options = {
            "1000": 1000,
            "5000": 5000,
            "10000": 10000,
            "Semua Data": len(df)
        }
        
        selected_size = st.selectbox("Jumlah Data:", list(data_options.keys()))
        n_samples = data_options[selected_size]
        
        if st.button("🔄 Mulai Processing"):
            with st.spinner("Processing data..."):
                # Sample data
                if n_samples < len(df):
                    df_sample = df.sample(n=n_samples, random_state=42)
                else:
                    df_sample = df.copy()
                
                # Initialize processors
                preprocessor = TextPreprocessor()
                labeler = SentimentLabeler()
                
                # Assume text column is named 'text' or first column
                text_column = 'text' if 'text' in df_sample.columns else df_sample.columns[0]
                
                # Create processing results
                processing_results = []
                
                # Step 1: Labeling
                st.write("**Step 1: Labeling Sentiment**")
                df_sample['sentiment'] = df_sample[text_column].apply(labeler.label_sentiment)
                sentiment_counts = df_sample['sentiment'].value_counts()
                
                # Display sentiment distribution
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Distribusi Sentiment:")
                    st.write(sentiment_counts)
                with col2:
                    fig_pie = px.pie(values=sentiment_counts.values, 
                                   names=sentiment_counts.index,
                                   title="Distribusi Sentiment")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Step 2: Text Cleaning
                st.write("**Step 2: Pembersihan Teks**")
                df_sample['cleaned_text'] = df_sample[text_column].apply(preprocessor.clean_text)
                
                # Show before/after examples
                examples = df_sample.head(3)
                for i, row in examples.iterrows():
                    st.write(f"**Original:** {row[text_column][:100]}...")
                    st.write(f"**Cleaned:** {row['cleaned_text'][:100]}...")
                    st.write("---")
                
                # Step 3: Normalization
                st.write("**Step 3: Normalisasi Kata**")
                df_sample['normalized_text'] = df_sample['cleaned_text'].apply(preprocessor.normalize_text)
                
                # Step 4: Tokenization and Stopword Removal
                st.write("**Step 4: Tokenisasi dan Penghapusan Stopwords**")
                df_sample['tokenized_text'] = df_sample['normalized_text'].apply(preprocessor.tokenize_and_remove_stopwords)
                
                # Step 5: Stemming
                st.write("**Step 5: Stemming**")
                df_sample['stemmed_text'] = df_sample['tokenized_text'].apply(preprocessor.stem_text)
                
                # Final processed text
                df_sample['processed_text'] = df_sample['stemmed_text']
                
                # Remove empty texts
                df_sample = df_sample[df_sample['processed_text'].str.len() > 0]
                
                # Save processed data
                processed_filename = f"processed_data_{n_samples}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                processed_path = os.path.join('dataset_sudah', processed_filename)
                df_sample.to_csv(processed_path, index=False)
                
                # Store in session state
                st.session_state.processed_data = df_sample
                
                st.success(f"✅ Data berhasil diproses dan disimpan ke {processed_path}")
                
                # Show processing statistics
                st.markdown("### 📊 Statistik Processing")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Data Diproses", len(df_sample))
                with col2:
                    avg_length_before = df_sample[text_column].str.len().mean()
                    avg_length_after = df_sample['processed_text'].str.len().mean()
                    st.metric("Rata-rata Panjang Teks", f"{avg_length_after:.1f}", 
                             f"{avg_length_after - avg_length_before:.1f}")
                with col3:
                    st.metric("Data Valid", len(df_sample))
                
                # Show final processed data
                st.markdown("### 📋 Data Hasil Processing")
                display_columns = ['processed_text', 'sentiment']
                if len(df_sample) > 100:
                    st.write(df_sample[display_columns].head(100))
                    st.info(f"Menampilkan 100 data teratas dari {len(df_sample)} total data")
                else:
                    st.write(df_sample[display_columns])

# PAGE 2: TRAIN MODEL
elif page == "Train Model":
    st.markdown('<h2 class="section-header">🤖 Train Model</h2>', unsafe_allow_html=True)
    
    # Check if processed data exists
    processed_files = [f for f in os.listdir('dataset_sudah') if f.endswith('.csv')]
    
    if not processed_files and st.session_state.processed_data is None:
        st.warning("⚠️ Belum ada data yang diproses. Silakan lakukan processing data terlebih dahulu.")
    else:
        # Data selection
        data_source = st.radio(
            "Pilih sumber data:",
            ["Gunakan data dari session", "Load dari file yang tersimpan"]
        )
        
        df_processed = None
        
        if data_source == "Gunakan data dari session" and st.session_state.processed_data is not None:
            df_processed = st.session_state.processed_data
        elif data_source == "Load dari file yang tersimpan":
            if processed_files:
                selected_file = st.selectbox("Pilih file:", processed_files)
                df_processed = pd.read_csv(os.path.join('dataset_sudah', selected_file))
            else:
                st.error("Tidak ada file processed yang tersedia.")
        
        if df_processed is not None:
            st.write(f"**Data Shape:** {df_processed.shape}")
            st.write("**Distribusi Sentiment:**")
            st.write(df_processed['sentiment'].value_counts())
            
            # Feature Engineering Section
            st.markdown("### 🔧 Feature Engineering")
            
            feature_methods = st.multiselect(
                "Pilih metode representasi fitur:",
                ["TF-IDF", "Bag of Words", "Word2Vec"],
                default=["TF-IDF", "Word2Vec"]
            )
            
            # Model Selection
            st.markdown("### 🎯 Model Selection")
            
            model_types = st.multiselect(
                "Pilih model untuk training:",
                ["Naive Bayes", "LSTM"],
                default=["Naive Bayes", "LSTM"]
            )
            
            if st.button("🚀 Start Training"):
                with st.spinner("Training models..."):
                    # Prepare data
                    X = df_processed['processed_text'].fillna('')
                    y = df_processed['sentiment']
                    
                    # Encode labels
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                    )
                    
                    results = {}
                    
                    # Feature Engineering and Model Training
                    for feature_method in feature_methods:
                        st.markdown(f"#### 🔧 {feature_method} Feature Engineering")
                        
                        if feature_method == "TF-IDF":
                            # TF-IDF Vectorization
                            tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                            X_train_tfidf = tfidf.fit_transform(X_train)
                            X_test_tfidf = tfidf.transform(X_test)
                            
                            # Save vectorizer
                            joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
                            st.session_state.vectorizers['tfidf'] = tfidf
                            
                            # Train models with TF-IDF
                            for model_type in model_types:
                                if model_type == "Naive Bayes":
                                    # Naive Bayes with TF-IDF
                                    nb_model = MultinomialNB()
                                    nb_model.fit(X_train_tfidf, y_train)
                                    
                                    # Predictions
                                    y_pred = nb_model.predict(X_test_tfidf)
                                    
                                    # Evaluation
                                    accuracy = accuracy_score(y_test, y_pred)
                                    precision = precision_score(y_test, y_pred, average='weighted')
                                    recall = recall_score(y_test, y_pred, average='weighted')
                                    f1 = f1_score(y_test, y_pred, average='weighted')
                                    
                                    # Cross-validation
                                    cv_scores = cross_val_score(nb_model, X_train_tfidf, y_train, cv=5)
                                    
                                    results[f"NB_TFIDF"] = {
                                        'model': nb_model,
                                        'accuracy': accuracy,
                                        'precision': precision,
                                        'recall': recall,
                                        'f1': f1,
                                        'cv_mean': cv_scores.mean(),
                                        'cv_std': cv_scores.std(),
                                        'y_test': y_test,
                                        'y_pred': y_pred,
                                        'labels': le.classes_
                                    }
                                    
                                    # Save model
                                    joblib.dump(nb_model, 'models/naive_bayes_tfidf.pkl')
                                    st.session_state.models['nb_tfidf'] = nb_model
                                    
                                    st.success(f"✅ Naive Bayes (TF-IDF) - Accuracy: {accuracy:.4f}")
                        
                        elif feature_method == "Bag of Words":
                            # Bag of Words Vectorization
                            bow = CountVectorizer(max_features=5000, ngram_range=(1, 2))
                            X_train_bow = bow.fit_transform(X_train)
                            X_test_bow = bow.transform(X_test)
                            
                            # Save vectorizer
                            joblib.dump(bow, 'models/bow_vectorizer.pkl')
                            st.session_state.vectorizers['bow'] = bow
                            
                            # Train Naive Bayes with BoW
                            if "Naive Bayes" in model_types:
                                nb_bow = MultinomialNB()
                                nb_bow.fit(X_train_bow, y_train)
                                
                                y_pred_bow = nb_bow.predict(X_test_bow)
                                
                                accuracy_bow = accuracy_score(y_test, y_pred_bow)
                                precision_bow = precision_score(y_test, y_pred_bow, average='weighted')
                                recall_bow = recall_score(y_test, y_pred_bow, average='weighted')
                                f1_bow = f1_score(y_test, y_pred_bow, average='weighted')
                                
                                cv_scores_bow = cross_val_score(nb_bow, X_train_bow, y_train, cv=5)
                                
                                results[f"NB_BOW"] = {
                                    'model': nb_bow,
                                    'accuracy': accuracy_bow,
                                    'precision': precision_bow,
                                    'recall': recall_bow,
                                    'f1': f1_bow,
                                    'cv_mean': cv_scores_bow.mean(),
                                    'cv_std': cv_scores_bow.std(),
                                    'y_test': y_test,
                                    'y_pred': y_pred_bow,
                                    'labels': le.classes_
                                }
                                
                                joblib.dump(nb_bow, 'models/naive_bayes_bow.pkl')
                                st.session_state.models['nb_bow'] = nb_bow
                                
                                st.success(f"✅ Naive Bayes (BoW) - Accuracy: {accuracy_bow:.4f}")
                        
                        elif feature_method == "Word2Vec":
                            # Word2Vec Training
                            sentences = [text.split() for text in X_train if text.strip()]
                            
                            # Train Word2Vec model
                            w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
                            
                            # Create document vectors
                            def get_doc_vector(text, model):
                                words = text.split()
                                word_vectors = [model.wv[word] for word in words if word in model.wv]
                                if word_vectors:
                                    return np.mean(word_vectors, axis=0)
                                else:
                                    return np.zeros(model.vector_size)
                            
                            X_train_w2v = np.array([get_doc_vector(text, w2v_model) for text in X_train])
                            X_test_w2v = np.array([get_doc_vector(text, w2v_model) for text in X_test])
                            
                            # Save Word2Vec model
                            w2v_model.save('models/word2vec.model')
                            
                            # Train LSTM with Word2Vec
                            if "LSTM" in model_types:
                                # Prepare data for LSTM
                                tokenizer = Tokenizer(num_words=5000)
                                tokenizer.fit_on_texts(X_train)
                                
                                X_train_seq = tokenizer.texts_to_sequences(X_train)
                                X_test_seq = tokenizer.texts_to_sequences(X_test)
                                
                                max_length = 100
                                X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
                                X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)
                                
                                # Build LSTM model
                                lstm_model = Sequential([
                                    Embedding(5000, 128, input_length=max_length),
                                    SpatialDropout1D(0.2),
                                    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
                                    Dense(32, activation='relu'),
                                    Dropout(0.5),
                                    Dense(len(le.classes_), activation='softmax')
                                ])
                                
                                lstm_model.compile(
                                    loss='sparse_categorical_crossentropy',
                                    optimizer='adam',
                                    metrics=['accuracy']
                                )
                                
                                # Callbacks
                                early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
                                reduce_lr = ReduceLROnPlateau(patience=2, factor=0.5)
                                
                                # Train LSTM
                                history = lstm_model.fit(
                                    X_train_pad, y_train,
                                    epochs=10,
                                    batch_size=32,
                                    validation_split=0.2,
                                    callbacks=[early_stopping, reduce_lr],
                                    verbose=0
                                )
                                
                                # Evaluate LSTM
                                y_pred_lstm = lstm_model.predict(X_test_pad)
                                y_pred_lstm_classes = np.argmax(y_pred_lstm, axis=1)
                                
                                accuracy_lstm = accuracy_score(y_test, y_pred_lstm_classes)
                                precision_lstm = precision_score(y_test, y_pred_lstm_classes, average='weighted')
                                recall_lstm = recall_score(y_test, y_pred_lstm_classes, average='weighted')
                                f1_lstm = f1_score(y_test, y_pred_lstm_classes, average='weighted')
                                
                                results["LSTM"] = {
                                    'model': lstm_model,
                                    'accuracy': accuracy_lstm,
                                    'precision': precision_lstm,
                                    'recall': recall_lstm,
                                    'f1': f1_lstm,
                                    'y_test': y_test,
                                    'y_pred': y_pred_lstm_classes,
                                    'labels': le.classes_,
                                    'tokenizer': tokenizer,
                                    'max_length': max_length
                                }
                                
                                # Save LSTM model and tokenizer
                                lstm_model.save('models/lstm_model.h5')
                                joblib.dump(tokenizer, 'models/tokenizer.pkl')
                                st.session_state.models['lstm'] = lstm_model
                                
                                st.success(f"✅ LSTM - Accuracy: {accuracy_lstm:.4f}")
                    
                    # Display Results
                    st.markdown("### 📊 Model Evaluation Results")
                    
                    # Create results table
                    results_df = []
                    for model_name, result in results.items():
                        results_df.append({
                            'Model': model_name,
                            'Accuracy': result['accuracy'],
                            'Precision': result['precision'],
                            'Recall': result['recall'],
                            'F1-Score': result['f1'],
                            'CV Mean': result.get('cv_mean', 'N/A'),
                            'CV Std': result.get('cv_std', 'N/A')
                        })
                    
                    results_table = pd.DataFrame(results_df)
                    st.dataframe(results_table, use_container_width=True)
                    
                    # Plot confusion matrices
                    st.markdown("### 🔥 Confusion Matrices")
                    
                    n_models = len(results)
                    if n_models > 0:
                        cols = st.columns(min(n_models, 2))
                        
                        for idx, (model_name, result) in enumerate(results.items()):
                            col = cols[idx % 2]
                            
                            with col:
                                cm = confusion_matrix(result['y_test'], result['y_pred'])
                                
                                fig, ax = plt.subplots(figsize=(6, 5))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                          xticklabels=result['labels'], 
                                          yticklabels=result['labels'], ax=ax)
                                ax.set_title(f'Confusion Matrix - {model_name}')
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                
                                st.pyplot(fig)
                                plt.close()
                    
                    # Save label encoder
                    joblib.dump(le, 'models/label_encoder.pkl')
                    
                    st.success("🎉 Training completed! All models and vectorizers saved successfully.")

# PAGE 3: VISUALISASI DATASET
elif page == "Visualisasi Dataset":
    st.markdown('<h2 class="section-header">📈 Visualisasi Dataset</h2>', unsafe_allow_html=True)
    
    # Check for processed data
    processed_files = [f for f in os.listdir('dataset_sudah') if f.endswith('.csv')]
    
    if not processed_files and st.session_state.processed_data is None:
        st.warning("⚠️ Belum ada data yang diproses. Silakan lakukan processing data terlebih dahulu.")
    else:
        # Data selection
        data_source = st.radio(
            "Pilih sumber data:",
            ["Gunakan data dari session", "Load dari file yang tersimpan"]
        )
        
        df_viz = None
        
        if data_source == "Gunakan data dari session" and st.session_state.processed_data is not None:
            df_viz = st.session_state.processed_data
        elif data_source == "Load dari file yang tersimpan":
            if processed_files:
                selected_file = st.selectbox("Pilih file:", processed_files)
                df_viz = pd.read_csv(os.path.join('dataset_sudah', selected_file))
        
        if df_viz is not None:
            # Dataset Overview
            st.markdown("### 📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Data", len(df_viz))
            with col2:
                st.metric("Jumlah Kolom", len(df_viz.columns))
            with col3:
                st.metric("Data Valid", df_viz['processed_text'].notna().sum())
            
            # Sentiment Distribution
            st.markdown("### 🎯 Distribusi Sentiment")
            sentiment_counts = df_viz['sentiment'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig_pie = px.pie(
                    values=sentiment_counts.values, 
                    names=sentiment_counts.index,
                    title="Distribusi Sentiment",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart
                fig_bar = px.bar(
                    x=sentiment_counts.index, 
                    y=sentiment_counts.values,
                    title="Jumlah per Kategori Sentiment",
                    color=sentiment_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Text Length Analysis
            st.markdown("### 📏 Analisis Panjang Teks")
            
            # Calculate text lengths
            df_viz['text_length'] = df_viz['processed_text'].str.len()
            df_viz['word_count'] = df_viz['processed_text'].str.split().str.len()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Text length distribution
                fig_hist = px.histogram(
                    df_viz, 
                    x='text_length', 
                    color='sentiment',
                    title="Distribusi Panjang Teks",
                    nbins=50
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Word count distribution
                fig_word = px.box(
                    df_viz, 
                    x='sentiment', 
                    y='word_count',
                    title="Distribusi Jumlah Kata per Sentiment"
                )
                st.plotly_chart(fig_word, use_container_width=True)
            
            # Word Cloud Analysis
            st.markdown("### ☁️ Word Cloud Analysis")
            
            try:
                from wordcloud import WordCloud
                
                sentiment_selected = st.selectbox(
                    "Pilih sentiment untuk Word Cloud:",
                    ['All'] + list(df_viz['sentiment'].unique())
                )
                
                if sentiment_selected == 'All':
                    text_data = ' '.join(df_viz['processed_text'].dropna())
                else:
                    text_data = ' '.join(df_viz[df_viz['sentiment'] == sentiment_selected]['processed_text'].dropna())
                
                if text_data.strip():
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis'
                    ).generate(text_data)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Word Cloud - {sentiment_selected}', fontsize=16)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Tidak ada data teks untuk membuat word cloud.")
                    
            except ImportError:
                st.info("Install wordcloud package untuk menampilkan word cloud: pip install wordcloud")
            
            # Top Words Analysis
            st.markdown("### 🔝 Top Words Analysis")
            
            from collections import Counter
            
            # Get top words for each sentiment
            sentiment_words = {}
            for sentiment in df_viz['sentiment'].unique():
                sentiment_text = df_viz[df_viz['sentiment'] == sentiment]['processed_text'].dropna()
                all_words = ' '.join(sentiment_text).split()
                word_counts = Counter(all_words)
                sentiment_words[sentiment] = word_counts.most_common(10)
            
            # Display top words
            cols = st.columns(len(sentiment_words))
            
            for idx, (sentiment, words) in enumerate(sentiment_words.items()):
                with cols[idx]:
                    st.write(f"**Top 10 Words - {sentiment.title()}**")
                    words_df = pd.DataFrame(words, columns=['Word', 'Count'])
                    st.dataframe(words_df, hide_index=True)
            
            # Correlation Analysis
            st.markdown("### 🔗 Correlation Analysis")
            
            # Text statistics
            df_viz['char_count'] = df_viz['processed_text'].str.len()
            df_viz['word_count'] = df_viz['processed_text'].str.split().str.len()
            df_viz['sentence_count'] = df_viz['processed_text'].str.count('\.') + 1
            df_viz['avg_word_length'] = df_viz['char_count'] / df_viz['word_count']
            
            # Correlation matrix
            numeric_cols = ['char_count', 'word_count', 'sentence_count', 'avg_word_length']
            corr_matrix = df_viz[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix - Text Statistics')
            st.pyplot(fig)
            plt.close()
            
            # Data Quality Report
            st.markdown("### 📋 Data Quality Report")
            
            quality_metrics = {
                'Total Records': len(df_viz),
                'Records with Text': df_viz['processed_text'].notna().sum(),
                'Empty Text Records': df_viz['processed_text'].isna().sum(),
                'Average Text Length': df_viz['char_count'].mean(),
                'Average Word Count': df_viz['word_count'].mean(),
                'Shortest Text': df_viz['char_count'].min(),
                'Longest Text': df_viz['char_count'].max()
            }
            
            quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['Metric', 'Value'])
            st.dataframe(quality_df, hide_index=True, use_container_width=True)

# PAGE 4: TEXT ANALYSIS
elif page == "Text Analysis":
    st.markdown('<h2 class="section-header">🔍 Text Analysis</h2>', unsafe_allow_html=True)
    
    # Check for trained models
    model_files = os.listdir('models')
    
    if not model_files:
        st.warning("⚠️ Belum ada model yang di-train. Silakan lakukan training model terlebih dahulu.")
    else:
        st.markdown("### 🎯 Analisis Sentimen Real-time")
        
        # Text input
        user_text = st.text_area(
            "Masukkan teks yang ingin dianalisis:",
            placeholder="Ketik ulasan atau komentar di sini...",
            height=100
        )
        
        # Model selection
        available_models = []
        if 'naive_bayes_tfidf.pkl' in model_files:
            available_models.append("Naive Bayes (TF-IDF)")
        if 'naive_bayes_bow.pkl' in model_files:
            available_models.append("Naive Bayes (BoW)")
        if 'lstm_model.h5' in model_files:
            available_models.append("LSTM")
        
        if available_models:
            selected_model = st.selectbox("Pilih Model:", available_models)
            
            if st.button("🔮 Prediksi Sentimen") and user_text.strip():
                with st.spinner("Menganalisis sentimen..."):
                    try:
                        # Initialize preprocessor
                        preprocessor = TextPreprocessor()
                        
                        # Preprocess input text
                        processed_input = preprocessor.preprocess_text(user_text)
                        
                        if not processed_input.strip():
                            st.warning("Teks tidak dapat diproses. Coba masukkan teks yang berbeda.")
                        else:
                            # Load label encoder
                            le = joblib.load('models/label_encoder.pkl')
                            
                            prediction = None
                            confidence_scores = None
                            
                            if selected_model == "Naive Bayes (TF-IDF)":
                                # Load TF-IDF model
                                nb_model = joblib.load('models/naive_bayes_tfidf.pkl')
                                tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
                                
                                # Transform input
                                input_tfidf = tfidf_vectorizer.transform([processed_input])
                                
                                # Predict
                                prediction = nb_model.predict(input_tfidf)[0]
                                confidence_scores = nb_model.predict_proba(input_tfidf)[0]
                            
                            elif selected_model == "Naive Bayes (BoW)":
                                # Load BoW model
                                nb_model = joblib.load('models/naive_bayes_bow.pkl')
                                bow_vectorizer = joblib.load('models/bow_vectorizer.pkl')
                                
                                # Transform input
                                input_bow = bow_vectorizer.transform([processed_input])
                                
                                # Predict
                                prediction = nb_model.predict(input_bow)[0]
                                confidence_scores = nb_model.predict_proba(input_bow)[0]
                            
                            elif selected_model == "LSTM":
                                # Load LSTM model
                                lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
                                tokenizer = joblib.load('models/tokenizer.pkl')
                                
                                # Transform input
                                input_seq = tokenizer.texts_to_sequences([processed_input])
                                input_pad = pad_sequences(input_seq, maxlen=100)  # Same as training
                                
                                # Predict
                                prediction_proba = lstm_model.predict(input_pad, verbose=0)
                                prediction = np.argmax(prediction_proba[0])
                                confidence_scores = prediction_proba[0]
                            
                            if prediction is not None:
                                # Get sentiment label
                                sentiment_label = le.inverse_transform([prediction])[0]
                                
                                # Display results
                                st.markdown("### 🎯 Hasil Analisis")
                                
                                # Main prediction
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    sentiment_colors = {
                                        'positive': '🟢',
                                        'negative': '🔴',
                                        'neutral': '🟡'
                                    }
                                    
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <h3>{sentiment_colors.get(sentiment_label, '⚪')} Prediksi Sentimen</h3>
                                        <h2 style="color: {'green' if sentiment_label == 'positive' else 'red' if sentiment_label == 'negative' else 'orange'};">
                                            {sentiment_label.upper()}
                                        </h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    max_confidence = confidence_scores.max()
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <h3>📊 Confidence Score</h3>
                                        <h2>{max_confidence:.2%}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Confidence breakdown
                                st.markdown("### 📈 Confidence Breakdown")
                                
                                confidence_df = pd.DataFrame({
                                    'Sentiment': le.classes_,
                                    'Confidence': confidence_scores
                                })
                                
                                fig_conf = px.bar(
                                    confidence_df,
                                    x='Sentiment',
                                    y='Confidence',
                                    title="Confidence Score per Sentiment",
                                    color='Confidence',
                                    color_continuous_scale='viridis'
                                )
                                fig_conf.update_layout(showlegend=False)
                                st.plotly_chart(fig_conf, use_container_width=True)
                                
                                # Text processing steps
                                st.markdown("### 🔧 Text Processing Steps")
                                
                                steps_data = {
                                    'Step': ['Original Text', 'Cleaned Text', 'Normalized Text', 'Final Processed'],
                                    'Text': [
                                        user_text[:100] + "..." if len(user_text) > 100 else user_text,
                                        preprocessor.clean_text(user_text)[:100] + "..." if len(preprocessor.clean_text(user_text)) > 100 else preprocessor.clean_text(user_text),
                                        preprocessor.normalize_text(preprocessor.clean_text(user_text))[:100] + "..." if len(preprocessor.normalize_text(preprocessor.clean_text(user_text))) > 100 else preprocessor.normalize_text(preprocessor.clean_text(user_text)),
                                        processed_input[:100] + "..." if len(processed_input) > 100 else processed_input
                                    ]
                                }
                                
                                steps_df = pd.DataFrame(steps_data)
                                st.dataframe(steps_df, hide_index=True, use_container_width=True)
                                
                                # Key words highlight
                                st.markdown("### 🔑 Key Words Analysis")
                                
                                words = processed_input.split()
                                if words:
                                    st.write("**Processed Words:**", ", ".join(words[:20]))
                                    st.write(f"**Word Count:** {len(words)}")
                                    st.write(f"**Character Count:** {len(processed_input)}")
                    
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")
        
        # Batch Analysis
        st.markdown("### 📁 Batch Analysis")
        
        uploaded_batch = st.file_uploader(
            "Upload file CSV untuk analisis batch:",
            type=['csv'],
            help="File harus memiliki kolom 'text' yang berisi teks untuk dianalisis"
        )
        
        if uploaded_batch is not None:
            batch_df = pd.read_csv(uploaded_batch)
            st.write("**Preview Data:**")
            st.write(batch_df.head())
            
            if 'text' in batch_df.columns:
                if st.button("🚀 Analisis Batch"):
                    with st.spinner("Memproses batch analysis..."):
                        # Initialize preprocessor
                        preprocessor = TextPreprocessor()
                        
                        # Preprocess all texts
                        batch_df['processed_text'] = batch_df['text'].apply(preprocessor.preprocess_text)
                        
                        # Load model and predict
                        if available_models:
                            # Use first available model for batch
                            model_name = available_models[0]
                            le = joblib.load('models/label_encoder.pkl')
                            
                            predictions = []
                            confidences = []
                            
                            for text in batch_df['processed_text']:
                                if not text.strip():
                                    predictions.append('neutral')
                                    confidences.append(0.33)
                                    continue
                                
                                try:
                                    if "TF-IDF" in model_name:
                                        nb_model = joblib.load('models/naive_bayes_tfidf.pkl')
                                        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
                                        input_vec = tfidf_vectorizer.transform([text])
                                        pred = nb_model.predict(input_vec)[0]
                                        conf = nb_model.predict_proba(input_vec)[0].max()
                                    elif "BoW" in model_name:
                                        nb_model = joblib.load('models/naive_bayes_bow.pkl')
                                        bow_vectorizer = joblib.load('models/bow_vectorizer.pkl')
                                        input_vec = bow_vectorizer.transform([text])
                                        pred = nb_model.predict(input_vec)[0]
                                        conf = nb_model.predict_proba(input_vec)[0].max()
                                    else:  # LSTM
                                        lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
                                        tokenizer = joblib.load('models/tokenizer.pkl')
                                        input_seq = tokenizer.texts_to_sequences([text])
                                        input_pad = pad_sequences(input_seq, maxlen=100)
                                        pred_proba = lstm_model.predict(input_pad, verbose=0)
                                        pred = np.argmax(pred_proba[0])
                                        conf = pred_proba[0].max()
                                    
                                    sentiment = le.inverse_transform([pred])[0]
                                    predictions.append(sentiment)
                                    confidences.append(conf)
                                
                                except Exception as e:
                                    predictions.append('neutral')
                                    confidences.append(0.33)
                            
                            # Add results to dataframe
                            batch_df['predicted_sentiment'] = predictions
                            batch_df['confidence'] = confidences
                            
                            # Display results
                            st.markdown("### 📊 Batch Analysis Results")
                            
                            # Summary
                            result_counts = pd.Series(predictions).value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Sentiment Distribution:**")
                                st.write(result_counts)
                            
                            with col2:
                                fig_batch = px.pie(
                                    values=result_counts.values,
                                    names=result_counts.index,
                                    title="Batch Analysis Results"
                                )
                                st.plotly_chart(fig_batch, use_container_width=True)
                            
                            # Download results
                            csv_result = batch_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv_result,
                                file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime='text/csv'
                            )
                            
                            # Show detailed results
                            st.markdown("### 📋 Detailed Results")
                            display_cols = ['text', 'predicted_sentiment', 'confidence']
                            st.dataframe(batch_df[display_cols], use_container_width=True)
            
            else:
                st.error("File CSV harus memiliki kolom 'text'")

# PAGE 5: TENTANG
elif page == "Tentang":
    st.markdown('<h2 class="section-header">ℹ️ Tentang Aplikasi</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Sistem Analisis Sentimen Bahasa Indonesia
    
    Aplikasi ini adalah sistem komprehensif untuk analisis sentimen teks berbahasa Indonesia yang mengimplementasikan 
    berbagai teknik machine learning dan deep learning.
    
    ### ✨ Fitur Utama
    
    #### 1. 📝 Processing Data
    - **Labelling Otomatis**: Sistem pelabelan sentimen berdasarkan kata kunci
    - **Pembersihan Teks**: Menghilangkan URL, emoji, karakter khusus, dan noise
    - **Normalisasi**: Mengubah kata tidak baku menjadi kata baku Bahasa Indonesia
    - **Tokenisasi**: Pemisahan teks menjadi token-token
    - **Stopword Removal**: Menghapus kata-kata yang tidak informatif
    - **Stemming**: Mengubah kata ke bentuk dasarnya menggunakan Sastrawi
    
    #### 2. 🤖 Model Machine Learning
    - **Naive Bayes**: Model klasik untuk klasifikasi teks
    - **LSTM**: Model deep learning untuk sequence processing
    - **Feature Engineering**: TF-IDF, Bag of Words, dan Word2Vec
    - **Cross Validation**: Evaluasi model dengan 5-fold cross validation
    
    #### 3. 📊 Evaluasi Model
    - **Metrics**: Accuracy, Precision, Recall, F1-Score
    - **Confusion Matrix**: Visualisasi performa model
    - **Model Comparison**: Perbandingan berbagai model
    
    #### 4. 📈 Visualisasi Dataset
    - **Distribusi Sentiment**: Pie chart dan bar chart
    - **Analisis Panjang Teks**: Histogram dan box plot
    - **Word Cloud**: Visualisasi kata-kata populer
    - **Top Words**: Analisis kata-kata teratas per sentiment
    - **Correlation Analysis**: Analisis korelasi antar fitur
    
    #### 5. 🔍 Text Analysis
    - **Real-time Prediction**: Prediksi sentimen langsung
    - **Confidence Score**: Tingkat keyakinan prediksi
    - **Batch Analysis**: Analisis massal dari file CSV
    - **Processing Steps**: Visualisasi tahapan preprocessing
    
    ### 🛠️ Teknologi yang Digunakan
    
    #### Python Libraries:
    - **Streamlit**: Framework web app
    - **Pandas & NumPy**: Data manipulation
    - **Scikit-learn**: Machine learning
    - **TensorFlow/Keras**: Deep learning
    - **Sastrawi**: Indonesian NLP
    - **Plotly & Matplotlib**: Visualisasi
    - **NLTK**: Natural Language Processing
    - **Gensim**: Word embeddings
    
    #### Machine Learning Models:
    - **Multinomial Naive Bayes**: Untuk klasifikasi teks
    - **LSTM Neural Network**: Untuk sequence modeling
    - **TF-IDF Vectorizer**: Feature extraction
    - **Word2Vec**: Word embeddings
    
    ### 📋 Cara Penggunaan
    
    1. **Upload Dataset**: Upload file CSV dengan kolom teks
    2. **Processing Data**: Pilih jumlah data dan jalankan preprocessing
    3. **Train Model**: Pilih feature engineering dan model untuk training
    4. **Visualisasi**: Lihat insight dari dataset yang sudah diproses
    5. **Text Analysis**: Gunakan model untuk prediksi sentiment baru
    
    ### 🎨 Keunggulan Sistem
    
    - ✅ **User-Friendly Interface**: Antarmuka yang mudah digunakan
    - ✅ **Comprehensive Pipeline**: Pipeline lengkap dari preprocessing hingga prediksi
    - ✅ **Multiple Models**: Implementasi berbagai model ML dan DL
    - ✅ **Bahasa Indonesia**: Optimized untuk teks Bahasa Indonesia
    - ✅ **Real-time Analysis**: Analisis sentimen secara real-time
    - ✅ **Batch Processing**: Analisis massal untuk dataset besar
    - ✅ **Rich Visualizations**: Visualisasi yang informatif dan menarik
    - ✅ **Model Persistence**: Menyimpan dan memuat model yang sudah dilatih
    
    ### 📊 Model Performance
    
    Sistem ini menggunakan berbagai metrics untuk evaluasi:
    - **Accuracy**: Akurasi keseluruhan
    - **Precision**: Ketepatan prediksi positif
    - **Recall**: Kemampuan mendeteksi kelas positif
    - **F1-Score**: Harmonic mean dari precision dan recall
    - **Cross Validation**: Validasi silang untuk robustness
    
    ### 🔬 Research & Development
    
    Aplikasi ini dikembangkan untuk keperluan:
    - Penelitian analisis sentimen Bahasa Indonesia
    - Analisis opini publik dari media sosial
    - Monitoring brand sentiment
    - Academic research and learning
    
    ### 💡 Tips Penggunaan
    
    1. **Data Quality**: Pastikan data input berkualitas baik
    2. **Preprocessing**: Lakukan preprocessing yang tepat sesuai domain
    3. **Model Selection**: Pilih model sesuai dengan karakteristik data
    4. **Validation**: Selalu validasi hasil dengan cross-validation
    5. **Interpretation**: Interpretasikan hasil dengan mempertimbangkan konteks
    
    ---
    
    ### 👨‍💻 Developed with ❤️ 
    
    Sistem ini dikembangkan menggunakan best practices dalam machine learning dan natural language processing 
    untuk memberikan solusi analisis sentimen yang komprehensif dan mudah digunakan.
    
    **Version**: 1.0.0  
    **Last Updated**: May 2025
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 Sistem Analisis Sentimen Indonesia | Built with Streamlit & Python</p>
        <p>💡 Machine Learning • Deep Learning • Natural Language Processing</p>
    </div>
    """, 
    unsafe_allow_html=True
)