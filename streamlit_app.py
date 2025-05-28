import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import os
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our sentiment analysis system
from sentiment_analysis_system import SentimentAnalysisSystem

# Configure page
st.set_page_config(
    page_title="Sistem Analisis Sentimen",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sentiment_system' not in st.session_state:
    st.session_state.sentiment_system = SentimentAnalysisSystem()

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Sistem Analisis Sentimen Indonesia</h1>
    <p>Analisis sentimen otomatis untuk ulasan dan komentar berbahasa Indonesia</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Menu Navigasi")
menu_options = [
    "🏠 Beranda",
    "⚙️ Preprocessing Data", 
    "🤖 Train Model",
    "📊 Visualisasi Dataset",
    "📝 Text Analysis",
    "ℹ️ Tentang"
]

selected_menu = st.sidebar.selectbox("Pilih Menu:", menu_options)

# Helper functions
def display_preprocessing_stats(df_original, df_processed):
    """Display preprocessing statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Asli", f"{len(df_original):,}", help="Jumlah data sebelum preprocessing")
    
    with col2:
        st.metric("Data Setelah Preprocessing", f"{len(df_processed):,}", 
                 delta=f"{len(df_processed) - len(df_original):,}")
    
    with col3:
        avg_length_original = df_original['content'].str.len().mean()
        avg_length_processed = df_processed['processed_content'].str.len().mean()
        st.metric("Rata-rata Panjang Teks", f"{avg_length_processed:.1f}", 
                 delta=f"{avg_length_processed - avg_length_original:.1f}")
    
    with col4:
        unique_words = len(set(' '.join(df_processed['processed_content']).split()))
        st.metric("Kosakata Unik", f"{unique_words:,}")

def plot_sentiment_distribution(df):
    """Plot sentiment distribution"""
    sentiment_counts = df['sentiment'].value_counts()
    
    # Create pie chart
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                title="Distribusi Sentimen", color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def plot_text_length_distribution(df):
    """Plot text length distribution"""
    # Handle NaN and convert to string
    valid_content = df['processed_content'].dropna().astype(str)
    valid_content = valid_content[valid_content.str.len() > 0]
    
    if len(valid_content) == 0:
        # Return empty figure if no valid content
        fig = px.bar(x=[], y=[], title="Tidak ada data valid untuk ditampilkan")
        return fig
    
    # Calculate text lengths
    text_lengths = valid_content.str.len()
    
    fig = px.histogram(
        x=text_lengths, 
        nbins=min(50, len(text_lengths)), 
        title="Distribusi Panjang Teks Setelah Preprocessing",
        labels={'x': 'Panjang Teks', 'y': 'Frekuensi'}
    )
    
    fig.update_layout(height=400)
    return fig

def create_wordcloud(text_data, title="Word Cloud"):
    """Create word cloud"""
    try:
        # Handle NaN values and convert to string
        valid_texts = text_data.dropna().astype(str)
        valid_texts = valid_texts[valid_texts.str.len() > 0]
        
        if len(valid_texts) == 0:
            st.warning("Tidak ada teks valid untuk membuat word cloud")
            return None
        
        # Combine all text
        all_text = ' '.join(valid_texts.tolist())
        
        if len(all_text.strip()) == 0:
            st.warning("Teks kosong, tidak dapat membuat word cloud")
            return None
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis',
            min_font_size=10
        ).generate(all_text)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {e}")
        return None

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot confusion matrix"""
    fig = px.imshow(cm, 
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=class_names, y=class_names,
                   color_continuous_scale="Blues",
                   title=title)
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
            )
    
    fig.update_layout(height=400)
    return fig

def display_model_metrics(results, model_name):
    """Display model evaluation metrics"""
    st.subheader(f"📊 Hasil Evaluasi {model_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.4f}")
    
    with col2:
        st.metric("Precision", f"{results['precision']:.4f}")
    
    with col3:
        st.metric("Recall", f"{results['recall']:.4f}")
    
    with col4:
        st.metric("F1-Score", f"{results['f1_score']:.4f}")
    
    # Confusion Matrix
    if 'confusion_matrix' in results:
        class_names = ['Negative', 'Neutral', 'Positive']
        fig_cm = plot_confusion_matrix(results['confusion_matrix'], class_names, 
                                     f"Confusion Matrix - {model_name}")
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Classification Report
    if 'classification_report' in results:
        st.subheader("📋 Classification Report")
        st.text(results['classification_report'])

# Main content based on selected menu
if selected_menu == "🏠 Beranda":
    st.markdown("""
    ## 🎯 Selamat Datang di Sistem Analisis Sentimen Indonesia
    
    Sistem ini dirancang untuk menganalisis sentimen dari teks berbahasa Indonesia secara otomatis.
    
    ### ✨ Fitur Utama:
    - **Preprocessing Data**: Pembersihan dan normalisasi teks bahasa Indonesia
    - **Multiple Models**: Naive Bayes dan LSTM untuk klasifikasi sentimen
    - **Feature Engineering**: TF-IDF, Bag of Words, dan Word Embeddings
    - **Evaluasi Komprehensif**: Accuracy, Precision, Recall, F1-Score, dan Cross-validation
    - **Analisis Real-time**: Input teks dan dapatkan prediksi sentimen langsung
    
    ### 🚀 Cara Menggunakan:
    1. **Preprocessing Data**: Upload dataset dan lakukan preprocessing
    2. **Train Model**: Latih model dengan data yang sudah diproses
    3. **Visualisasi**: Lihat statistik dan visualisasi dataset
    4. **Text Analysis**: Analisis sentimen teks secara real-time
    
    ### 📁 Requirements:
    Pastikan file `Dataset.csv` tersedia dengan kolom:
    - `userName`: Nama pengguna
    - `content`: Konten ulasan/komentar
    - `score`: Skor rating (1-5)
    - `at`: Tanggal
    - `appVersion`: Versi aplikasi
    """)
    
    # Check if dataset exists
    if os.path.exists('Dataset.csv'):
        st.success("✅ Dataset.csv ditemukan!")
        df_sample = pd.read_csv('Dataset.csv').head()
        st.subheader("👀 Preview Dataset")
        st.dataframe(df_sample)
    else:
        st.error("❌ Dataset.csv tidak ditemukan! Pastikan file tersedia di direktori yang sama.")

elif selected_menu == "⚙️ Preprocessing Data":
    st.header("⚙️ Preprocessing Data")
    
    if not os.path.exists('Dataset.csv'):
        st.error("❌ Dataset.csv tidak ditemukan!")
        st.stop()
    
    # Load dataset info
    df_info = pd.read_csv('Dataset.csv')
    st.info(f"📊 Dataset tersedia: {len(df_info):,} baris data")
    
    # Sample size selection
    st.subheader("📏 Pilih Jumlah Data untuk Diproses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_options = {
            "Semua Data": len(df_info),
            "10,000 Data": min(10000, len(df_info)),
            "5,000 Data": min(5000, len(df_info)),
            "1,000 Data": min(1000, len(df_info)),
            "500 Data": min(500, len(df_info))
        }
        
        selected_option = st.selectbox("Pilih jumlah data:", list(sample_options.keys()))
        sample_size = sample_options[selected_option]
    
    with col2:
        st.metric("Data yang akan diproses", f"{sample_size:,}")
        if sample_size < len(df_info):
            st.info(f"Akan mengambil sampel acak dari {len(df_info):,} data")
    
    # Preprocessing button
    if st.button("🚀 Mulai Preprocessing", type="primary", key="start_preprocessing_btn"):
        with st.spinner("Sedang memproses data..."):
            try:
                # Load and preprocess data
                df_processed = st.session_state.sentiment_system.load_and_preprocess_data(
                    'Dataset.csv', sample_size=sample_size if sample_size < len(df_info) else None
                )
                
                # Save processed data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_dataset_{sample_size}_{timestamp}.csv"
                filepath = st.session_state.sentiment_system.save_processed_data(df_processed, filename)
                
                # Store in session state
                st.session_state.processed_data = df_processed
                
                st.success(f"✅ Preprocessing selesai! Data disimpan ke: {filepath}")
                
                # Display statistics
                st.subheader("📊 Statistik Preprocessing")
                display_preprocessing_stats(df_info.head(sample_size), df_processed)
                
                # Show sample of processed data
                st.subheader("👀 Sample Data Setelah Preprocessing")
                display_cols = ['content', 'processed_content', 'sentiment']
                if all(col in df_processed.columns for col in display_cols):
                    st.dataframe(df_processed[display_cols].head(10))
                
                # Sentiment distribution
                st.subheader("📈 Distribusi Sentimen")
                fig_sentiment = plot_sentiment_distribution(df_processed)
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
                # Text length distribution
                st.subheader("📏 Distribusi Panjang Teks")
                fig_length = plot_text_length_distribution(df_processed)
                st.plotly_chart(fig_length, use_container_width=True)
                
                # Word cloud
                st.subheader("☁️ Word Cloud")
                fig_wc = create_wordcloud(df_processed['processed_content'], "Word Cloud - Processed Text")
                if fig_wc:
                    st.pyplot(fig_wc)
                
            except Exception as e:
                st.error(f"❌ Error saat preprocessing: {e}")
    
    # Show existing processed files
    if os.path.exists('dataset_sudah') and os.listdir('dataset_sudah'):
        st.subheader("📁 File yang Sudah Diproses")
        processed_files = [f for f in os.listdir('dataset_sudah') if f.endswith('.csv')]
        
        if processed_files:
            selected_file = st.selectbox("Pilih file untuk dimuat:", processed_files)
            
            if st.button("📂 Muat Data Processed", key="load_processed_btn"):
                filepath = os.path.join('dataset_sudah', selected_file)
                df_loaded = pd.read_csv(filepath)
                st.session_state.processed_data = df_loaded
                st.success(f"✅ Data loaded: {selected_file}")
                st.dataframe(df_loaded.head())

elif selected_menu == "🤖 Train Model":
    st.header("🤖 Training Model")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Belum ada data yang diproses. Silakan lakukan preprocessing terlebih dahulu.")
        st.stop()
    
    df = st.session_state.processed_data
    st.success(f"✅ Data siap untuk training: {len(df):,} baris")
    
    # Model selection
    st.subheader("🎯 Pilih Model untuk Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_nb = st.checkbox("🔢 Train Naive Bayes", value=True)
        if train_nb:
            nb_feature = st.selectbox("Feature untuk Naive Bayes:", ["TF-IDF", "Bag of Words"])
    
    with col2:
        train_lstm = st.checkbox("🧠 Train LSTM", value=True)
    
    # Training parameters
    st.subheader("⚙️ Parameter Training")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size (%)", 10, 30, 20) / 100
    
    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    with col3:
        random_state = st.number_input("Random State", 1, 100, 42)
    
    # Training button
    if st.button("🚀 Mulai Training", type="primary", key="start_training_btn"):
        with st.spinner("Training models..."):
            try:
                results = {}
                
                # Create features
                st.info("🔧 Membuat features...")
                features = st.session_state.sentiment_system.create_features(df)
                
                # Prepare data
                from sklearn.model_selection import train_test_split
                
                # Train Naive Bayes
                if train_nb:
                    st.info(f"🔢 Training Naive Bayes dengan {nb_feature}...")
                    
                    # Select features
                    if nb_feature == "TF-IDF":
                        X = features['tfidf']
                    else:
                        X = features['bow']
                    
                    y = df['sentiment']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Train model
                    nb_model = st.session_state.sentiment_system.train_naive_bayes(
                        X_train, y_train, nb_feature.lower().replace('-', '').replace(' ', '')
                    )
                    
                    # Evaluate model
                    nb_results = st.session_state.sentiment_system.evaluate_model(
                        nb_model, X_test, y_test, "Naive Bayes"
                    )
                    
                    # Cross validation
                    cv_scores = st.session_state.sentiment_system.cross_validate_model(
                        nb_model, X, y, cv=cv_folds
                    )
                    nb_results['cv_scores'] = cv_scores
                    
                    results['naive_bayes'] = nb_results
                
                # Train LSTM
                if train_lstm:
                    st.info("🧠 Training LSTM...")
                    
                    # Prepare LSTM data
                    X_lstm, y_lstm = st.session_state.sentiment_system.prepare_lstm_data(
                        df['processed_content'], df['sentiment']
                    )
                    
                    # Split data
                    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                        X_lstm, y_lstm, test_size=test_size, random_state=random_state
                    )
                    
                    # Further split for validation
                    X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
                        X_train_lstm, y_train_lstm, test_size=0.2, random_state=random_state
                    )
                    
                    # Train model
                    lstm_history = st.session_state.sentiment_system.train_lstm(
                        X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm
                    )
                    
                    # Evaluate model
                    lstm_results = st.session_state.sentiment_system.evaluate_model(
                        st.session_state.sentiment_system.lstm_model, 
                        X_test_lstm, y_test_lstm, "LSTM", is_lstm=True
                    )
                    
                    results['lstm'] = lstm_results
                    results['lstm_history'] = lstm_history
                
                # Save vectorizers
                st.session_state.sentiment_system.save_vectorizers()
                
                # Mark as trained
                st.session_state.model_trained = True
                
                st.success("✅ Training selesai!")
                
                # Display results
                for model_name, model_results in results.items():
                    if model_name != 'lstm_history':
                        display_model_metrics(model_results, model_name.replace('_', ' ').title())
                
                # LSTM training history
                if 'lstm_history' in results:
                    st.subheader("📈 LSTM Training History")
                    
                    history = results['lstm_history'].history
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Accuracy', 'Loss')
                    )
                    
                    # Accuracy plot
                    fig.add_trace(
                        go.Scatter(y=history['accuracy'], name='Train Accuracy'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(y=history['val_accuracy'], name='Val Accuracy'),
                        row=1, col=1
                    )
                    
                    # Loss plot
                    fig.add_trace(
                        go.Scatter(y=history['loss'], name='Train Loss'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Scatter(y=history['val_loss'], name='Val Loss'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error saat training: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show existing models
    if os.path.exists('models') and os.listdir('models'):
        st.subheader("💾 Model yang Tersimpan")
        model_files = os.listdir('models')
        
        if model_files:
            st.info("Model yang tersedia:")
            for file in model_files:
                st.text(f"📁 {file}")
            
            if st.button("📂 Load Models", key="load_models_btn"):
                st.session_state.sentiment_system.load_models()
                st.session_state.model_trained = True
                st.success("✅ Models loaded successfully!")

elif selected_menu == "📊 Visualisasi Dataset":
    st.header("📊 Visualisasi Dataset")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Belum ada data yang diproses. Silakan lakukan preprocessing terlebih dahulu.")
        st.stop()
    
    df = st.session_state.processed_data
    
    # Dataset overview
    st.subheader("📋 Overview Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data", f"{len(df):,}")
    
    with col2:
        # Handle NaN values in processed_content
        valid_content = df['processed_content'].dropna().astype(str)
        valid_content = valid_content[valid_content.str.len() > 0]
        
        if len(valid_content) > 0:
            avg_length = valid_content.str.len().mean()
            st.metric("Rata-rata Panjang Teks", f"{avg_length:.1f}")
        else:
            st.metric("Rata-rata Panjang Teks", "0")
    
    with col3:
        try:
            # Filter valid content and join safely
            valid_texts = df['processed_content'].dropna().astype(str)
            valid_texts = valid_texts[valid_texts.str.len() > 0]
            
            if len(valid_texts) > 0:
                all_text = ' '.join(valid_texts.tolist())
                unique_words = len(set(all_text.split()))
                st.metric("Kosakata Unik", f"{unique_words:,}")
            else:
                st.metric("Kosakata Unik", "0")
        except Exception as e:
            st.metric("Kosakata Unik", "Error")
            st.error(f"Error calculating vocabulary: {e}")
    
    with col4:
        try:
            sentiment_balance = df['sentiment'].value_counts().std()
            st.metric("Balance Score", f"{sentiment_balance:.2f}")
        except:
            st.metric("Balance Score", "N/A")
    
    # Sentiment distribution
    st.subheader("📈 Distribusi Sentimen")
    fig_sentiment = plot_sentiment_distribution(df)
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Sentiment by time (if date column exists)
    if 'at' in df.columns:
        st.subheader("📅 Trend Sentimen dari Waktu ke Waktu")
        
        # Debug info ekspander
        with st.expander("🔍 Debug Info - Data Tanggal"):
            st.write("**Informasi kolom 'at':**")
            st.write(f"- Data type: {df['at'].dtype}")
            st.write(f"- Missing values: {df['at'].isnull().sum()}")
            st.write(f"- Total rows: {len(df)}")
            
            # Show sample values
            if 'at' in df.columns:
                st.write("**Sample values (first 10):**")
                sample_values = df['at'].dropna().head(10)
                for i, val in enumerate(sample_values):
                    st.write(f"{i+1}. `{val}` (type: {type(val).__name__})")
                
                # Show unique date formats
                unique_formats = df['at'].dropna().astype(str).apply(lambda x: len(x)).value_counts()
                st.write("**String length distribution:**")
                st.write(unique_formats)
        
        try:
            # Step 1: Data cleaning
            df_time = df.copy()
            df_time = df_time.dropna(subset=['at'])
            
            if len(df_time) == 0:
                st.warning("❌ Tidak ada data tanggal yang valid")
            else:
                st.info(f"📊 Processing {len(df_time)} rows with date data...")
            
            # Step 2: Multiple date parsing attempts
            parsing_success = False
            parsed_dates = None
            error_msgs = []
            
            # Method 1: Standard parsing
            try:
                parsed_dates = pd.to_datetime(df_time['at'])
                parsing_success = True
                st.success("✅ Method 1: Standard parsing berhasil")
            except Exception as e:
                error_msgs.append(f"Method 1 failed: {str(e)}")
                
                # Method 2: Infer format
                try:
                    parsed_dates = pd.to_datetime(df_time['at'], infer_datetime_format=True)
                    parsing_success = True
                    st.success("✅ Method 2: Infer format berhasil")
                except Exception as e2:
                    error_msgs.append(f"Method 2 failed: {str(e2)}")
                    
                    # Method 3: Common formats
                    common_formats = [
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%d',
                        '%d/%m/%Y %H:%M:%S',
                        '%d/%m/%Y',
                        '%m/%d/%Y %H:%M:%S',
                        '%m/%d/%Y',
                        '%d-%m-%Y %H:%M:%S',
                        '%d-%m-%Y'
                    ]
                    
                    for fmt in common_formats:
                        try:
                            parsed_dates = pd.to_datetime(df_time['at'], format=fmt)
                            parsing_success = True
                            st.success(f"✅ Method 3: Format '{fmt}' berhasil")
                            break
                        except:
                            continue
                    
                    if not parsing_success:
                        # Method 4: Coerce errors (last resort)
                        try:
                            parsed_dates = pd.to_datetime(df_time['at'], errors='coerce')
                            valid_dates = parsed_dates.notna()
                            if valid_dates.sum() > 0:
                                parsing_success = True
                                st.warning(f"⚠️ Method 4: Coerce errors - {valid_dates.sum()}/{len(parsed_dates)} dates valid")
                            else:
                                st.error("❌ All date parsing methods failed")
                        except Exception as e4:
                            error_msgs.append(f"Method 4 failed: {str(e4)}")
            
            if parsing_success and parsed_dates is not None:
                # Apply parsed dates
                df_time['date'] = parsed_dates
                
                # Remove invalid dates (NaT)
                df_time = df_time[df_time['date'].notna()]
                
                if len(df_time) == 0:
                    st.error("❌ Semua tanggal tidak valid setelah parsing")
                else:
                    st.info(f"📅 Successfully parsed {len(df_time)} valid dates")
                    
                    # Step 3: Create time features
                    df_time['month_year'] = df_time['date'].dt.to_period('M')
                    df_time['year'] = df_time['date'].dt.year
                    df_time['month'] = df_time['date'].dt.month
                    df_time['weekday'] = df_time['date'].dt.day_name()
                    
                    # Step 4: Group by time and sentiment
                    sentiment_time = df_time.groupby(['month_year', 'sentiment']).size().unstack(fill_value=0)
                    
                    if sentiment_time.empty:
                        st.warning("❌ Tidak ada data untuk trend sentimen")
                    else:
                        # Step 5: Create visualization
                        # Convert PeriodIndex to string for plotting
                        sentiment_time_plot = sentiment_time.copy()
                        sentiment_time_plot.index = sentiment_time_plot.index.astype(str)
                        
                        # Reset index for plotly
                        plot_data = sentiment_time_plot.reset_index()
                        
                        # Melt data for plotly
                        plot_data_melted = plot_data.melt(
                            id_vars=['month_year'], 
                            var_name='sentiment', 
                            value_name='count'
                        )
                        
                        # Create line plot
                        fig_time = px.line(
                            plot_data_melted,
                            x='month_year',
                            y='count',
                            color='sentiment',
                            title="📈 Trend Sentimen Bulanan",
                            labels={
                                'month_year': 'Bulan-Tahun',
                                'count': 'Jumlah Review',
                                'sentiment': 'Sentimen'
                            }
                        )
                        
                        fig_time.update_layout(
                            height=500,
                            xaxis_tickangle=45,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig_time, use_container_width=True)
                        
                        # Step 6: Summary statistics
                        st.subheader("📊 Statistik Trend")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            date_range = f"{df_time['date'].min().strftime('%Y-%m-%d')}"
                            st.metric("Tanggal Mulai", date_range)
                        
                        with col2:
                            date_range = f"{df_time['date'].max().strftime('%Y-%m-%d')}"
                            st.metric("Tanggal Akhir", date_range)
                        
                        with col3:
                            total_months = len(sentiment_time)
                            st.metric("Total Bulan", total_months)
                        
                        with col4:
                            avg_per_month = df_time.groupby('month_year').size().mean()
                            st.metric("Rata-rata Review/Bulan", f"{avg_per_month:.1f}")
                        
                        # Step 7: Monthly breakdown table
                        st.subheader("📋 Detail per Bulan")
                        
                        # Create summary table
                        monthly_summary = df_time.groupby('month_year').agg({
                            'sentiment': 'count',
                            'date': ['min', 'max']
                        }).round(1)
                        
                        monthly_summary.columns = ['Total Reviews', 'First Date', 'Last Date']
                        monthly_summary.index = monthly_summary.index.astype(str)
                        
                        # Add sentiment breakdown
                        sentiment_breakdown = df_time.groupby(['month_year', 'sentiment']).size().unstack(fill_value=0)
                        sentiment_breakdown.index = sentiment_breakdown.index.astype(str)
                        
                        # Combine tables
                        combined_summary = pd.concat([monthly_summary, sentiment_breakdown], axis=1)
                        st.dataframe(combined_summary, use_container_width=True)
                        
                        # Step 8: Additional insights
                        st.subheader("🔍 Insights Tambahan")
                        
                        # Most active month
                        most_active = df_time.groupby('month_year').size().idxmax()
                        most_active_count = df_time.groupby('month_year').size().max()
                        
                        # Sentiment trends
                        positive_trend = sentiment_time.get('positive', pd.Series()).mean()
                        negative_trend = sentiment_time.get('negative', pd.Series()).mean()
                        neutral_trend = sentiment_time.get('neutral', pd.Series()).mean()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(f"📈 **Bulan Paling Aktif:** {most_active} ({most_active_count} reviews)")
                            st.info(f"🎯 **Rata-rata Positive:** {positive_trend:.1f} reviews/bulan")
                        
                        with col2:
                            st.info(f"📉 **Rata-rata Negative:** {negative_trend:.1f} reviews/bulan")
                            st.info(f"😐 **Rata-rata Neutral:** {neutral_trend:.1f} reviews/bulan")
                
            else:
                st.error("❌ Gagal memparse tanggal dengan semua metode")
                st.error("**Error messages:**")
                for msg in error_msgs:
                    st.write(f"- {msg}")
                    
        except Exception as e:
            st.error(f"❌ Error memproses data tanggal: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("ℹ️ Kolom tanggal 'at' tidak ditemukan dalam dataset")
    
    # Text length analysis
    st.subheader("📏 Analisis Panjang Teks")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_length = plot_text_length_distribution(df)
        st.plotly_chart(fig_length, use_container_width=True)
    
    with col2:
        # Box plot by sentiment
        try:
            # Handle NaN values in processed_content
            df_copy = df.copy()
            valid_mask = df_copy['processed_content'].notna()
            df_filtered = df_copy[valid_mask].copy()
            
            if len(df_filtered) > 0:
                df_filtered['text_length'] = df_filtered['processed_content'].astype(str).str.len()
                
                fig_box = px.box(df_filtered, x='sentiment', y='text_length',
                                title="Distribusi Panjang Teks per Sentimen")
                fig_box.update_layout(height=400)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("Tidak ada data valid untuk box plot")
        except Exception as e:
            st.error(f"Error creating box plot: {e}")
    
    # Word clouds by sentiment
    st.subheader("☁️ Word Cloud per Sentimen")
    
    try:
        # Filter valid data first
        df_valid = df[df['processed_content'].notna()].copy()
        df_valid = df_valid[df_valid['processed_content'].astype(str).str.len() > 0]
        
        if len(df_valid) > 0:
            sentiments = df_valid['sentiment'].unique()
            cols = st.columns(len(sentiments))
            
            for i, sentiment in enumerate(sentiments):
                with cols[i]:
                    sentiment_data = df_valid[df_valid['sentiment'] == sentiment]['processed_content']
                    if len(sentiment_data) > 0:
                        fig_wc = create_wordcloud(sentiment_data, f"Word Cloud - {sentiment.title()}")
                        if fig_wc:
                            st.pyplot(fig_wc)
                        else:
                            st.info(f"Tidak cukup data untuk {sentiment}")
                    else:
                        st.warning(f"Tidak ada data untuk sentimen {sentiment}")
        else:
            st.warning("Tidak ada data valid untuk membuat word cloud")
            
    except Exception as e:
        st.error(f"Error creating sentiment word clouds: {e}")
    
    # Top words analysis
    st.subheader("🔤 Analisis Kata Teratas")
    
    try:
        from collections import Counter
        import re
        
        # Handle NaN values and get valid text
        valid_texts = df['processed_content'].dropna().astype(str)
        valid_texts = valid_texts[valid_texts.str.len() > 0]
        
        if len(valid_texts) > 0:
            # Get all words
            all_words = []
            for text in valid_texts:
                all_words.extend(text.split())
            
            word_freq = Counter(all_words)
            
            # Remove single characters and numbers
            word_freq = {word: count for word, count in word_freq.items() 
                        if len(word) > 1 and not word.isdigit()}
            
            # Top words using Counter
            word_counter = Counter(word_freq)
            top_words = dict(word_counter.most_common(20))
            
            if top_words:
                fig_words = px.bar(x=list(top_words.values()), y=list(top_words.keys()),
                                  orientation='h', title="20 Kata Paling Sering Muncul")
                fig_words.update_layout(height=600)
                st.plotly_chart(fig_words, use_container_width=True)
            else:
                st.warning("Tidak ada kata yang ditemukan untuk analisis")
        else:
            st.warning("Tidak ada teks valid untuk analisis kata")
            
    except Exception as e:
        st.error(f"Error dalam analisis kata: {e}")

elif selected_menu == "📝 Text Analysis":
    st.header("📝 Analisis Teks Real-time")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Model belum ditraining. Silakan train model terlebih dahulu.")
        
        # Try to load existing models
        if st.button("🔄 Coba Load Model yang Ada", key="try_load_models_btn"):
            st.session_state.sentiment_system.load_models()
            if (st.session_state.sentiment_system.nb_model or 
                st.session_state.sentiment_system.lstm_model):
                st.session_state.model_trained = True
                st.success("✅ Models loaded!")
                st.rerun()
            else:
                st.error("❌ Tidak ada model yang ditemukan.")
        st.stop()
    
    # Text input
    st.subheader("✍️ Input Teks untuk Dianalisis")
    
    # Sample texts for demo
    sample_texts = [
        "Aplikasi ini sangat bagus dan mudah digunakan!",
        "Pelayanannya mengecewakan, lambat sekali responnya.",
        "Biasa saja, tidak ada yang istimewa.",
        "Fitur-fiturnya lengkap dan interface nya user friendly",
        "Banyak bug dan sering crash, sangat mengganggu"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Update input_text from session state if it was changed by sample buttons
        default_value = st.session_state.get('input_text', '')
        input_text = st.text_area(
            "Masukkan teks di sini:",
            value=default_value,
            height=100,
            placeholder="Contoh: Aplikasi ini sangat bagus dan mudah digunakan!",
            key="text_input_area"
        )
        
        # Update session state when text is manually changed
        if input_text != st.session_state.get('input_text', ''):
            st.session_state.input_text = input_text
    
    with col2:
        st.write("**Contoh teks:**")
        for i, sample in enumerate(sample_texts):
            with st.expander(f"📝 Sample {i+1}"):
                st.write(sample)
                if st.button(f"Gunakan", key=f"use_sample_{i}"):
                    # Set the text in session state and rerun to update text area
                    st.session_state.input_text = sample
                    st.rerun()
    
    # Model selection for prediction
    st.subheader("🤖 Pilih Model untuk Prediksi")
    
    available_models = []
    if st.session_state.sentiment_system.nb_model:
        available_models.append("Naive Bayes")
    if st.session_state.sentiment_system.lstm_model:
        available_models.append("LSTM")
    
    if not available_models:
        st.error("❌ Tidak ada model yang tersedia!")
        st.stop()
    
    selected_model = st.selectbox("Pilih model:", available_models)
    
    # Analyze button
    analyze_clicked = st.button("🔍 Analisis Sentimen", type="primary", key="analyze_text_btn")
    
    if analyze_clicked:
        if input_text.strip():
            with st.spinner("Menganalisis..."):
                try:
                    # Select model type
                    model_type = 'nb' if selected_model == "Naive Bayes" else 'lstm'
                    
                    # Get prediction
                    result = st.session_state.sentiment_system.predict_sentiment(
                        input_text, model_type=model_type
                    )
                    
                    # Display results
                    st.subheader("📊 Hasil Analisis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sentiment = result['prediction']
                        emoji_map = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
                        st.metric("Prediksi Sentimen", 
                                 f"{emoji_map.get(sentiment, '🤔')} {sentiment.title()}")
                    
                    with col2:
                        confidence = result['confidence']
                        st.metric("Confidence Score", f"{confidence:.4f}")
                    
                    with col3:
                        # Determine confidence level
                        if confidence > 0.8:
                            conf_level = "Sangat Tinggi"
                            conf_color = "success"
                        elif confidence > 0.6:
                            conf_level = "Tinggi"
                            conf_color = "info"
                        elif confidence > 0.4:
                            conf_level = "Sedang"
                            conf_color = "warning"
                        else:
                            conf_level = "Rendah"
                            conf_color = "error"
                        
                        st.metric("Tingkat Keyakinan", conf_level)
                    
                    # Probability distribution
                    st.subheader("📈 Distribusi Probabilitas")
                    
                    probs = result['probabilities']
                    prob_df = pd.DataFrame({
                        'Sentimen': list(probs.keys()),
                        'Probabilitas': list(probs.values())
                    })
                    
                    fig_prob = px.bar(prob_df, x='Sentimen', y='Probabilitas',
                                     title="Probabilitas setiap Kelas Sentimen",
                                     color='Probabilitas',
                                     color_continuous_scale='viridis')
                    fig_prob.update_layout(height=400)
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Word importance (for Naive Bayes)
                    if model_type == 'nb':
                        st.subheader("🔤 Kata-kata Penting")
                        
                        word_importance = st.session_state.sentiment_system.get_word_importance(
                            input_text, model_type='nb'
                        )
                        
                        if word_importance:
                            # Create DataFrame for word importance
                            word_df = pd.DataFrame(word_importance, columns=['Kata', 'Skor'])
                            
                            # Display as bar chart
                            fig_words = px.bar(word_df, x='Skor', y='Kata',
                                              orientation='h',
                                              title="Kontribusi Kata terhadap Prediksi")
                            fig_words.update_layout(height=400)
                            st.plotly_chart(fig_words, use_container_width=True)
                            
                            # Highlight important words in text
                            st.subheader("✨ Teks dengan Highlight Kata Penting")
                            
                            highlighted_text = input_text
                            important_words = [word for word, _ in word_importance[:5]]
                            
                            for word in important_words:
                                highlighted_text = highlighted_text.replace(
                                    word, f"**{word}**"
                                )
                            
                            st.markdown(highlighted_text)
                    
                    # Processed text
                    st.subheader("🔧 Teks Setelah Preprocessing")
                    processed = st.session_state.sentiment_system.preprocess_text(input_text)
                    st.code(processed)
                    
                except Exception as e:
                    st.error(f"❌ Error saat analisis: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")
    
    # Batch analysis
    st.subheader("📊 Analisis Batch")
    
    uploaded_file = st.file_uploader(
        "Upload file CSV untuk analisis batch:",
        type=['csv'],
        help="File harus memiliki kolom 'content' yang berisi teks untuk dianalisis"
    )
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            if 'content' not in batch_df.columns:
                st.error("❌ File harus memiliki kolom 'content'")
            else:
                st.success(f"✅ File loaded: {len(batch_df)} baris")
                
                if st.button("🚀 Analisis Semua", key="batch_analyze_btn"):
                    with st.spinner("Menganalisis batch data..."):
                        model_type = 'nb' if selected_model == "Naive Bayes" else 'lstm'
                        
                        predictions = []
                        confidences = []
                        
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(batch_df['content']):
                            result = st.session_state.sentiment_system.predict_sentiment(
                                str(text), model_type=model_type
                            )
                            predictions.append(result['prediction'])
                            confidences.append(result['confidence'])
                            
                            progress_bar.progress((i + 1) / len(batch_df))
                        
                        batch_df['predicted_sentiment'] = predictions
                        batch_df['confidence'] = confidences
                        
                        st.success("✅ Analisis batch selesai!")
                        
                        # Results summary
                        st.subheader("📊 Ringkasan Hasil")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            sentiment_dist = batch_df['predicted_sentiment'].value_counts()
                            fig_batch = px.pie(values=sentiment_dist.values, 
                                             names=sentiment_dist.index,
                                             title="Distribusi Sentimen Hasil Prediksi")
                            st.plotly_chart(fig_batch, use_container_width=True)
                        
                        with col2:
                            avg_confidence = batch_df['confidence'].mean()
                            st.metric("Rata-rata Confidence", f"{avg_confidence:.4f}")
                        
                        with col3:
                            high_conf = (batch_df['confidence'] > 0.8).sum()
                            st.metric("Prediksi High Confidence", f"{high_conf}")
                        
                        # Show results table
                        st.subheader("📋 Hasil Detail")
                        display_cols = ['content', 'predicted_sentiment', 'confidence']
                        if all(col in batch_df.columns for col in display_cols):
                            st.dataframe(batch_df[display_cols].head(20))
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Hasil CSV",
                            data=csv,
                            file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"❌ Error memuat file: {e}")

elif selected_menu == "ℹ️ Tentang":
    st.header("ℹ️ Tentang Sistem")
    
    st.markdown("""
    ## 🎯 Sistem Analisis Sentimen Indonesia
    
    ### 📋 Deskripsi
    Sistem ini merupakan aplikasi web untuk melakukan analisis sentimen pada teks berbahasa Indonesia secara otomatis. 
    Sistem menggunakan kombinasi teknik machine learning tradisional dan deep learning untuk mengklasifikasikan 
    sentimen teks menjadi tiga kategori: **Positif**, **Negatif**, dan **Netral**.
    
    ### 🔧 Teknologi yang Digunakan
    
    #### Preprocessing:
    - **Text Cleaning**: Penghapusan URL, emoji, karakter khusus
    - **Normalisasi**: Konversi kata tidak baku bahasa Indonesia
    - **Tokenisasi**: Pemecahan teks menjadi token
    - **Stopwords Removal**: Penghapusan kata tidak bermakna
    - **Stemming**: Menggunakan library Sastrawi
    
    #### Feature Engineering:
    - **TF-IDF**: Term Frequency-Inverse Document Frequency
    - **Bag of Words**: Representasi frekuensi kata
    - **Word2Vec**: Word embeddings dengan Gensim
    - **FastText**: Subword embeddings
    
    #### Model Machine Learning:
    - **Naive Bayes**: Multinomial Naive Bayes untuk klasifikasi teks
    - **LSTM**: Long Short-Term Memory neural network
    
    #### Evaluasi:
    - **Accuracy**: Tingkat ketepatan prediksi
    - **Precision**: Ketepatan prediksi positif
    - **Recall**: Kemampuan mendeteksi kelas positif
    - **F1-Score**: Harmonic mean precision dan recall
    - **Confusion Matrix**: Matriks evaluasi detail
    - **Cross-Validation**: Validasi silang 5-fold
    
    ### 📊 Dataset
    Dataset yang digunakan berisi ulasan/komentar berbahasa Indonesia dengan struktur:
    - `userName`: Nama pengguna
    - `content`: Konten ulasan/komentar
    - `score`: Skor rating (1-5)
    - `at`: Tanggal ulasan
    - `appVersion`: Versi aplikasi
    
    ### 🚀 Fitur Utama
    
    #### 1. Preprocessing Data
    - Upload dan preprocessing dataset
    - Pilihan jumlah data yang akan diproses
    - Visualisasi statistik preprocessing
    - Penyimpanan data yang sudah diproses
    
    #### 2. Training Model
    - Training multiple models (Naive Bayes & LSTM)
    - Pilihan feature engineering methods
    - Cross-validation evaluation
    - Model persistence (save/load)
    
    #### 3. Visualisasi Dataset
    - Distribusi sentimen
    - Analisis panjang teks
    - Word cloud per sentimen
    - Trend sentimen berdasarkan waktu
    - Top words analysis
    
    #### 4. Text Analysis
    - Prediksi sentimen real-time
    - Confidence score
    - Word importance analysis
    - Batch analysis untuk multiple texts
    - Export hasil analisis
    
    ### 📁 Struktur Sistem
    ```
    sentiment/
    ├── Dataset.csv                 # Dataset mentah
    ├── sentiment_analysis_system.py # Core system
    ├── streamlit_app.py           # Web application
    ├── requirements.txt           # Dependencies
    ├── README.md                 # Documentation
    ├── docs/
    │   └── documentation.md      # Detailed docs
    ├── models/                   # Trained models
    │   ├── naive_bayes_tfidf.pkl
    │   ├── lstm_model.h5
    │   ├── tokenizer.pkl
    │   └── vectorizers...
    └── dataset_sudah/           # Processed datasets
        └── processed_*.csv
    ```
    
    ### 📈 Performance
    Sistem ini telah dioptimalkan untuk:
    - **Akurasi tinggi** pada teks bahasa Indonesia
    - **Preprocessing robust** untuk berbagai jenis teks
    - **Scalability** untuk dataset besar
    - **User-friendly interface** dengan Streamlit
    
    ### 🔮 Future Improvements
    - Support untuk lebih banyak model (BERT, RoBERTa)
    - Implementasi active learning
    - API endpoint untuk integrasi
    - Real-time monitoring dan retraining
    - Support multi-label classification
    
    ### 👨‍💻 Pengembang
    Sistem ini dikembangkan sebagai solusi komprehensif untuk analisis sentimen 
    teks berbahasa Indonesia dengan fokus pada kemudahan penggunaan dan akurasi tinggi.
    
    ### 📞 Support
    Untuk pertanyaan atau masalah teknis, silakan hubungi tim pengembang 
    atau buat issue di repository GitHub.
    """)
    
    # System statistics
    st.subheader("📊 Statistik Sistem")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_count = 0
        if os.path.exists('models'):
            model_count = len([f for f in os.listdir('models') if f.endswith(('.pkl', '.h5'))])
        st.metric("Models Tersimpan", model_count)
    
    with col2:
        dataset_count = 0
        if os.path.exists('dataset_sudah'):
            dataset_count = len([f for f in os.listdir('dataset_sudah') if f.endswith('.csv')])
        st.metric("Dataset Processed", dataset_count)
    
    with col3:
        original_data = 0
        if os.path.exists('Dataset.csv'):
            original_data = len(pd.read_csv('Dataset.csv'))
        st.metric("Data Mentah", f"{original_data:,}")
    
    with col4:
        processed_data = 0
        if st.session_state.processed_data is not None:
            processed_data = len(st.session_state.processed_data)
        st.metric("Data Siap Pakai", f"{processed_data:,}")
    
    # Technical specifications
    st.subheader("⚙️ Spesifikasi Teknis")
    
    tech_specs = {
        "Framework Web": "Streamlit",
        "Machine Learning": "scikit-learn, TensorFlow/Keras",
        "Text Processing": "NLTK, Sastrawi",
        "Word Embeddings": "Gensim (Word2Vec, FastText)",
        "Visualization": "Matplotlib, Seaborn, Plotly",
        "Data Processing": "Pandas, NumPy",
        "Language": "Python 3.7+"
    }
    
    for tech, desc in tech_specs.items():
        st.text(f"• {tech}: {desc}")
    
    # Performance metrics (if available)
    if st.session_state.model_trained:
        st.subheader("🎯 Performance Metrics")
        st.info("Model telah ditraining dan siap digunakan untuk prediksi!")
        
        try:
            # Try to load and display some basic model info
            if os.path.exists('models'):
                model_files = os.listdir('models')
                st.text("Model yang tersedia:")
                for file in model_files:
                    st.text(f"• {file}")
        except:
            pass

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>🎯 Sistem Analisis Sentimen Indonesia</p>
    <p>Dikembangkan dengan ❤️ menggunakan Streamlit dan Python</p>
    <p>© 2024 - Semua hak dilindungi</p>
</div>
""", unsafe_allow_html=True)