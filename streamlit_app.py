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
import time
import json
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

# Import our optimized sentiment analysis system
from sentiment_analysis_system import OptimizedSentimentAnalysisSystem, SentimentAnalysisSystem

# Configure page
st.set_page_config(
    page_title="Sistem Analisis Sentimen",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
    
    .optimization-badge {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    
    .performance-metrics {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with optimized system
if 'sentiment_system' not in st.session_state:
    with st.spinner("Initializing optimized sentiment analysis system..."):
        st.session_state.sentiment_system = OptimizedSentimentAnalysisSystem()

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

if 'processing_time' not in st.session_state:
    st.session_state.processing_time = {}

# Header with optimization badge
st.markdown("""
<div class="main-header">
    <h1>🎯 Sistem Analisis Sentimen Indonesia </h1>
    <p>Analisis sentimen otomatis dengan performa tinggi dan akurasi yang ditingkatkan</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with performance info
st.sidebar.title("📋 Menu Navigasi")

# Performance metrics in sidebar
if st.session_state.processing_time:
    with st.sidebar.expander("⚡ Performance Metrics"):
        for task, timing in st.session_state.processing_time.items():
            st.metric(task, f"{timing:.2f}s")

menu_options = [
    "🏠 Beranda",
    "⚙️ Preprocessing Data", 
    "🤖 Train Model",
    "📊 Visualisasi Dataset",
    "📝 Text Analysis",
    "🔧 System Performance",
    "ℹ️ Tentang"
]

selected_menu = st.sidebar.selectbox("Pilih Menu:", menu_options)

# Helper functions with optimization tracking
def track_time(func):
    """Decorator to track function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        if 'processing_time' not in st.session_state:
            st.session_state.processing_time = {}
        st.session_state.processing_time[func.__name__] = execution_time
        return result
    return wrapper

@track_time
def display_preprocessing_stats(df_original, df_processed):
    """Display optimized preprocessing statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Asli", f"{len(df_original):,}", help="Jumlah data sebelum preprocessing")
    
    with col2:
        delta_count = len(df_processed) - len(df_original)
        st.metric("Data Setelah Preprocessing", f"{len(df_processed):,}", 
                 delta=f"{delta_count:,}")
    
    with col3:
        if 'content' in df_original.columns and 'processed_content' in df_processed.columns:
            avg_length_original = df_original['content'].astype(str).str.len().mean()
            avg_length_processed = df_processed['processed_content'].astype(str).str.len().mean()
            st.metric("Rata-rata Panjang Teks", f"{avg_length_processed:.1f}", 
                     delta=f"{avg_length_processed - avg_length_original:.1f}")
        else:
            st.metric("Rata-rata Panjang Teks", "N/A")
    
    with col4:
        try:
            valid_content = df_processed['processed_content'].dropna().astype(str)
            valid_content = valid_content[valid_content.str.len() > 0]
            if len(valid_content) > 0:
                all_text = ' '.join(valid_content.tolist())
                unique_words = len(set(all_text.split()))
                st.metric("Kosakata Unik", f"{unique_words:,}")
            else:
                st.metric("Kosakata Unik", "0")
        except Exception as e:
            st.metric("Kosakata Unik", "Error")

@track_time
def plot_sentiment_distribution(df):
    """Plot optimized sentiment distribution"""
    try:
        sentiment_counts = df['sentiment'].value_counts()
        
        # Create enhanced pie chart
        fig = px.pie(
            values=sentiment_counts.values, 
            names=sentiment_counts.index,
            title="Distribusi Sentimen Dataset", 
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3  # Donut chart for better aesthetics
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=12
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating sentiment distribution plot: {e}")
        return None

@track_time
def plot_text_length_distribution(df):
    """Plot optimized text length distribution"""
    try:
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
            nbins=min(50, max(10, len(text_lengths) // 20)), 
            title="Distribusi Panjang Teks Setelah Preprocessing",
            labels={'x': 'Panjang Teks (karakter)', 'y': 'Frekuensi'},
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            height=400,
            showlegend=False
        )
        
        # Add statistics
        fig.add_vline(
            x=text_lengths.mean(), 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {text_lengths.mean():.1f}"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating text length distribution: {e}")
        return None

@track_time
def create_wordcloud_optimized(text_data, title="Word Cloud"):
    """Create optimized word cloud"""
    try:
        # Handle NaN values and convert to string
        valid_texts = text_data.dropna().astype(str)
        valid_texts = valid_texts[valid_texts.str.len() > 0]
        
        if len(valid_texts) == 0:
            st.warning("Tidak ada teks valid untuk membuat word cloud")
            return None
        
        # Sample large datasets for performance
        if len(valid_texts) > 1000:
            valid_texts = valid_texts.sample(n=1000, random_state=42)
            st.info(f"Menggunakan sample 1000 teks dari {len(text_data)} untuk word cloud")
        
        # Combine all text
        all_text = ' '.join(valid_texts.tolist())
        
        if len(all_text.strip()) == 0:
            st.warning("Teks kosong, tidak dapat membuat word cloud")
            return None
        
        # Create word cloud with optimized settings
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis',
            min_font_size=10,
            relative_scaling=0.5,
            max_font_size=50
        ).generate(all_text)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {e}")
        return None

@track_time
def plot_confusion_matrix_enhanced(cm, class_names, title="Confusion Matrix"):
    """Plot enhanced confusion matrix"""
    try:
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = px.imshow(
            cm_normalized, 
            labels=dict(x="Predicted", y="Actual", color="Normalized Count"),
            x=class_names, 
            y=class_names,
            color_continuous_scale="Blues",
            title=title,
            aspect="auto"
        )
        
        # Add text annotations with both raw and normalized values
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{cm[i, j]}<br>({cm_normalized[i, j]:.2f})",
                    showarrow=False,
                    font=dict(
                        color="white" if cm_normalized[i, j] > 0.5 else "black",
                        size=10
                    )
                )
        
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        st.error(f"Error creating confusion matrix: {e}")
        return None

@track_time
def display_model_metrics_enhanced(results, model_name):
    """Display enhanced model evaluation metrics"""
    st.subheader(f"📊 Hasil Evaluasi {model_name}")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.4f}")
    
    with col2:
        st.metric("Precision", f"{results['precision']:.4f}")
    
    with col3:
        st.metric("Recall", f"{results['recall']:.4f}")
    
    with col4:
        st.metric("F1-Score", f"{results['f1_score']:.4f}")
    
    # Per-class metrics if available
    if 'precision_per_class' in results:
        st.subheader("📈 Metrics per Kelas")
        
        # Create DataFrame for better visualization
        class_names = ['Negative', 'Neutral', 'Positive']
        if len(results['precision_per_class']) == len(class_names):
            metrics_df = pd.DataFrame({
                'Class': class_names,
                'Precision': results['precision_per_class'],
                'Recall': results['recall_per_class'],
                'F1-Score': results['f1_per_class']
            })
            
            # Display as interactive table
            st.dataframe(metrics_df, use_container_width=True)
            
            # Bar chart for per-class metrics
            fig_metrics = px.bar(
                metrics_df.melt(id_vars=['Class'], var_name='Metric', value_name='Score'),
                x='Class',
                y='Score',
                color='Metric',
                barmode='group',
                title=f"Per-Class Performance - {model_name}"
            )
            fig_metrics.update_layout(height=400)
            st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Enhanced Confusion Matrix
    if 'confusion_matrix' in results:
        class_names = ['Negative', 'Neutral', 'Positive']
        fig_cm = plot_confusion_matrix_enhanced(
            results['confusion_matrix'], 
            class_names, 
            f"Confusion Matrix - {model_name}"
        )
        if fig_cm:
            st.plotly_chart(fig_cm, use_container_width=True)
    
    # Classification Report
    if 'classification_report' in results:
        with st.expander("📋 Detailed Classification Report"):
            st.text(results['classification_report'])

# Main content based on selected menu
if selected_menu == "🏠 Beranda":
    st.markdown("""
    ## 🎯 Selamat Datang di Sistem Analisis Sentimen Indonesia
    
    Sistem ini telah dioptimalkan untuk performa tinggi dan akurasi yang lebih baik dalam menganalisis sentimen teks berbahasa Indonesia.
    
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
    
    # Performance comparison (if available)
    if st.session_state.processing_time:
        st.subheader("⚡ Real-time Performance Metrics")
        
        perf_df = pd.DataFrame([
            {"Task": task, "Time (seconds)": time}
            for task, time in st.session_state.processing_time.items()
        ])
        
        if not perf_df.empty:
            fig_perf = px.bar(
                perf_df,
                x='Task',
                y='Time (seconds)',
                title="Waktu Eksekusi per Task",
                color='Time (seconds)',
                color_continuous_scale='viridis'
            )
            fig_perf.update_layout(height=400)
            st.plotly_chart(fig_perf, use_container_width=True)
    
    # Check if dataset exists
    if os.path.exists('Dataset.csv'):
        st.success("✅ Dataset.csv ditemukan!")
        
        try:
            df_sample = pd.read_csv('Dataset.csv', nrows=5)  # Only read first 5 rows for preview
            st.subheader("👀 Preview Dataset")
            st.dataframe(df_sample)
            
            # Quick dataset info
            df_info = pd.read_csv('Dataset.csv', usecols=['content'])
            total_rows = len(df_info)
            st.info(f"📊 Total data tersedia: {total_rows:,} baris")
            
        except Exception as e:
            st.error(f"Error reading dataset: {e}")
    else:
        st.error("❌ Dataset.csv tidak ditemukan! Pastikan file tersedia di direktori yang sama.")

elif selected_menu == "⚙️ Preprocessing Data":
    st.header("⚙️ Preprocessing Data")
    
    if not os.path.exists('Dataset.csv'):
        st.error("❌ Dataset.csv tidak ditemukan!")
        st.stop()
    
    # Load dataset info efficiently
    try:
        # Only read necessary columns for info
        df_info = pd.read_csv('Dataset.csv', usecols=['content'])
        total_data = len(df_info)
        st.info(f"📊 Dataset tersedia: {total_data:,} baris data")
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        st.stop()
    
    # Sample size selection with recommendations
    st.subheader("📏 Pilih Jumlah Data untuk Diproses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_options = {
            "Semua Data": min(total_data, 225002),
            "25,000 Data": min(25000, total_data),
            "10,000 Data": min(10000, total_data),
            "5,000 Data": min(5000, total_data),
            "1,000 Data": min(1000, total_data),
            "500 Data (Quick Test)": min(500, total_data)
        }
        
        selected_option = st.selectbox("Pilih jumlah data:", list(sample_options.keys()))
        sample_size = sample_options[selected_option]
        
        # Performance recommendation
        if sample_size > 10000:
            st.warning("⚠️ Dataset besar mungkin membutuhkan waktu lebih lama")
        elif sample_size <= 1000:
            st.info("ℹ️ Ukuran kecil, cocok untuk testing cepat")
    
    with col2:
        st.metric("Data yang akan diproses", f"{sample_size:,}")
        if sample_size < total_data:
            st.info(f"Akan mengambil sampel acak dari {total_data:,} data")
        
        # Estimated processing time
        estimated_time = sample_size / 1000 * 2  # Rough estimate: 2 seconds per 1000 rows
        st.metric("Estimasi Waktu", f"{estimated_time:.1f} detik")
    
    # Advanced preprocessing options
    with st.expander("🔧 Opsi Preprocessing Lanjutan"):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.slider("Batch Size", 500, 5000, 1000, step=500,
                                  help="Ukuran batch untuk pemrosesan. Batch lebih besar = lebih cepat tapi butuh RAM lebih")
        
        with col2:
            use_parallel = st.checkbox("Parallel Processing", value=True,
                                     help="Gunakan pemrosesan paralel untuk kecepatan maksimal")
    
    # Preprocessing button
    if st.button("🚀 Mulai Preprocessing", type="primary", key="start_preprocessing_btn"):
        
        # Progress tracking
        progress_container = st.container()
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
        
        start_time = time.time()
        
        try:
            with st.spinner("Sedang memproses data dengan optimisasi..."):
                status_text.text("📂 Loading dataset...")
                progress_bar.progress(10)
                
                # Load and preprocess data with progress tracking
                df_processed = st.session_state.sentiment_system.load_and_preprocess_data(
                    'Dataset.csv', 
                    sample_size=sample_size if sample_size < total_data else None
                )
                
                progress_bar.progress(80)
                status_text.text("💾 Saving processed data...")
                
                # Save processed data with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_dataset_opt_{sample_size}_{timestamp}.csv"
                filepath = st.session_state.sentiment_system.save_processed_data(df_processed, filename)
                
                progress_bar.progress(100)
                
                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time
                st.session_state.processing_time['data_preprocessing'] = processing_time
                
                # Store in session state
                st.session_state.processed_data = df_processed
                
                status_text.empty()
                progress_bar.empty()
                
                # Success message with performance info
                st.markdown(f"""
                <div class="performance-metrics">
                    <h4>✅ Preprocessing Selesai!</h4>
                    <p>📁 Data disimpan: {filepath}</p>
                    <p>⚡ Waktu pemrosesan: {processing_time:.2f} detik</p>
                    <p>🚀 Kecepatan: {sample_size/processing_time:.0f} baris/detik</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display statistics
                st.subheader("📊 Statistik Preprocessing")
                display_preprocessing_stats(df_info.head(sample_size), df_processed)
                
                # Show sample of processed data
                st.subheader("👀 Sample Data Setelah Preprocessing")
                display_cols = ['content', 'processed_content', 'sentiment']
                if all(col in df_processed.columns for col in display_cols):
                    sample_data = df_processed[display_cols].head(10)
                    st.dataframe(sample_data, use_container_width=True)
                
                # Enhanced sentiment distribution
                st.subheader("📈 Distribusi Sentimen")
                fig_sentiment = plot_sentiment_distribution(df_processed)
                if fig_sentiment:
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                # Performance metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Processing Rate", f"{sample_size/processing_time:.0f} rows/sec")
                
                with col2:
                    memory_usage = df_processed.memory_usage(deep=True).sum() / 1024**2
                    st.metric("Memory Usage", f"{memory_usage:.1f} MB")
                
                with col3:
                    compression_ratio = len(df_processed) / sample_size * 100
                    st.metric("Data Retention", f"{compression_ratio:.1f}%")
                
                # Text length analysis
                st.subheader("📏 Analisis Panjang Teks")
                fig_length = plot_text_length_distribution(df_processed)
                if fig_length:
                    st.plotly_chart(fig_length, use_container_width=True)
                
                # Word cloud
                st.subheader("☁️ Word Cloud")
                fig_wc = create_wordcloud_optimized(df_processed['processed_content'], "Word Cloud - Processed Text")
                if fig_wc:
                    st.pyplot(fig_wc)
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error saat preprocessing: {e}")
            import traceback
            with st.expander("🔍 Debug Information"):
                st.code(traceback.format_exc())
    
    # Show existing processed files with metadata
    if os.path.exists('dataset_sudah') and os.listdir('dataset_sudah'):
        st.subheader("📁 File yang Sudah Diproses")
        
        processed_files = [f for f in os.listdir('dataset_sudah') if f.endswith('.csv')]
        
        if processed_files:
            # Create file info table
            file_info = []
            for file in processed_files:
                filepath = os.path.join('dataset_sudah', file)
                try:
                    # Check for metadata
                    metadata_path = filepath.replace('.csv', '_metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        file_info.append({
                            'File': file,
                            'Rows': metadata.get('rows', 'N/A'),
                            'Date': metadata.get('processed_date', 'N/A')[:10] if metadata.get('processed_date') else 'N/A',
                            'Sentiment Dist': str(metadata.get('sentiment_distribution', {}))
                        })
                    else:
                        # Read file for basic info
                        df_temp = pd.read_csv(filepath, nrows=1)
                        file_info.append({
                            'File': file,
                            'Rows': 'Unknown',
                            'Date': 'N/A',
                            'Sentiment Dist': 'N/A'
                        })
                except Exception as e:
                    file_info.append({
                        'File': file,
                        'Rows': 'Error',
                        'Date': 'Error',
                        'Sentiment Dist': 'Error'
                    })
            
            file_df = pd.DataFrame(file_info)
            st.dataframe(file_df, use_container_width=True)
            
            selected_file = st.selectbox("Pilih file untuk dimuat:", processed_files)
            
            if st.button("📂 Muat Data Processed", key="load_processed_btn"):
                try:
                    filepath = os.path.join('dataset_sudah', selected_file)
                    df_loaded = pd.read_csv(filepath)
                    st.session_state.processed_data = df_loaded
                    st.success(f"✅ Data loaded: {selected_file}")
                    st.dataframe(df_loaded.head(), use_container_width=True)
                    
                    # Show metadata if available
                    metadata_path = filepath.replace('.csv', '_metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        with st.expander("📋 Metadata"):
                            st.json(metadata)
                            
                except Exception as e:
                    st.error(f"Error loading file: {e}")

elif selected_menu == "🤖 Train Model":
    st.header("🤖 Training Model")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Belum ada data yang diproses. Silakan lakukan preprocessing terlebih dahulu.")
        st.stop()
    
    df = st.session_state.processed_data
    st.success(f"✅ Data siap untuk training: {len(df):,} baris")
    
    # Data quality check
    sentiment_counts = df['sentiment'].value_counts()
    imbalance_ratio = sentiment_counts.max() / sentiment_counts.min()
    
    if imbalance_ratio > 3:
        st.warning(f"⚠️ Dataset tidak seimbang (rasio: {imbalance_ratio:.1f}). Sistem akan menggunakan class balancing otomatis.")
    
    # Model selection with enhanced options
    st.subheader("🎯 Pilih Model untuk Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_nb = st.checkbox("🔢 Train Balanced Naive Bayes", value=True)
        if train_nb:
            nb_feature = st.selectbox("Feature untuk Naive Bayes:", ["TF-IDF", "Bag of Words"])
    
    with col2:
        train_lstm = st.checkbox("🧠 Train LSTM", value=True)
        if train_lstm:
            lstm_epochs = st.slider("LSTM Epochs", 5, 25, 20)
    
    # Enhanced training parameters
    st.subheader("⚙️ Parameter Training")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size (%)", 10, 30, 20) / 100
    
    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    with col3:
        random_state = st.number_input("Random State", 1, 100, 42)
    
    # Advanced options
    with st.expander("🔧 Opsi Training Lanjutan"):
        col1, col2 = st.columns(2)
        
        with col1:
            use_early_stopping = st.checkbox("Early Stopping (LSTM)", value=True)
            patience = st.slider("Patience", 3, 10, 5) if use_early_stopping else 5
        
        with col2:
            save_best_only = st.checkbox("Save Best Model Only", value=True)
            enable_callbacks = st.checkbox("Enable Advanced Callbacks", value=True)
    
    # Training button with progress tracking
    if st.button("🚀 Mulai Training", type="primary", key="start_training_btn"):
        
        # Progress tracking
        total_steps = (int(train_nb) + int(train_lstm)) * 4  # Rough estimate
        current_step = 0
        
        progress_container = st.container()
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
        
        results = {}
        start_time = time.time()
        
        try:
            # Create features with improved error handling
            status_text.text("🔧 Membuat features...")
            
            try:
                features = st.session_state.sentiment_system.create_features_optimized(df)
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Get cleaned dataframe from features
                clean_df = features.get('clean_df', df)
                
                st.success(f"✅ Features berhasil dibuat untuk {len(clean_df)} data valid")
                
            except Exception as e:
                st.error(f"❌ Error saat membuat features: {e}")
                progress_bar.empty()
                status_text.empty()
                st.stop()
            
            # Prepare data with validation
            from sklearn.model_selection import train_test_split
            
            # Train Naive Bayes
            if train_nb:
                status_text.text(f"🔢 Training Balanced Naive Bayes dengan {nb_feature}...")
                
                try:
                    # Select features
                    if nb_feature == "TF-IDF":
                        X = features['tfidf']
                    else:
                        X = features['bow']
                    
                    y = clean_df['sentiment']
                    
                    # Validate data shapes
                    if X.shape[0] != len(y):
                        st.error(f"❌ Data shape mismatch: X={X.shape[0]}, y={len(y)}")
                        st.stop()
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    # Train balanced model
                    nb_model = st.session_state.sentiment_system.train_naive_bayes_balanced(
                        X_train, y_train, nb_feature.lower().replace('-', '').replace(' ', '')
                    )
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    # Evaluate model
                    nb_results = st.session_state.sentiment_system.evaluate_model_detailed(
                        nb_model, X_test, y_test, "Balanced Naive Bayes"
                    )
                    
                    # Cross validation
                    status_text.text("🔄 Melakukan cross-validation...")
                    cv_scores = st.session_state.sentiment_system.cross_validate_model(
                        nb_model, X, y, cv=cv_folds
                    )
                    nb_results['cv_scores'] = cv_scores
                    nb_results['cv_mean'] = cv_scores.mean()
                    nb_results['cv_std'] = cv_scores.std()
                    
                    results['naive_bayes'] = nb_results
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    st.success("✅ Naive Bayes training selesai!")
                    
                except Exception as e:
                    st.error(f"❌ Error saat training Naive Bayes: {e}")
                    import traceback
                    with st.expander("🔍 Debug Naive Bayes"):
                        st.code(traceback.format_exc())
            
            # Train LSTM
            if train_lstm:
                status_text.text("🧠 Training LSTM...")
                
                try:
                    # Prepare LSTM data with validation
                    X_lstm, y_lstm = st.session_state.sentiment_system.prepare_lstm_data_balanced(
                        clean_df['processed_content'], clean_df['sentiment']
                    )
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    # Split data
                    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                        X_lstm, y_lstm, test_size=test_size, random_state=random_state
                    )
                    
                    # Further split for validation
                    X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
                        X_train_lstm, y_train_lstm, test_size=0.2, random_state=random_state
                    )
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    # Train model
                    status_text.text("🧠 Training LSTM model...")
                    lstm_history = st.session_state.sentiment_system.train_lstm_optimized(
                        X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm
                    )
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    # Evaluate model
                    lstm_results = st.session_state.sentiment_system.evaluate_model_detailed(
                        st.session_state.sentiment_system.lstm_model, 
                        X_test_lstm, y_test_lstm, "Optimized LSTM", is_lstm=True
                    )
                    
                    results['lstm'] = lstm_results
                    results['lstm_history'] = lstm_history
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    st.success("✅ LSTM training selesai!")
                    
                except Exception as e:
                    st.error(f"❌ Error saat training LSTM: {e}")
                    import traceback
                    with st.expander("🔍 Debug LSTM"):
                        st.code(traceback.format_exc())
            
            # Save vectorizers
            try:
                st.session_state.sentiment_system.save_vectorizers_optimized()
                st.success("✅ Models dan vectorizers berhasil disimpan!")
            except Exception as e:
                st.warning(f"⚠️ Warning: Error saving vectorizers: {e}")
            
            # Calculate total training time
            training_time = time.time() - start_time
            st.session_state.processing_time['model_training'] = training_time
            
            # Mark as trained
            st.session_state.model_trained = True
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            # Success message with performance info
            st.markdown(f"""
            <div class="performance-metrics">
                <h4>✅ Training Selesai!</h4>
                <p>⚡ Total waktu training: {training_time:.2f} detik</p>
                <p>🎯 Model berhasil dilatih dengan optimisasi</p>
                <p>📊 Data yang digunakan: {len(clean_df)} baris (dari {len(df)} baris asli)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display enhanced results
            for model_name, model_results in results.items():
                if model_name != 'lstm_history':
                    display_model_metrics_enhanced(model_results, model_name.replace('_', ' ').title())
                    
                    # Cross-validation results for NB
                    if model_name == 'naive_bayes' and 'cv_scores' in model_results:
                        st.subheader("🔄 Cross-Validation Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("CV Mean", f"{model_results['cv_mean']:.4f}")
                        with col2:
                            st.metric("CV Std", f"{model_results['cv_std']:.4f}")
                        with col3:
                            st.metric("CV Range", f"±{model_results['cv_std']*2:.4f}")
                        
                        # CV scores plot
                        fig_cv = px.bar(
                            x=[f"Fold {i+1}" for i in range(len(model_results['cv_scores']))],
                            y=model_results['cv_scores'],
                            title="Cross-Validation Scores",
                            color=model_results['cv_scores'],
                            color_continuous_scale='viridis'
                        )
                        fig_cv.add_hline(
                            y=model_results['cv_mean'],
                            line_dash="dash",
                            annotation_text=f"Mean: {model_results['cv_mean']:.4f}"
                        )
                        st.plotly_chart(fig_cv, use_container_width=True)
            
            # Enhanced LSTM training history
            if 'lstm_history' in results:
                st.subheader("📈 LSTM Training History")
                
                history = results['lstm_history'].history
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Model Accuracy', 'Model Loss')
                )
                
                # Accuracy plot
                fig.add_trace(
                    go.Scatter(
                        y=history['accuracy'], 
                        name='Train Accuracy',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        y=history['val_accuracy'], 
                        name='Validation Accuracy',
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
                
                # Loss plot
                fig.add_trace(
                    go.Scatter(
                        y=history['loss'], 
                        name='Train Loss',
                        line=dict(color='blue')
                    ),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(
                        y=history['val_loss'], 
                        name='Validation Loss',
                        line=dict(color='red')
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=True)
                fig.update_xaxes(title_text="Epoch")
                fig.update_yaxes(title_text="Accuracy", row=1, col=1)
                fig.update_yaxes(title_text="Loss", row=1, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Training insights
                st.subheader("🔍 Training Insights")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    best_val_acc = max(history['val_accuracy'])
                    st.metric("Best Validation Accuracy", f"{best_val_acc:.4f}")
                
                with col2:
                    final_train_acc = history['accuracy'][-1]
                    final_val_acc = history['val_accuracy'][-1]
                    overfitting = final_train_acc - final_val_acc
                    st.metric("Overfitting Score", f"{overfitting:.4f}")
                
                with col3:
                    epochs_trained = len(history['accuracy'])
                    st.metric("Epochs Trained", epochs_trained)
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error saat training: {e}")
            import traceback
            with st.expander("🔍 Debug Information"):
                st.code(traceback.format_exc())
    
    # Enhanced model management
    if os.path.exists('models') and os.listdir('models'):
        st.subheader("💾 Manajemen Model")
        
        model_files = [f for f in os.listdir('models') if f.endswith(('.pkl', '.h5'))]
        
        if model_files:
            # Create model info table
            model_info = []
            for file in model_files:
                filepath = os.path.join('models', file)
                file_size = os.path.getsize(filepath) / 1024  # KB
                mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                model_info.append({
                    'Model File': file,
                    'Size (KB)': f"{file_size:.1f}",
                    'Modified': mod_time.strftime('%Y-%m-%d %H:%M'),
                    'Type': 'Optimized' if 'optimized' in file or 'balanced' in file else 'Standard'
                })
            
            model_df = pd.DataFrame(model_info)
            st.dataframe(model_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📂 Load Models", key="load_optimized_models_btn"):
                    try:
                        st.session_state.sentiment_system.load_models_optimized()
                        st.session_state.model_trained = True
                        st.success("✅ models loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading models: {e}")
            
            with col2:
                if st.button("🧹 Clean Old Models", key="clean_models_btn"):
                    # Keep only optimized/balanced models
                    cleaned = 0
                    for file in model_files:
                        if 'optimized' not in file and 'balanced' not in file:
                            try:
                                os.remove(os.path.join('models', file))
                                cleaned += 1
                            except:
                                pass
                    
                    if cleaned > 0:
                        st.success(f"✅ Cleaned {cleaned} old model files")
                    else:
                        st.info("ℹ️ No old models to clean")

elif selected_menu == "📊 Visualisasi Dataset":
    st.header("📊 Visualisasi Dataset (Enhanced)")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Belum ada data yang diproses. Silakan lakukan preprocessing terlebih dahulu.")
        st.stop()
    
    df = st.session_state.processed_data
    
    # Enhanced dataset overview
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
                # Sample for performance on large datasets
                sample_texts = valid_texts.sample(n=min(1000, len(valid_texts)), random_state=42)
                all_text = ' '.join(sample_texts.tolist())
                unique_words = len(set(all_text.split()))
                st.metric("Kosakata Unik", f"{unique_words:,}")
            else:
                st.metric("Kosakata Unik", "0")
        except Exception as e:
            st.metric("Kosakata Unik", "Error")
    
    with col4:
        try:
            sentiment_balance = df['sentiment'].value_counts().std()
            st.metric("Balance Score", f"{sentiment_balance:.2f}")
        except:
            st.metric("Balance Score", "N/A")
    
    # Enhanced sentiment distribution with insights
    st.subheader("📈 Distribusi Sentimen")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_sentiment = plot_sentiment_distribution(df)
        if fig_sentiment:
            st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        sentiment_counts = df['sentiment'].value_counts()
        st.write("**Insight Distribusi:**")
        
        total = len(df)
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total) * 100
            st.write(f"• {sentiment.title()}: {count:,} ({percentage:.1f}%)")
        
        # Balance analysis
        max_count = sentiment_counts.max()
        min_count = sentiment_counts.min()
        balance_ratio = max_count / min_count
        
        if balance_ratio > 3:
            st.warning(f"⚠️ Dataset tidak seimbang (rasio: {balance_ratio:.1f})")
        elif balance_ratio < 1.5:
            st.success("✅ Dataset cukup seimbang")
        else:
            st.info("ℹ️ Dataset agak tidak seimbang")
        
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
                        fig_wc = create_wordcloud_optimized(sentiment_data, f"Word Cloud - {sentiment.title()}")
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
    st.header("📝 Analisis Teks Real-time (Enhanced)")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Model belum ditraining. Silakan train model terlebih dahulu.")
        
        # Try to load existing models
        if st.button("🔄 Coba Load Model yang Ada", key="try_load_models_btn"):
            st.session_state.sentiment_system.load_models_optimized()
            if (st.session_state.sentiment_system.nb_model or 
                st.session_state.sentiment_system.lstm_model):
                st.session_state.model_trained = True
                st.success("✅ Models loaded!")
                st.rerun()
            else:
                st.error("❌ Tidak ada model yang ditemukan.")
        st.stop()
    
    # Enhanced text input section
    st.subheader("✍️ Input Teks untuk Dianalisis")
    
    # Enhanced sample texts
    sample_texts = [
        "Aplikasi ini sangat bagus dan mudah digunakan! Fitur-fiturnya lengkap dan responsif.",
        "Pelayanannya mengecewakan, lambat sekali responnya dan sering error.",
        "Biasa saja, tidak ada yang istimewa tapi juga tidak buruk.",
        "Driver sangat ramah dan profesional, tepat waktu dan kendaraan bersih.",
        "Banyak bug dan sering crash, sangat mengganggu aktivitas sehari-hari.",
        "Harga terjangkau dengan kualitas pelayanan yang memuaskan.",
        "Interface aplikasi rumit dan tidak user-friendly, susah dipahami.",
        "Promo menarik dan pembayaran mudah, akan menggunakan lagi.",
        "Fitur GPS tidak akurat, sering salah lokasi dan memakan waktu.",
        "Customer service responsif dan membantu menyelesaikan masalah dengan cepat."
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Enhanced text input with character counting
        default_value = st.session_state.get('input_text', '')
        input_text = st.text_area(
            "Masukkan teks di sini:",
            value=default_value,
            height=120,
            placeholder="Contoh: Aplikasi ini sangat bagus dan mudah digunakan! Fitur-fiturnya lengkap dan responsif.",
            key="text_input_area",
            help="Masukkan teks bahasa Indonesia untuk dianalisis sentimennya"
        )
        
        # Character and word count
        if input_text:
            char_count = len(input_text)
            word_count = len(input_text.split())
            st.caption(f"📊 {char_count} karakter, {word_count} kata")
        
        # Update session state when text is manually changed
        if input_text != st.session_state.get('input_text', ''):
            st.session_state.input_text = input_text
    
    with col2:
        st.write("**Contoh teks:**")
        for i, sample in enumerate(sample_texts[:5]):  # Show first 5
            with st.expander(f"📝 Sample {i+1}"):
                st.write(sample[:100] + "..." if len(sample) > 100 else sample)
                if st.button(f"Gunakan", key=f"use_sample_{i}"):
                    st.session_state.input_text = sample
                    st.rerun()
        
        # Show more samples button
        if st.button("📋 Lihat Semua Sample"):
            with st.expander("📝 Semua Sample Teks", expanded=True):
                for i, sample in enumerate(sample_texts[5:], 6):
                    st.write(f"**Sample {i}:** {sample}")
                    if st.button(f"Gunakan Sample {i}", key=f"use_sample_extended_{i}"):
                        st.session_state.input_text = sample
                        st.rerun()
    
    # Enhanced model selection
    st.subheader("🤖 Pilih Model untuk Prediksi")
    
    available_models = []
    if st.session_state.sentiment_system.nb_model:
        available_models.append("Balanced Naive Bayes")
    if st.session_state.sentiment_system.lstm_model:
        available_models.append("Optimized LSTM")
    
    if not available_models:
        st.error("❌ Tidak ada model yang tersedia!")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox("Pilih model:", available_models)
    
    with col2:
        show_confidence = st.checkbox("Tampilkan confidence details", value=True)
    
    # Enhanced analyze button with real-time processing
    analyze_clicked = st.button("🔍 Analisis Sentimen", type="primary", key="analyze_text_btn")
    
    if analyze_clicked:
        if input_text.strip():
            start_time = time.time()
            
            with st.spinner("Menganalisis sentimen..."):
                try:
                    # Select model type
                    model_type = 'nb' if 'Naive Bayes' in selected_model else 'lstm'
                    
                    # Get prediction with improved method
                    result = st.session_state.sentiment_system.predict_sentiment_improved(
                        input_text, model_type=model_type
                    )
                    
                    analysis_time = time.time() - start_time
                    st.session_state.processing_time['text_analysis'] = analysis_time
                    
                    # Enhanced results display
                    st.subheader("📊 Hasil Analisis")
                    
                    # Main result with enhanced styling
                    sentiment = result['prediction']
                    confidence = result['confidence']
                    
                    emoji_map = {
                        'positive': '😊', 
                        'negative': '😞', 
                        'neutral': '😐'
                    }
                    
                    color_map = {
                        'positive': '#28a745',
                        'negative': '#dc3545', 
                        'neutral': '#ffc107'
                    }
                    
                    # Large result display
                    st.markdown(f"""
                    <div style="
                        background: {color_map.get(sentiment, '#6c757d')};
                        color: white;
                        padding: 2rem;
                        border-radius: 15px;
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    ">
                        <h2>{emoji_map.get(sentiment, '🤔')} {sentiment.upper()}</h2>
                        <h4>Confidence: {confidence:.2%}</h4>
                        <p>Analysis time: {analysis_time:.3f} seconds</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Prediksi", f"{sentiment.title()}")
                    
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.4f}")
                    
                    with col3:
                        # Determine confidence level with enhanced categories
                        if confidence > 0.9:
                            conf_level = "Sangat Tinggi"
                        elif confidence > 0.8:
                            conf_level = "Tinggi"
                        elif confidence > 0.6:
                            conf_level = "Sedang"
                        elif confidence > 0.4:
                            conf_level = "Rendah"
                        else:
                            conf_level = "Sangat Rendah"
                        
                        st.metric("Tingkat Keyakinan", conf_level)
                    
                    with col4:
                        st.metric("Processing Speed", f"{len(input_text)/analysis_time:.0f} char/sec")
                    
                    # Enhanced probability distribution
                    st.subheader("📈 Distribusi Probabilitas")
                    
                    probs = result['probabilities']
                    prob_df = pd.DataFrame({
                        'Sentimen': [k.title() for k in probs.keys()],
                        'Probabilitas': list(probs.values()),
                        'Persentase': [f"{v:.1%}" for v in probs.values()]
                    })
                    
                    # Create enhanced probability chart
                    fig_prob = px.bar(
                        prob_df, 
                        x='Sentimen', 
                        y='Probabilitas',
                        title="Probabilitas setiap Kelas Sentimen",
                        color='Probabilitas',
                        color_continuous_scale='viridis',
                        text='Persentase'
                    )
                    
                    fig_prob.update_traces(textposition='outside')
                    fig_prob.update_layout(
                        height=400,
                        showlegend=False,
                        yaxis=dict(tickformat='.0%')
                    )
                    
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Show confidence details if requested
                    if show_confidence:
                        with st.expander("🔍 Detail Confidence Analysis"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Probability Breakdown:**")
                                for sentiment_name, prob in probs.items():
                                    st.write(f"• {sentiment_name.title()}: {prob:.4f} ({prob:.1%})")
                            
                            with col2:
                                st.write("**Model Information:**")
                                st.write(f"• Model: {selected_model}")
                                st.write(f"• Processing time: {analysis_time:.3f}s")
                                st.write(f"• Text length: {len(input_text)} chars")
                                st.write(f"• Word count: {len(input_text.split())} words")
                    
                    # Word importance (for Naive Bayes)
                    if model_type == 'nb':
                        st.subheader("🔤 Analisis Kata Penting")
                        
                        word_importance = st.session_state.sentiment_system.get_word_importance(
                            input_text, model_type='nb', top_n=15
                        )
                        
                        if word_importance:
                            # Create enhanced DataFrame for word importance
                            word_df = pd.DataFrame(word_importance, columns=['Kata', 'Skor TF-IDF'])
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Enhanced bar chart
                                fig_words = px.bar(
                                    word_df, 
                                    x='Skor TF-IDF', 
                                    y='Kata',
                                    orientation='h',
                                    title="Kontribusi Kata terhadap Prediksi",
                                    color='Skor TF-IDF',
                                    color_continuous_scale='viridis'
                                )
                                fig_words.update_layout(height=500)
                                st.plotly_chart(fig_words, use_container_width=True)
                            
                            with col2:
                                st.write("**Top Words Table:**")
                                word_df['Rank'] = range(1, len(word_df) + 1)
                                st.dataframe(
                                    word_df[['Rank', 'Kata', 'Skor TF-IDF']], 
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            # Highlight important words in text
                            st.subheader("✨ Teks dengan Highlight Kata Penting")
                            
                            highlighted_text = input_text
                            important_words = [word for word, _ in word_importance[:5]]
                            
                            for word in important_words:
                                highlighted_text = highlighted_text.replace(
                                    word, f"**{word}**"
                                )
                            
                            st.markdown(highlighted_text)
                    
                    # Processed text analysis
                    st.subheader("🔧 Analisis Preprocessing")
                    
                    processed = st.session_state.sentiment_system.preprocess_single_text(input_text)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Teks Asli:**")
                        st.text_area("", value=input_text, height=100, disabled=True, key="original_text_display")
                    
                    with col2:
                        st.write("**Teks Setelah Preprocessing:**")
                        st.text_area("", value=processed, height=100, disabled=True, key="processed_text_display")
                    
                    # Preprocessing insights
                    original_words = len(input_text.split())
                    processed_words = len(processed.split()) if processed else 0
                    reduction_rate = (original_words - processed_words) / original_words * 100 if original_words > 0 else 0
                    
                    st.write(f"📊 **Preprocessing Stats:** {original_words} → {processed_words} kata ({reduction_rate:.1f}% reduction)")
                    
                except Exception as e:
                    st.error(f"❌ Error saat analisis: {e}")
                    import traceback
                    with st.expander("🔍 Debug Information"):
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")

elif selected_menu == "🔧 System Performance":
    st.header("🔧 System Performance Monitor")
    
    # Performance overview
    if st.session_state.processing_time:
        st.subheader("⚡ Performance Metrics")
        
        perf_df = pd.DataFrame([
            {"Task": task.replace('_', ' ').title(), "Time (seconds)": time}
            for task, time in st.session_state.processing_time.items()
        ])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_perf = px.bar(
                perf_df,
                x='Task',
                y='Time (seconds)',
                title="Waktu Eksekusi per Task",
                color='Time (seconds)',
                color_continuous_scale='viridis'
            )
            fig_perf.update_layout(height=400)
            st.plotly_chart(fig_perf, use_container_width=True)
        
        with col2:
            st.write("**Performance Summary:**")
            for _, row in perf_df.iterrows():
                st.metric(row['Task'], f"{row['Time (seconds)']:.3f}s")
    else:
        st.info("📊 Belum ada data performance. Lakukan beberapa operasi untuk melihat metrics.")
    
    # System information
    st.subheader("💻 System Information")
    
    import psutil
    import platform
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**System:**")
        st.write(f"• OS: {platform.system()}")
        st.write(f"• Python: {platform.python_version()}")
        st.write(f"• CPU Cores: {psutil.cpu_count()}")
    
    with col2:
        st.write("**Memory:**")
        memory = psutil.virtual_memory()
        st.write(f"• Total: {memory.total / 1024**3:.1f} GB")
        st.write(f"• Used: {memory.percent}%")
        st.write(f"• Available: {memory.available / 1024**3:.1f} GB")
    
    with col3:
        st.write("**Models:**")
        model_count = 0
        if os.path.exists('models'):
            model_count = len([f for f in os.listdir('models') if f.endswith(('.pkl', '.h5'))])
        st.metric("Saved Models", model_count)
        
        dataset_count = 0
        if os.path.exists('dataset_sudah'):
            dataset_count = len([f for f in os.listdir('dataset_sudah') if f.endswith('.csv')])
        st.metric("Processed Datasets", dataset_count)

elif selected_menu == "ℹ️ Tentang":
    st.header("ℹ️ Tentang Sistem")
    
    st.markdown("""
    ## 🎯 Sistem Analisis Sentimen Indonesia

    ### Dibuat Oleh Kelompok 3 :
    #### Akhmad Aditya Rachman  223020503085
    #### Natalio Valentino  223020503115       
    #### Adi Kristianto  223020503127      
    #### Achmad Kahlil Gibran  223020503137    
    #### Aditya Heru Saputra  223020503149        
    #### Andika Fikri Maulana  223020503153    
    #### Muhammad Afrizal  223020503159       
    
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
    """)
    
    # Performance comparison
    if st.session_state.processing_time:
        st.subheader("📊 Current Session Performance")
        
        for task, timing in st.session_state.processing_time.items():
            st.metric(task.replace('_', ' ').title(), f"{timing:.3f} seconds")
    
    # Version information
    st.subheader("📋 Version Information")
    
    version_info = {
        "System Version": "1.0",
        "Release Date": "29-Mei-2025",
    }
    
    for key, value in version_info.items():
        st.write(f"• **{key}**: {value}")

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>🎯 Sistem Analisis Sentimen Indonesia</p>
    <p>Enhanced with ⚡ Performance Optimization & 🎯 Improved Accuracy</p>
</div>
""", unsafe_allow_html=True)