import pandas as pd
import numpy as np
import re
import os
import pickle
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Text preprocessing libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Machine Learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Deep Learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, SpatialDropout1D, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Word embeddings
from gensim.models import Word2Vec, FastText

# Download required NLTK data
def download_nltk_data():
    """Download all required NLTK data"""
    required_nltk_data = [
        'punkt',
        'punkt_tab', 
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    for data_name in required_nltk_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}')
        except LookupError:
            try:
                nltk.download(data_name, quiet=True)
                print(f"Downloaded NLTK data: {data_name}")
            except:
                pass
        except:
            try:
                nltk.download(data_name, quiet=True)
                print(f"Downloaded NLTK data: {data_name}")
            except:
                pass

# Download NLTK data
download_nltk_data()

class OptimizedSentimentAnalysisSystem:
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.word2vec_model = None
        self.fasttext_model = None
        self.nb_model = None
        self.lstm_model = None
        self.tokenizer = None
        self.max_length = 100
        self.class_weights = None
        
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('dataset_sudah', exist_ok=True)
        
        # Enhanced Indonesian slang dictionary
        self.slang_dict = {
            # Negations (penting untuk sentiment)
            'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
            'gk': 'tidak', 'g': 'tidak', 'nda': 'tidak', 'ndak': 'tidak',
            'kagak': 'tidak', 'kaga': 'tidak', 'enggak': 'tidak',
            
            # Time and state
            'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah', 'dh': 'sudah',
            'blm': 'belum', 'blom': 'belum', 'belom': 'belum',
            'lg': 'lagi', 'lgi': 'lagi', 'lgsg': 'langsung',
            
            # Common words
            'yg': 'yang', 'krn': 'karena', 'dgn': 'dengan', 'utk': 'untuk',
            'tdk': 'tidak', 'jd': 'jadi', 'jgn': 'jangan', 'tp': 'tetapi',
            'ttg': 'tentang', 'sm': 'sama', 'dr': 'dari', 'ke': 'ke',
            'di': 'di', 'pd': 'pada', 'sbg': 'sebagai', 'stlh': 'setelah',
            
            # Intensifiers
            'bgt': 'banget', 'bener': 'benar', 'emg': 'memang', 'bngt': 'banget',
            'bgt': 'banget', 'bner': 'benar', 'bnr': 'benar',
            
            # Questions
            'gmn': 'bagaimana', 'gmna': 'bagaimana', 'knp': 'kenapa',
            'knpa': 'kenapa', 'kpn': 'kapan', 'dmn': 'dimana',
            
            # Actions and states
            'hrs': 'harus', 'hbs': 'habis', 'abis': 'habis',
            'org': 'orang', 'orng': 'orang', 'ornag': 'orang',
            'klo': 'kalau', 'kalo': 'kalau', 'kl': 'kalau',
            
            # Pronouns
            'sy': 'saya', 'gw': 'saya', 'gue': 'saya', 'aku': 'saya',
            'w': 'saya', 'ane': 'saya', 'ana': 'saya',
            'lu': 'kamu', 'lo': 'kamu', 'km': 'kamu', 'ente': 'kamu',
            'u': 'kamu', 'mu': 'kamu',
            
            # Actions
            'dgr': 'dengar', 'liat': 'lihat', 'tau': 'tahu', 'tw': 'tahu',
            'mau': 'mau', 'mo': 'mau', 'pgn': 'pengen', 'pen': 'pengen',
            
            # Modern slang
            'wkwk': 'haha', 'kwkw': 'haha', 'wkwkwk': 'haha',
            'hehe': 'haha', 'hihi': 'haha', 'xixi': 'haha'
        }
        
        # Enhanced sentiment keywords
        self.positive_keywords = {
            # Basic positive
            'bagus', 'baik', 'suka', 'senang', 'puas', 'mantap', 'oke', 'ok',
            'recommended', 'rekomendasi', 'top', 'keren', 'asik', 'asyik',
            'love', 'like', 'good', 'great', 'excellent', 'amazing', 'awesome',
            'hebat', 'luar biasa', 'sempurna', 'memuaskan', 'terbaik', 'best',
            
            # Service quality
            'cepat', 'tepat waktu', 'responsif', 'ramah', 'sopan', 'profesional',
            'membantu', 'terbantu', 'bermanfaat', 'mudah', 'praktis', 'efisien',
            'nyaman', 'aman', 'amanah', 'terpercaya', 'reliable', 'lancar',
            'smooth', 'gercep', 'sigap', 'mantul', 'jos', 'gandos',
            
            # Appreciation
            'terima kasih', 'thanks', 'makasih', 'appreciate', 'grateful',
            'sukses', 'success', 'berhasil', 'perfect', 'nice', 'cool',
            'mantabs', 'mantaap', 'mantabb', 'kece', 'juara', 'champion',
            
            # Loyalty indicators
            'setia', 'loyal', 'andalkan', 'favorit', 'pilihan', 'unggulan',
            'the best', 'terdepan', 'nomor satu', 'juara', 'winner',
            
            # Ease of use
            'gampang', 'simple', 'user friendly', 'intuitif', 'mudah dipahami',
            'tidak ribet', 'straightforward', 'accessible', 'convenient',
            
            # Slang variations
            'bgus', 'bgs', 'mantep', 'mntp', 'kerennn', 'cooool', 'nicee',
            'okee', 'oks', 'goks', 'mantappu', 'kereenn', 'baguss'
        }
        
        self.negative_keywords = {
            # Basic negative
            'buruk', 'jelek', 'tidak suka', 'kecewa', 'mengecewakan', 'payah',
            'bad', 'terrible', 'awful', 'worst', 'hate', 'boring', 'membosankan',
            'lambat', 'lelet', 'rusak', 'error', 'bug', 'masalah', 'problem',
            'susah', 'sulit', 'ribet', 'complicated', 'menyebalkan',
            
            # Performance issues
            'lama', 'telat', 'terlambat', 'delay', 'pending', 'hang', 'freeze',
            'crash', 'eror', 'gangguan', 'trouble', 'issue', 'glitch',
            'loading', 'stuck', 'macet', 'lemot', 'slow', 'not working',
            'tidak jalan', 'tidak berfungsi', 'tidak bisa', 'gabisa', 'ga bisa',
            
            # Service dissatisfaction
            'cancel', 'batal', 'dibatalkan', 'reject', 'ditolak', 'gagal',
            'failed', 'unsuccessful', 'tidak berhasil', 'gak bisa', 'cannot',
            'unable', 'impossible', 'mustahil', 'sia sia', 'percuma',
            
            # Technical problems
            'keluar sendiri', 'force close', 'restart', 'uninstall', 'reinstall',
            'update gagal', 'koneksi terputus', 'no signal', 'offline',
            'maintenance', 'down', 'server error', 'timeout', 'expired',
            
            # Price/value issues
            'mahal', 'expensive', 'overpriced', 'kemahalan', 'boros', 'rugi',
            'loss', 'kerugian', 'tidak worth it', 'not worth', 'sia-sia',
            'buang-buang', 'wasteful', 'money wasting', 'tidak sebanding',
            
            # Usability issues
            'rumit', 'complex', 'confusing', 'membingungkan', 'tidak jelas',
            'unclear', 'ambiguous', 'tidak mudah', 'sukar', 'challenging',
            'difficult', 'hard to use', 'user unfriendly', 'ribet banget',
            
            # Service quality
            'tidak ramah', 'kasar', 'rude', 'impolite', 'tidak sopan',
            'arrogant', 'sombong', 'jutek', 'galak', 'unfriendly',
            'tidak profesional', 'unprofessional', 'asal-asalan', 'sembarangan',
            
            # Quality issues
            'kualitas rendah', 'poor quality', 'tidak berkualitas', 'inferior',
            'tidak memuaskan', 'unsatisfactory', 'disappointing', 'frustrating',
            'annoying', 'menjengkelkan', 'mengganggu', 'bikin kesel',
            
            # Availability issues
            'tidak tersedia', 'unavailable', 'kosong', 'habis', 'out of stock',
            'tidak ada', 'missing', 'hilang', 'lost', 'tidak ditemukan',
            'not found', 'empty', 'void', 'null', 'tdk ada',
            
            # Stability issues
            'tidak stabil', 'unstable', 'inconsistent', 'tidak konsisten',
            'berubah-ubah', 'fluktuatif', 'naik turun', 'unpredictable',
            
            # Security concerns
            'tidak aman', 'unsafe', 'berbahaya', 'dangerous', 'risky',
            'berisiko', 'mencurigakan', 'suspicious', 'questionable',
            
            # Trust issues
            'tidak percaya', 'distrust', 'curiga', 'doubt', 'ragu',
            'skeptical', 'tidak yakin', 'unsure', 'uncertain', 'meragukan',
            
            # Slang variations
            'jlek', 'jelek banget', 'payahhh', 'burruk', 'ga jelas',
            'gajelas', 'gabener', 'parah', 'kacau', 'berantakan',
            'amburadul', 'ngaco', 'tolol', 'bodoh', 'stupid', 'idiot'
        }
    
    def improved_auto_label_sentiment(self, content, score=None):
        """Improved sentiment labeling with better logic"""
        if pd.isna(content) or content is None:
            return 'neutral'
        
        content_str = str(content).lower()
        
        # 1. Score-based labeling (if score is available and valid)
        if score is not None and not pd.isna(score):
            try:
                score_val = float(score)
                if score_val >= 4:
                    return 'positive'
                elif score_val <= 2:
                    return 'negative'
                elif score_val == 3:
                    # For score 3, use content analysis as tiebreaker
                    pass
            except (ValueError, TypeError):
                pass
        
        # 2. Content-based analysis
        # Count positive and negative keywords
        pos_count = sum(1 for word in self.positive_keywords if word in content_str)
        neg_count = sum(1 for word in self.negative_keywords if word in content_str)
        
        # 3. Negation handling
        negation_words = ['tidak', 'bukan', 'jangan', 'gak', 'ga', 'nggak', 'enggak']
        has_negation = any(neg in content_str for neg in negation_words)
        
        # 4. Length-based adjustment (very short texts tend to be less reliable)
        text_length = len(content_str.split())
        confidence_multiplier = min(1.0, text_length / 5.0)  # Full confidence at 5+ words
        
        # 5. Decision logic with improved scoring
        if has_negation:
            # If negation present, be more conservative
            if neg_count > pos_count + 1:
                return 'negative'
            elif pos_count > neg_count + 1:
                return 'positive'
            else:
                return 'neutral'
        else:
            # Standard scoring
            if pos_count > neg_count:
                if pos_count >= 2 or (pos_count == 1 and neg_count == 0):
                    return 'positive'
                else:
                    return 'neutral'
            elif neg_count > pos_count:
                if neg_count >= 2 or (neg_count == 1 and pos_count == 0):
                    return 'negative'
                else:
                    return 'neutral'
            else:
                # Equal counts or no keywords found
                if score is not None:
                    try:
                        score_val = float(score)
                        if score_val >= 3.5:
                            return 'positive'
                        elif score_val <= 2.5:
                            return 'negative'
                    except:
                        pass
                return 'neutral'
    
    def clean_text_vectorized(self, texts):
        """Vectorized text cleaning for better performance"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Convert to pandas Series for vectorized operations
        text_series = pd.Series(texts)
        
        # Handle None/NaN values
        text_series = text_series.fillna('')
        text_series = text_series.astype(str)
        
        # Vectorized cleaning operations
        # Remove URLs
        text_series = text_series.str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
        
        # Remove email addresses
        text_series = text_series.str.replace(r'\S+@\S+', '', regex=True)
        
        # Remove mentions and hashtags
        text_series = text_series.str.replace(r'@\w+|#\w+', '', regex=True)
        
        # Remove emojis and special characters
        text_series = text_series.str.replace(r'[^\w\s]', ' ', regex=True)
        
        # Remove extra whitespaces
        text_series = text_series.str.replace(r'\s+', ' ', regex=True)
        text_series = text_series.str.strip()
        
        # Remove digits
        text_series = text_series.str.replace(r'\d+', '', regex=True)
        
        # Convert to lowercase
        text_series = text_series.str.lower()
        
        # Remove empty strings
        text_series = text_series.replace('', np.nan)
        text_series = text_series.fillna('')
        
        return text_series.tolist()
    
    def normalize_slang_vectorized(self, texts):
        """Vectorized slang normalization"""
        if isinstance(texts, str):
            texts = [texts]
        
        normalized_texts = []
        for text in texts:
            if not text or pd.isna(text):
                normalized_texts.append('')
                continue
                
            words = str(text).split()
            normalized_words = [self.slang_dict.get(word, word) for word in words]
            normalized_texts.append(' '.join(normalized_words))
        
        return normalized_texts
    
    def preprocess_text_batch(self, texts, batch_size=1000):
        """Batch preprocessing for better performance"""
        if isinstance(texts, str):
            return self.preprocess_single_text(texts)
        
        processed_texts = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Clean texts
            cleaned_batch = self.clean_text_vectorized(batch)
            
            # Normalize slang
            normalized_batch = self.normalize_slang_vectorized(cleaned_batch)
            
            # Process each text in batch
            batch_processed = []
            for text in normalized_batch:
                if not text or pd.isna(text):
                    batch_processed.append('')
                    continue
                
                # Remove stopwords
                text_no_stopwords = self.stopword_remover.remove(str(text))
                
                # Stemming
                stemmed_text = self.stemmer.stem(text_no_stopwords)
                
                # Tokenization
                try:
                    tokens = word_tokenize(stemmed_text)
                except:
                    tokens = stemmed_text.split()
                
                # Filter tokens
                filtered_tokens = [token for token in tokens if len(token) > 1 and token.strip()]
                
                result = ' '.join(filtered_tokens)
                batch_processed.append(result if result.strip() else '')
            
            processed_texts.extend(batch_processed)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed batch {i // batch_size + 1}/{total_batches}")
        
        return processed_texts
    
    def preprocess_single_text(self, text):
        """Preprocess single text (for real-time prediction)"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # Clean text
        cleaned = self.clean_text_vectorized([text])[0]
        
        # Normalize slang
        normalized = self.normalize_slang_vectorized([cleaned])[0]
        
        # Remove stopwords
        no_stopwords = self.stopword_remover.remove(normalized)
        
        # Stemming
        stemmed = self.stemmer.stem(no_stopwords)
        
        # Tokenization
        try:
            tokens = word_tokenize(stemmed)
        except:
            tokens = stemmed.split()
        
        # Filter tokens
        filtered_tokens = [token for token in tokens if len(token) > 1 and token.strip()]
        
        result = ' '.join(filtered_tokens)
        return result if result.strip() else ""
    
    def load_and_preprocess_data(self, file_path, sample_size=None):
        """Optimized data loading and preprocessing"""
        print("Loading dataset...")
        
        # Read CSV with optimized settings
        df = pd.read_csv(file_path, low_memory=False)
        
        # Sample data if specified
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {sample_size} rows from dataset")
        
        print(f"Dataset shape: {df.shape}")
        
        # Data validation and cleaning
        print("Validating and cleaning data...")
        
        # Ensure content column exists
        if 'content' not in df.columns:
            raise ValueError("Dataset must have 'content' column")
        
        # Handle missing values efficiently
        initial_count = len(df)
        df = df.dropna(subset=['content']).reset_index(drop=True)
        df['content'] = df['content'].astype(str)
        
        # Remove empty content
        df = df[df['content'].str.strip() != ''].reset_index(drop=True)
        final_count = len(df)
        
        if final_count < initial_count:
            print(f"Removed {initial_count - final_count} rows with missing/empty content")
        
        # Improved auto-labeling
        print("Auto-labeling sentiment with improved algorithm...")
        if 'score' in df.columns:
            df['sentiment'] = df.apply(
                lambda row: self.improved_auto_label_sentiment(row['content'], row.get('score')), 
                axis=1
            )
        else:
            df['sentiment'] = df['content'].apply(
                lambda x: self.improved_auto_label_sentiment(x)
            )
        
        # Preprocess text with batch processing
        print("Preprocessing text with optimized batch processing...")
        df['processed_content'] = self.preprocess_text_batch(df['content'].tolist())
        
        # Convert to string and remove empty processed content
        df['processed_content'] = df['processed_content'].astype(str)
        df = df[df['processed_content'].str.strip() != ''].reset_index(drop=True)
        
        # Final validation
        if len(df) == 0:
            raise ValueError("No valid data remaining after preprocessing")
        
        print(f"Final dataset shape after preprocessing: {df.shape}")
        
        # Check and balance dataset
        sentiment_counts = df['sentiment'].value_counts()
        print("Sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        # Balance check and warning
        min_class_size = sentiment_counts.min()
        max_class_size = sentiment_counts.max()
        imbalance_ratio = max_class_size / min_class_size
        
        if imbalance_ratio > 3:
            print(f"⚠️  Warning: Dataset is imbalanced (ratio: {imbalance_ratio:.1f})")
            print("Consider using class weights or resampling techniques")
        
        return df
    
    def create_features_optimized(self, df):
        """Optimized feature creation with better data validation"""
        print("Creating optimized features...")
        
        # Enhanced data validation and cleaning
        if 'processed_content' not in df.columns:
            raise ValueError("Dataset must have 'processed_content' column")
        
        # Handle NaN values more thoroughly
        df_clean = df.copy()
        
        # Remove rows with NaN processed_content
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['processed_content']).reset_index(drop=True)
        
        # Convert to string and filter empty strings
        df_clean['processed_content'] = df_clean['processed_content'].astype(str)
        df_clean = df_clean[df_clean['processed_content'].str.strip() != ''].reset_index(drop=True)
        
        # Additional cleaning: remove entries that are just whitespace or very short
        df_clean = df_clean[df_clean['processed_content'].str.len() > 2].reset_index(drop=True)
        
        final_count = len(df_clean)
        
        if final_count != initial_count:
            print(f"Filtered {initial_count - final_count} invalid entries")
        
        if final_count == 0:
            raise ValueError("No valid texts remaining after cleaning")
        
        # Get clean text list
        texts = df_clean['processed_content'].tolist()
        
        # Double-check for any remaining NaN or invalid entries
        valid_texts = []
        for text in texts:
            if text and pd.notna(text) and str(text).strip():
                valid_texts.append(str(text).strip())
            else:
                valid_texts.append("")  # Replace invalid with empty string
        
        # Remove empty strings
        texts_final = [text for text in valid_texts if text]
        
        if not texts_final:
            raise ValueError("No valid texts after final cleaning")
        
        print(f"Processing {len(texts_final)} valid texts for feature extraction")
        
        # TF-IDF Features with optimized parameters
        print("Creating TF-IDF features...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 2),
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        
        try:
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts_final)
            print(f"TF-IDF features shape: {tfidf_features.shape}")
        except Exception as e:
            print(f"Error creating TF-IDF features: {e}")
            raise
        
        # Bag of Words Features
        print("Creating Bag of Words features...")
        self.count_vectorizer = CountVectorizer(
            max_features=5000, 
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        try:
            bow_features = self.count_vectorizer.fit_transform(texts_final)
            print(f"Bag of Words features shape: {bow_features.shape}")
        except Exception as e:
            print(f"Error creating Bag of Words features: {e}")
            raise
        
        # Word embeddings with optimized parameters
        print("Training Word2Vec model...")
        tokenized_texts = [text.split() for text in texts_final if text.strip()]
        
        # Filter out very short texts for word embeddings
        tokenized_texts = [tokens for tokens in tokenized_texts if len(tokens) >= 3]
        
        if tokenized_texts:
            try:
                self.word2vec_model = Word2Vec(
                    sentences=tokenized_texts,
                    vector_size=100,
                    window=5,
                    min_count=2,  # Ignore words that appear less than 2 times
                    workers=min(4, cpu_count()),
                    epochs=10
                )
                print("Word2Vec model trained successfully")
            except Exception as e:
                print(f"Warning: Word2Vec training failed: {e}")
                self.word2vec_model = None
            
            print("Training FastText model...")
            try:
                self.fasttext_model = FastText(
                    sentences=tokenized_texts,
                    vector_size=100,
                    window=5,
                    min_count=2,
                    workers=min(4, cpu_count()),
                    epochs=10
                )
                print("FastText model trained successfully")
            except Exception as e:
                print(f"Warning: FastText training failed: {e}")
                self.fasttext_model = None
        else:
            print("Warning: Not enough valid texts for word embeddings")
            self.word2vec_model = None
            self.fasttext_model = None
        
        return {
            'tfidf': tfidf_features,
            'bow': bow_features,
            'texts': texts_final,
            'tokenized': tokenized_texts,
            'clean_df': df_clean  # Return cleaned dataframe
        }
    
    def train_naive_bayes_balanced(self, X_train, y_train, feature_type='tfidf'):
        """Train Naive Bayes with class balancing"""
        print(f"Training balanced Naive Bayes with {feature_type} features...")
        
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # Create sample weights
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        
        # Train with balanced approach
        self.nb_model = MultinomialNB(alpha=0.1)  # Reduced alpha for less smoothing
        self.nb_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Save model and weights
        model_path = f'models/naive_bayes_{feature_type}_balanced.pkl'
        joblib.dump(self.nb_model, model_path)
        joblib.dump(class_weight_dict, f'models/class_weights_{feature_type}.pkl')
        
        print(f"Balanced Naive Bayes model saved to {model_path}")
        return self.nb_model
    
    def cross_validate_model(self, model, X, y, cv=5):
        """Perform cross-validation on the model"""
        print(f"Performing {cv}-fold cross-validation...")
        
        # Use StratifiedKFold to maintain class distribution
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def get_word_importance(self, text, model_type='nb', top_n=10):
        """Get word importance for prediction"""
        if model_type == 'nb' and self.nb_model and self.tfidf_vectorizer:
            try:
                # Preprocess text
                processed_text = self.preprocess_single_text(text)
                if not processed_text.strip():
                    return []
                
                # Transform text to TF-IDF
                tfidf_vector = self.tfidf_vectorizer.transform([processed_text])
                
                # Get feature names
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                
                # Get TF-IDF scores
                tfidf_scores = tfidf_vector.toarray()[0]
                
                # Create word-score pairs
                word_scores = [(feature_names[i], tfidf_scores[i]) 
                              for i in range(len(feature_names)) 
                              if tfidf_scores[i] > 0]
                
                # Sort by score and return top N
                word_scores.sort(key=lambda x: x[1], reverse=True)
                return word_scores[:top_n]
                
            except Exception as e:
                print(f"Error getting word importance: {e}")
                return []
        
        return []
    
    def prepare_lstm_data_balanced(self, texts, labels):
        """Prepare balanced LSTM data"""
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(texts) if text and str(text).strip()]
        
        if not valid_indices:
            raise ValueError("No valid texts for LSTM training")
        
        filtered_texts = [str(texts[i]).strip() for i in valid_indices]
        filtered_labels = [labels[i] for i in valid_indices]
        
        print(f"Preparing LSTM data with {len(filtered_texts)} valid texts")
        
        # Tokenize texts
        self.tokenizer = Tokenizer(
            num_words=5000, 
            oov_token='<OOV>',
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        self.tokenizer.fit_on_texts(filtered_texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(filtered_texts)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(filtered_labels)
        y = to_categorical(y_encoded, num_classes=len(np.unique(filtered_labels)))
        
        # Calculate class weights
        classes = np.unique(y_encoded)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
        self.class_weights = dict(zip(classes, class_weights))
        
        print(f"LSTM Class weights: {self.class_weights}")
        print(f"LSTM data shape: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def train_lstm_optimized(self, X_train, y_train, X_val, y_val):
        """Train optimized LSTM model"""
        print("Training optimized LSTM model...")
        
        # Build improved LSTM model
        self.lstm_model = Sequential([
            Embedding(
                input_dim=5000, 
                output_dim=128, 
                input_length=self.max_length,
                mask_zero=True
            ),
            SpatialDropout1D(0.3),
            LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
            LSTM(32, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(y_train.shape[1], activation='softmax')
        ])
        
        # Compile with optimized parameters
        self.lstm_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model with class weights
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        self.lstm_model.save('models/lstm_model_optimized.h5')
        joblib.dump(self.tokenizer, 'models/tokenizer_optimized.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder_optimized.pkl')
        joblib.dump(self.class_weights, 'models/lstm_class_weights.pkl')
        
        print("Optimized LSTM model saved")
        return history
    
    def evaluate_model_detailed(self, model, X_test, y_test, model_name, is_lstm=False):
        """Detailed model evaluation"""
        print(f"\nEvaluating {model_name}...")
        
        if is_lstm:
            y_pred_prob = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test, axis=1)
        else:
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)
            y_true = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        print(f"Overall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision (weighted): {precision:.4f}")
        print(f"  Recall (weighted): {recall:.4f}")
        print(f"  F1-Score (weighted): {f1:.4f}")
        
        # Class-specific metrics
        if is_lstm:
            class_names = self.label_encoder.classes_
        else:
            class_names = np.unique(y_true)
        
        print(f"\nPer-class Metrics:")
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                print(f"  {class_name}:")
                print(f"    Precision: {precision_per_class[i]:.4f}")
                print(f"    Recall: {recall_per_class[i]:.4f}")
                print(f"    F1-Score: {f1_per_class[i]:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification Report
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        print(f"\nClassification Report:\n{report}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_prob
        }
    
    def predict_sentiment_improved(self, text, model_type='nb'):
        """Improved sentiment prediction with confidence calibration"""
        # Preprocess text
        processed_text = self.preprocess_single_text(text)
        
        if not processed_text.strip():
            return {
                'prediction': 'neutral',
                'confidence': 0.33,
                'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            }
        
        if model_type == 'nb' and self.nb_model and self.tfidf_vectorizer:
            # Use Naive Bayes
            tfidf_features = self.tfidf_vectorizer.transform([processed_text])
            prediction = self.nb_model.predict(tfidf_features)[0]
            probabilities = self.nb_model.predict_proba(tfidf_features)[0]
            
            # Apply confidence calibration
            max_prob = max(probabilities)
            
            # Adjust confidence based on text length and content
            word_count = len(processed_text.split())
            if word_count < 3:
                max_prob *= 0.8  # Reduce confidence for very short texts
            
            class_names = self.nb_model.classes_
            prob_dict = dict(zip(class_names, probabilities))
            
            return {
                'prediction': prediction,
                'confidence': max_prob,
                'probabilities': prob_dict
            }
        
        elif model_type == 'lstm' and self.lstm_model and self.tokenizer:
            # Use LSTM
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post')
            
            prediction_prob = self.lstm_model.predict(padded_sequence, verbose=0)[0]
            prediction_idx = np.argmax(prediction_prob)
            prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
            
            # Confidence calibration for LSTM
            max_prob = max(prediction_prob)
            
            # Adjust confidence based on sequence length
            non_zero_count = np.count_nonzero(padded_sequence[0])
            if non_zero_count < 3:
                max_prob *= 0.8
            
            prob_dict = dict(zip(self.label_encoder.classes_, prediction_prob))
            
            return {
                'prediction': prediction,
                'confidence': max_prob,
                'probabilities': prob_dict
            }
        
        else:
            # Fallback to rule-based prediction
            fallback_prediction = self.improved_auto_label_sentiment(text)
            return {
                'prediction': fallback_prediction,
                'confidence': 0.5,
                'probabilities': {
                    'positive': 0.33 if fallback_prediction != 'positive' else 0.6,
                    'negative': 0.33 if fallback_prediction != 'negative' else 0.6,
                    'neutral': 0.34 if fallback_prediction != 'neutral' else 0.6
                }
            }
    
    def load_models_optimized(self):
        """Load optimized models"""
        try:
            # Load Naive Bayes
            if os.path.exists('models/naive_bayes_tfidf_balanced.pkl'):
                self.nb_model = joblib.load('models/naive_bayes_tfidf_balanced.pkl')
                print("Balanced Naive Bayes model loaded")
            elif os.path.exists('models/naive_bayes_tfidf.pkl'):
                self.nb_model = joblib.load('models/naive_bayes_tfidf.pkl')
                print("Standard Naive Bayes model loaded")
            
            # Load TF-IDF vectorizer
            if os.path.exists('models/tfidf_vectorizer.pkl'):
                self.tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
                print("TF-IDF vectorizer loaded")
            
            # Load LSTM model
            if os.path.exists('models/lstm_model_optimized.h5'):
                self.lstm_model = tf.keras.models.load_model('models/lstm_model_optimized.h5')
                self.tokenizer = joblib.load('models/tokenizer_optimized.pkl')
                self.label_encoder = joblib.load('models/label_encoder_optimized.pkl')
                if os.path.exists('models/lstm_class_weights.pkl'):
                    self.class_weights = joblib.load('models/lstm_class_weights.pkl')
                print("Optimized LSTM model loaded")
            elif os.path.exists('models/lstm_model.h5'):
                self.lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
                self.tokenizer = joblib.load('models/tokenizer.pkl')
                self.label_encoder = joblib.load('models/label_encoder.pkl')
                print("Standard LSTM model loaded")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def save_vectorizers_optimized(self):
        """Save optimized vectorizers and models"""
        try:
            if self.tfidf_vectorizer:
                joblib.dump(self.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
                print("TF-IDF vectorizer saved")
            if self.count_vectorizer:
                joblib.dump(self.count_vectorizer, 'models/count_vectorizer.pkl')
                print("Count vectorizer saved")
            if self.word2vec_model:
                self.word2vec_model.save('models/word2vec_model.bin')
                print("Word2Vec model saved")
            if self.fasttext_model:
                self.fasttext_model.save('models/fasttext_model.bin')
                print("FastText model saved")
            print("All vectorizers and models saved successfully")
        except Exception as e:
            print(f"Error saving vectorizers: {e}")
    
    def save_processed_data(self, df, filename):
        """Save processed data with metadata"""
        filepath = os.path.join('dataset_sudah', filename)
        df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'rows': len(df),
            'columns': list(df.columns),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'processed_date': datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.csv', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Processed data saved to {filepath}")
        print(f"Metadata saved to {metadata_path}")
        return filepath

# Compatibility wrapper for existing code
class SentimentAnalysisSystem(OptimizedSentimentAnalysisSystem):
    """Backward compatible wrapper"""
    
    def __init__(self):
        super().__init__()
    
    def auto_label_sentiment(self, text):
        """Backward compatible method"""
        return self.improved_auto_label_sentiment(text)
    
    def clean_text(self, text):
        """Backward compatible method"""
        return self.clean_text_vectorized([text])[0] if text else ""
    
    def normalize_slang(self, text):
        """Backward compatible method"""
        return self.normalize_slang_vectorized([text])[0] if text else ""
    
    def preprocess_text(self, text):
        """Backward compatible method"""
        return self.preprocess_single_text(text)
    
    def create_features(self, df):
        """Backward compatible method"""
        return self.create_features_optimized(df)
    
    def train_naive_bayes(self, X_train, y_train, feature_type='tfidf'):
        """Backward compatible method"""
        return self.train_naive_bayes_balanced(X_train, y_train, feature_type)
    
    def prepare_lstm_data(self, texts, labels):
        """Backward compatible method"""
        return self.prepare_lstm_data_balanced(texts, labels)
    
    def train_lstm(self, X_train, y_train, X_val, y_val):
        """Backward compatible method"""
        return self.train_lstm_optimized(X_train, y_train, X_val, y_val)
    
    def evaluate_model(self, model, X_test, y_test, model_name, is_lstm=False):
        """Backward compatible method"""
        return self.evaluate_model_detailed(model, X_test, y_test, model_name, is_lstm)
    
    def predict_sentiment(self, text, model_type='nb'):
        """Backward compatible method"""
        return self.predict_sentiment_improved(text, model_type)
    
    def load_models(self):
        """Backward compatible method"""
        return self.load_models_optimized()
    
    def save_vectorizers(self):
        """Backward compatible method"""
        return self.save_vectorizers_optimized()

# Example usage
if __name__ == "__main__":
    # Initialize optimized system
    sentiment_system = OptimizedSentimentAnalysisSystem()
    
    # Load and preprocess data
    df = sentiment_system.load_and_preprocess_data('Dataset.csv', sample_size=1000)
    
    # Save processed data
    sentiment_system.save_processed_data(df, 'processed_dataset_optimized.csv')
    
    # Create features
    features = sentiment_system.create_features_optimized(df)
    
    # Prepare data for training
    X_tfidf = features['tfidf']
    y = features['clean_df']['sentiment']  # Use cleaned dataframe
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train balanced Naive Bayes
    nb_model = sentiment_system.train_naive_bayes_balanced(X_train, y_train)
    
    # Evaluate with detailed metrics
    nb_results = sentiment_system.evaluate_model_detailed(nb_model, X_test, y_test, "Balanced Naive Bayes")
    
    # Cross-validate
    cv_scores = sentiment_system.cross_validate_model(nb_model, X_tfidf, y)
    
    # Prepare and train LSTM
    clean_df = features['clean_df']
    X_lstm, y_lstm = sentiment_system.prepare_lstm_data_balanced(clean_df['processed_content'], clean_df['sentiment'])
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
        X_lstm, y_lstm, test_size=0.2, random_state=42
    )
    
    X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
        X_train_lstm, y_train_lstm, test_size=0.2, random_state=42
    )
    
    # Train optimized LSTM
    lstm_history = sentiment_system.train_lstm_optimized(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
    
    # Evaluate LSTM
    lstm_results = sentiment_system.evaluate_model_detailed(
        sentiment_system.lstm_model, X_test_lstm, y_test_lstm, "Optimized LSTM", is_lstm=True
    )
    
    # Save all models and vectorizers
    sentiment_system.save_vectorizers_optimized()
    
    print("\n✅ Optimized training completed successfully!")
    print("📁 Models and vectorizers saved to 'models' directory")
    print("📊 Processed data saved to 'dataset_sudah' directory")
    print("\n🎯 Key improvements:")
    print("  - Faster preprocessing with vectorized operations")
    print("  - Balanced training with class weights")
    print("  - Improved sentiment labeling algorithm")
    print("  - Enhanced model architectures")
    print("  - Better confidence calibration")