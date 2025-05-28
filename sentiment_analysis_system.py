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

# Deep Learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Word embeddings
from gensim.models import Word2Vec, FastText
import warnings
warnings.filterwarnings('ignore')

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

class SentimentAnalysisSystem:
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
        
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('dataset_sudah', exist_ok=True)
        
        # Indonesian slang words dictionary for normalization
        self.slang_dict = {
            'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
            'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
            'yg': 'yang', 'krn': 'karena', 'dgn': 'dengan', 'utk': 'untuk',
            'tdk': 'tidak', 'jd': 'jadi', 'jgn': 'jangan', 'tp': 'tetapi',
            'bgt': 'banget', 'bgt': 'banget', 'bener': 'benar', 'emg': 'memang',
            'gmn': 'bagaimana', 'gmna': 'bagaimana', 'knp': 'kenapa',
            'hrs': 'harus', 'hbs': 'habis', 'blm': 'belum', 'blom': 'belum',
            'lg': 'lagi', 'lgi': 'lagi', 'org': 'orang', 'orng': 'orang',
            'klo': 'kalau', 'kalo': 'kalau', 'kl': 'kalau',
            'sy': 'saya', 'gw': 'saya', 'gue': 'saya', 'aku': 'saya',
            'lu': 'kamu', 'lo': 'kamu', 'km': 'kamu',
            'dr': 'dari', 'ke': 'ke', 'di': 'di', 'pd': 'pada',
            'dgr': 'dengar', 'liat': 'lihat', 'tau': 'tahu', 'tau': 'tahu'
        }
    
    def auto_label_sentiment(self, text):
        """Automatically label sentiment based on keywords and patterns"""
        text_lower = text.lower()
        
        # Positive keywords (diperluas berdasarkan analisis dataset)
        positive_words = [
            # Kata dasar positif
            'bagus', 'baik', 'suka', 'senang', 'puas', 'mantap', 'oke', 'ok',
            'recommended', 'rekomendasi', 'top', 'keren', 'asik', 'asyik',
            'love', 'like', 'good', 'great', 'excellent', 'amazing', 'awesome',
            'hebat', 'luar biasa', 'sempurna', 'memuaskan', 'terbaik',
            
            # Kata positif spesifik untuk layanan transportasi/delivery
            'cepat', 'tepat waktu', 'responsif', 'ramah', 'sopan', 'profesional',
            'membantu', 'terbantu', 'bermanfaat', 'mudah', 'praktis', 'efisien',
            'nyaman', 'aman', 'amanah', 'terpercaya', 'reliable', 'lancar',
            'smooth', 'gercep', 'sigap', 'mantul', 'jos', 'gandos',
            
            # Kata apresiasi dan kepuasan
            'terima kasih', 'thanks', 'makasih', 'appreciate', 'grateful',
            'sukses', 'success', 'berhasil', 'perfect', 'nice', 'cool',
            'mantabs', 'mantaap', 'mantabb', 'kece', 'juara', 'champion',
            
            # Kata yang menunjukkan loyalitas
            'setia', 'loyal', 'andalkan', 'favorit', 'pilihan', 'unggulan',
            'the best', 'terdepan', 'nomor satu', 'juara', 'winner',
            
            # Kata yang menunjukkan kemudahan
            'gampang', 'simple', 'user friendly', 'intuitif', 'mudah dipahami',
            'tidak ribet', 'straightforward', 'accessible', 'convenient',
            
            # Variasi ejaan positif yang sering muncul
            'bgus', 'bgs', 'mantep', 'mntp', 'kerennn', 'cooool', 'nicee'
        ]

        # Negative keywords (diperluas berdasarkan analisis dataset)
        negative_words = [
            # Kata dasar negatif
            'buruk', 'jelek', 'tidak suka', 'kecewa', 'mengecewakan', 'payah',
            'bad', 'terrible', 'awful', 'worst', 'hate', 'boring', 'membosankan',
            'lambat', 'lelet', 'rusak', 'error', 'bug', 'masalah', 'problem',
            'susah', 'sulit', 'ribet', 'complicated',
            
            # Kata negatif spesifik untuk layanan transportasi/delivery
            'lama', 'telat', 'terlambat', 'delay', 'pending', 'hang', 'freeze',
            'crash', 'eror', 'gangguan', 'trouble', 'issue', 'glitch',
            'loading', 'stuck', 'macet', 'lemot', 'slow', 'not working',
            
            # Kata yang menunjukkan ketidakpuasan layanan
            'cancel', 'batal', 'dibatalkan', 'reject', 'ditolak', 'gagal',
            'failed', 'unsuccessful', 'tidak berhasil', 'tidak bisa', 'gabisa',
            'ga bisa', 'gak bisa', 'cannot', 'unable', 'impossible',
            
            # Kata yang menunjukkan masalah teknis
            'keluar sendiri', 'force close', 'restart', 'uninstall', 'reinstall',
            'update gagal', 'koneksi terputus', 'no signal', 'offline',
            'maintenance', 'down', 'server error', 'timeout', 'expired',
            
            # Kata yang menunjukkan ketidakpuasan finansial
            'mahal', 'expensive', 'overpriced', 'kemahalan', 'boros', 'rugi',
            'loss', 'kerugian', 'tidak worth it', 'not worth', 'sia-sia',
            'percuma', 'buang-buang', 'wasteful', 'money wasting',
            
            # Kata yang menunjukkan kesulitan penggunaan
            'rumit', 'complex', 'confusing', 'membingungkan', 'tidak jelas',
            'unclear', 'ambiguous', 'tidak mudah', 'sukar', 'challenging',
            'difficult', 'hard to use', 'user unfriendly',
            
            # Kata yang menunjukkan ketidakpuasan driver/service
            'tidak ramah', 'kasar', 'rude', 'impolite', 'tidak sopan',
            'arrogant', 'sombong', 'jutek', 'galak', 'unfriendly',
            'tidak profesional', 'unprofessional', 'asal-asalan', 'sembarangan',
            
            # Kata yang menunjukkan masalah kualitas
            'kualitas rendah', 'poor quality', 'tidak berkualitas', 'inferior',
            'tidak memuaskan', 'unsatisfactory', 'disappointing', 'frustrating',
            'annoying', 'menjengkelkan', 'menyebalkan', 'mengganggu',
            
            # Kata yang menunjukkan masalah availability
            'tidak tersedia', 'unavailable', 'kosong', 'habis', 'out of stock',
            'tidak ada', 'missing', 'hilang', 'lost', 'tidak ditemukan',
            'not found', 'empty', 'void', 'null',
            
            # Variasi ejaan negatif yang sering muncul
            'jlek', 'jelek banget', 'payahhh', 'burruk', 'ga jelas',
            'gajelas', 'gabener', 'parah', 'kacau', 'berantakan',
            'amburadul', 'ngaco', 'tolol', 'bodoh', 'stupid', 'idiot',
            
            # Kata yang menunjukkan ketidakstabilan
            'tidak stabil', 'unstable', 'inconsistent', 'tidak konsisten',
            'berubah-ubah', 'fluktuatif', 'naik turun', 'unpredictable',
            
            # Kata yang menunjukkan ketidakamanan
            'tidak aman', 'unsafe', 'berbahaya', 'dangerous', 'risky',
            'berisiko', 'mencurigakan', 'suspicious', 'questionable',
            
            # Kata yang menunjukkan ketidakpercayaan
            'tidak percaya', 'distrust', 'curiga', 'doubt', 'ragu',
            'skeptical', 'tidak yakin', 'unsure', 'uncertain'
        ]
        
        # Count positive and negative words
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Simple scoring based on rating if available
        if hasattr(self, 'score_column') and self.score_column:
            # Assuming score is from 1-5 scale
            score = float(text.split()[-1]) if text.split()[-1].isdigit() else 3
            if score >= 4:
                return 'positive'
            elif score <= 2:
                return 'negative'
            else:
                return 'neutral'
        
        # Rule-based classification
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    def clean_text(self, text):
        """Clean text by removing special characters, URLs, and emojis"""
        # Handle None or NaN values
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Return empty string if input is empty
        if not text.strip():
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove digits
        text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text if text.strip() else ""
    
    def normalize_slang(self, text):
        """Normalize Indonesian slang words"""
        words = text.split()
        normalized_words = []
        
        for word in words:
            if word in self.slang_dict:
                normalized_words.append(self.slang_dict[word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def preprocess_text(self, text):
        """Complete text preprocessing pipeline"""
        # Handle None or NaN values
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Clean text
        text = self.clean_text(text)
        
        # Normalize slang
        text = self.normalize_slang(text)
        
        # Remove stopwords
        text = self.stopword_remover.remove(text)
        
        # Stemming
        text = self.stemmer.stem(text)
        
        # Tokenization
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback if tokenization fails
            tokens = text.split()
        
        # Filter out single characters and empty tokens
        tokens = [token for token in tokens if len(token) > 1 and token.strip()]
        
        # Return processed text
        result = ' '.join(tokens)
        return result if result.strip() else ""
    
    def load_and_preprocess_data(self, file_path, sample_size=None):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        # Sample data if specified
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {sample_size} rows from dataset")
        
        print(f"Dataset shape: {df.shape}")
        
        # Data validation and cleaning
        print("Validating and cleaning data...")
        
        # Ensure content column exists and handle missing values
        if 'content' not in df.columns:
            raise ValueError("Dataset must have 'content' column")
        
        # Handle missing values in content
        initial_count = len(df)
        df = df.dropna(subset=['content']).reset_index(drop=True)
        df['content'] = df['content'].astype(str)
        
        # Remove empty content
        df = df[df['content'].str.strip() != ''].reset_index(drop=True)
        final_count = len(df)
        
        if final_count < initial_count:
            print(f"Removed {initial_count - final_count} rows with missing/empty content")
        
        # Auto-label sentiment based on content and score
        print("Auto-labeling sentiment...")
        df['sentiment'] = df.apply(lambda row: self.auto_label_sentiment(
            f"{row['content']} {row.get('score', 3)}"
        ), axis=1)
        
        # Preprocess text
        print("Preprocessing text...")
        df['processed_content'] = df['content'].apply(self.preprocess_text)
        
        # Remove empty processed content and convert to string
        df['processed_content'] = df['processed_content'].astype(str)
        df = df[df['processed_content'].str.strip() != ''].reset_index(drop=True)
        
        # Final validation
        if len(df) == 0:
            raise ValueError("No valid data remaining after preprocessing")
        
        print(f"Final dataset shape after preprocessing: {df.shape}")
        print("Sentiment distribution:")
        print(df['sentiment'].value_counts())
        
        return df
    
    def create_features(self, df):
        """Create feature representations"""
        texts = df['processed_content'].tolist()
        
        # TF-IDF Features
        print("Creating TF-IDF features...")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        
        # Bag of Words Features
        print("Creating Bag of Words features...")
        self.count_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
        bow_features = self.count_vectorizer.fit_transform(texts)
        
        # Word2Vec Features
        print("Training Word2Vec model...")
        tokenized_texts = [text.split() for text in texts]
        self.word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=100, 
                                     window=5, min_count=1, workers=4)
        
        # FastText Features
        print("Training FastText model...")
        self.fasttext_model = FastText(sentences=tokenized_texts, vector_size=100,
                                     window=5, min_count=1, workers=4)
        
        return {
            'tfidf': tfidf_features,
            'bow': bow_features,
            'texts': texts,
            'tokenized': tokenized_texts
        }
    
    def get_word_embeddings(self, texts, model_type='word2vec'):
        """Get word embeddings for texts"""
        embeddings = []
        model = self.word2vec_model if model_type == 'word2vec' else self.fasttext_model
        
        for text in texts:
            words = text.split()
            word_vectors = []
            
            for word in words:
                try:
                    word_vectors.append(model.wv[word])
                except KeyError:
                    # Use random vector for unknown words
                    word_vectors.append(np.random.normal(size=100))
            
            if word_vectors:
                # Average word vectors
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(100))
        
        return np.array(embeddings)
    
    def train_naive_bayes(self, X_train, y_train, feature_type='tfidf'):
        """Train Naive Bayes model"""
        print(f"Training Naive Bayes with {feature_type} features...")
        self.nb_model = MultinomialNB()
        self.nb_model.fit(X_train, y_train)
        
        # Save model
        model_path = f'models/naive_bayes_{feature_type}.pkl'
        joblib.dump(self.nb_model, model_path)
        print(f"Naive Bayes model saved to {model_path}")
        
        return self.nb_model
    
    def prepare_lstm_data(self, texts, labels):
        """Prepare data for LSTM model"""
        # Tokenize texts
        self.tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        y = to_categorical(y_encoded, num_classes=len(np.unique(labels)))
        
        return X, y
    
    def train_lstm(self, X_train, y_train, X_val, y_val):
        """Train LSTM model"""
        print("Training LSTM model...")
        
        # Build LSTM model
        self.lstm_model = Sequential([
            Embedding(input_dim=5000, output_dim=128, input_length=self.max_length),
            SpatialDropout1D(0.2),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(y_train.shape[1], activation='softmax')
        ])
        
        self.lstm_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            verbose=1
        )
        
        # Save model
        self.lstm_model.save('models/lstm_model.h5')
        joblib.dump(self.tokenizer, 'models/tokenizer.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        print("LSTM model saved to models/lstm_model.h5")
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, model_name, is_lstm=False):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        if is_lstm:
            y_pred_prob = model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test, axis=1)
        else:
            y_pred = model.predict(X_test)
            y_true = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification Report
        if is_lstm:
            class_names = self.label_encoder.classes_
        else:
            class_names = np.unique(y_true)
        
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(f"\nClassification Report:\n{report}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def cross_validate_model(self, model, X, y, cv=5):
        """Perform cross-validation"""
        print(f"Performing {cv}-fold cross-validation...")
        
        cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42))
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_processed_data(self, df, filename):
        """Save processed data"""
        filepath = os.path.join('dataset_sudah', filename)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
        return filepath
    
    def load_models(self):
        """Load trained models"""
        try:
            # Load Naive Bayes
            if os.path.exists('models/naive_bayes_tfidf.pkl'):
                self.nb_model = joblib.load('models/naive_bayes_tfidf.pkl')
                print("Naive Bayes model loaded")
            
            # Load TF-IDF vectorizer
            if os.path.exists('models/tfidf_vectorizer.pkl'):
                self.tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
                print("TF-IDF vectorizer loaded")
            
            # Load LSTM model
            if os.path.exists('models/lstm_model.h5'):
                self.lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
                self.tokenizer = joblib.load('models/tokenizer.pkl')
                self.label_encoder = joblib.load('models/label_encoder.pkl')
                print("LSTM model loaded")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def save_vectorizers(self):
        """Save feature vectorizers"""
        if self.tfidf_vectorizer:
            joblib.dump(self.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        if self.count_vectorizer:
            joblib.dump(self.count_vectorizer, 'models/count_vectorizer.pkl')
        if self.word2vec_model:
            self.word2vec_model.save('models/word2vec_model.bin')
        if self.fasttext_model:
            self.fasttext_model.save('models/fasttext_model.bin')
    
    def predict_sentiment(self, text, model_type='nb'):
        """Predict sentiment for new text"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if model_type == 'nb' and self.nb_model and self.tfidf_vectorizer:
            # Use Naive Bayes
            tfidf_features = self.tfidf_vectorizer.transform([processed_text])
            prediction = self.nb_model.predict(tfidf_features)[0]
            probabilities = self.nb_model.predict_proba(tfidf_features)[0]
            
            return {
                'prediction': prediction,
                'confidence': max(probabilities),
                'probabilities': dict(zip(self.nb_model.classes_, probabilities))
            }
        
        elif model_type == 'lstm' and self.lstm_model and self.tokenizer:
            # Use LSTM
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post')
            
            prediction_prob = self.lstm_model.predict(padded_sequence)[0]
            prediction_idx = np.argmax(prediction_prob)
            prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
            
            return {
                'prediction': prediction,
                'confidence': max(prediction_prob),
                'probabilities': dict(zip(self.label_encoder.classes_, prediction_prob))
            }
        
        else:
            return {
                'prediction': 'neutral',
                'confidence': 0.33,
                'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            }
    
    def get_word_importance(self, text, model_type='nb', top_n=10):
        """Get important words for prediction"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        if model_type == 'nb' and self.tfidf_vectorizer:
            # Get feature names
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            tfidf_scores = self.tfidf_vectorizer.transform([processed_text])
            
            # Get word importance
            word_scores = []
            for word in words:
                if word in feature_names:
                    idx = np.where(feature_names == word)[0]
                    if len(idx) > 0:
                        score = tfidf_scores[0, idx[0]]
                        word_scores.append((word, score))
            
            # Sort by importance
            word_scores.sort(key=lambda x: x[1], reverse=True)
            return word_scores[:top_n]
        
        return [(word, 0.5) for word in words[:top_n]]

# Example usage
if __name__ == "__main__":
    # Initialize system
    sentiment_system = SentimentAnalysisSystem()
    
    # Load and preprocess data
    df = sentiment_system.load_and_preprocess_data('Dataset.csv', sample_size=1000)
    
    # Save processed data
    sentiment_system.save_processed_data(df, 'processed_dataset.csv')
    
    # Create features
    features = sentiment_system.create_features(df)
    
    # Prepare data for training
    X_tfidf = features['tfidf']
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Naive Bayes
    nb_model = sentiment_system.train_naive_bayes(X_train, y_train)
    
    # Evaluate Naive Bayes
    nb_results = sentiment_system.evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    
    # Cross-validate Naive Bayes
    cv_scores = sentiment_system.cross_validate_model(nb_model, X_tfidf, y)
    
    # Prepare LSTM data
    X_lstm, y_lstm = sentiment_system.prepare_lstm_data(df['processed_content'], df['sentiment'])
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
        X_lstm, y_lstm, test_size=0.2, random_state=42
    )
    
    # Further split training data for validation
    X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
        X_train_lstm, y_train_lstm, test_size=0.2, random_state=42
    )
    
    # Train LSTM
    lstm_history = sentiment_system.train_lstm(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
    
    # Evaluate LSTM
    lstm_results = sentiment_system.evaluate_model(
        sentiment_system.lstm_model, X_test_lstm, y_test_lstm, "LSTM", is_lstm=True
    )
    
    # Save vectorizers
    sentiment_system.save_vectorizers()
    
    print("\nTraining completed successfully!")
    print("Models and vectorizers saved to 'models' directory")
    print("Processed data saved to 'dataset_sudah' directory")