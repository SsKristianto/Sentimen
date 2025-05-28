"""
Setup script untuk download semua NLTK data yang diperlukan
Jalankan sekali sebelum menggunakan aplikasi utama
"""

import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_all_nltk_data():
    """Download semua NLTK data yang diperlukan"""
    
    print("🔄 Downloading NLTK data...")
    
    # List of required NLTK data
    required_data = [
        ('punkt', 'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('omw-1.4', 'corpora/omw-1.4')
    ]
    
    success_count = 0
    total_count = len(required_data)
    
    for data_name, data_path in required_data:
        try:
            # Check if already exists
            try:
                nltk.data.find(data_path)
                print(f"✅ {data_name} - Already exists")
                success_count += 1
                continue
            except LookupError:
                pass
            
            # Download if not exists
            print(f"📥 Downloading {data_name}...")
            nltk.download(data_name, quiet=False)
            print(f"✅ {data_name} - Downloaded successfully")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {data_name} - Failed to download: {e}")
    
    print(f"\n🎯 Download Summary: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        print("🎉 All NLTK data downloaded successfully!")
        print("✅ You can now run the main application")
    else:
        print("⚠️  Some downloads failed, but the application might still work")
    
    return success_count == total_count

def test_nltk_imports():
    """Test apakah semua NLTK imports berfungsi"""
    print("\n🧪 Testing NLTK imports...")
    
    try:
        from nltk.tokenize import word_tokenize
        print("✅ word_tokenize - OK")
    except Exception as e:
        print(f"❌ word_tokenize - Failed: {e}")
    
    try:
        from nltk.corpus import stopwords
        print("✅ stopwords - OK")
    except Exception as e:
        print(f"❌ stopwords - Failed: {e}")
    
    try:
        # Test tokenization
        test_text = "Ini adalah teks percobaan untuk testing."
        tokens = word_tokenize(test_text)
        print(f"✅ Tokenization test - OK ({len(tokens)} tokens)")
    except Exception as e:
        print(f"❌ Tokenization test - Failed: {e}")
    
    try:
        # Test stopwords
        stop_words = set(stopwords.words('english'))
        print(f"✅ Stopwords test - OK ({len(stop_words)} English stopwords)")
    except Exception as e:
        print(f"❌ Stopwords test - Failed: {e}")

if __name__ == "__main__":
    print("🚀 NLTK Setup Script")
    print("=" * 50)
    
    success = download_all_nltk_data()
    test_nltk_imports()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("Now you can run: streamlit run streamlit_app.py")
    else:
        print("⚠️  Setup completed with some issues")
        print("Try running the application anyway, it might still work")
    
    print("=" * 50)