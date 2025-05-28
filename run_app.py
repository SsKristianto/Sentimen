"""
Startup script untuk menjalankan aplikasi Sentiment Analysis
Script ini akan melakukan setup otomatis sebelum menjalankan aplikasi
"""

import os
import sys
import subprocess
import importlib.util

def check_and_install_package(package_name):
    """Check if package is installed, if not install it"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def setup_nltk_data():
    """Setup NLTK data with comprehensive error handling"""
    print("🔄 Setting up NLTK data...")
    
    try:
        import nltk
        import ssl
        
        # Handle SSL issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        required_data = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for data_name in required_data:
            try:
                nltk.download(data_name, quiet=True)
            except Exception as e:
                print(f"⚠️  Warning: Could not download {data_name}: {e}")
        
        print("✅ NLTK setup completed")
        return True
        
    except Exception as e:
        print(f"❌ NLTK setup failed: {e}")
        return False

def check_dataset():
    """Check if Dataset.csv exists"""
    if os.path.exists('Dataset.csv'):
        print("✅ Dataset.csv found")
        return True
    else:
        print("⚠️  Dataset.csv not found")
        print("   Please make sure Dataset.csv is in the same directory")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'dataset_sudah', 'docs']
    
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")

def run_streamlit_app():
    """Run the Streamlit application"""
    print("\n🚀 Starting Streamlit application...")
    print("=" * 50)
    
    try:
        # Import streamlit to check if it's available
        import streamlit as st
        
        # Run the app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        if check_and_install_package("streamlit"):
            subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        else:
            print("❌ Failed to install Streamlit")
            return False
    
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        return False
    
    return True

def main():
    """Main setup and run function"""
    print("🎯 Sentiment Analysis System Startup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create directories
    create_directories()
    
    # Setup NLTK
    setup_nltk_data()
    
    # Check dataset
    dataset_exists = check_dataset()
    
    if not dataset_exists:
        print("\n" + "=" * 50)
        print("⚠️  IMPORTANT: Dataset.csv is required!")
        print("   Please add your Dataset.csv file with columns:")
        print("   - userName, content, score, at, appVersion")
        print("=" * 50)
        
        choice = input("\nContinue anyway? (y/n): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("👋 Exiting...")
            return False
    
    # Run the application
    print("\n🎉 Setup completed!")
    run_streamlit_app()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error and try again")