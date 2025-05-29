"""
Optimized startup script untuk menjalankan aplikasi Sentiment Analysis
Enhanced dengan better error handling, performance monitoring, dan system validation
"""

import os
import sys
import subprocess
import importlib.util
import platform
import time
import psutil
from pathlib import Path
import json
from datetime import datetime

class OptimizedSystemManager:
    """Enhanced system manager with performance monitoring and validation"""
    
    def __init__(self):
        self.start_time = time.time()
        self.system_info = self._get_system_info()
        self.required_packages = {
            'core': [
                'streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 
                'plotly', 'wordcloud', 'scikit-learn', 'tensorflow'
            ],
            'nlp': [
                'nltk', 'sastrawi', 'gensim'
            ],
            'optional': [
                'psutil', 'joblib', 'pillow'
            ]
        }
        self.installation_log = []
    
    def _get_system_info(self):
        """Get comprehensive system information"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count() if 'psutil' in sys.modules else os.cpu_count(),
            'memory_total': self._get_memory_info(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_memory_info(self):
        """Get memory information safely"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'percent_used': memory.percent
            }
        except ImportError:
            return {'total_gb': 'Unknown', 'available_gb': 'Unknown', 'percent_used': 'Unknown'}
    
    def print_header(self):
        """Print enhanced header with system info"""
        print("=" * 70)
        print("🎯 OPTIMIZED SENTIMENT ANALYSIS SYSTEM STARTUP")
        print("=" * 70)
        print(f"🖥️  Platform: {self.system_info['platform']} ({self.system_info['architecture']})")
        print(f"🐍 Python: {self.system_info['python_version']}")
        print(f"💾 CPU Cores: {self.system_info['cpu_count']}")
        
        if isinstance(self.system_info['memory_total'], dict):
            memory_info = self.system_info['memory_total']
            print(f"🧠 Memory: {memory_info['total_gb']} GB total, {memory_info['available_gb']} GB available")
        
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def check_python_version(self):
        """Enhanced Python version check"""
        print("🔍 Checking Python version...")
        
        current_version = sys.version_info
        required_major, required_minor = 3, 7
        
        if current_version.major < required_major or (current_version.major == required_major and current_version.minor < required_minor):
            print(f"❌ Python {required_major}.{required_minor}+ required, found {current_version.major}.{current_version.minor}")
            print("📥 Please upgrade Python to continue")
            return False
        
        print(f"✅ Python {current_version.major}.{current_version.minor}.{current_version.micro} - Compatible")
        
        # Check for specific version recommendations
        if current_version.minor >= 10:
            print("🚀 Excellent! You're using a recent Python version")
        elif current_version.minor >= 8:
            print("✨ Good! Modern Python version detected")
        else:
            print("⚠️  Consider upgrading to Python 3.8+ for better performance")
        
        return True
    
    def check_and_install_package(self, package_name, category='core'):
        """Enhanced package installation with better error handling"""
        try:
            # Check if package is already installed
            importlib.import_module(package_name)
            print(f"✅ {package_name} - Already installed")
            self.installation_log.append({
                'package': package_name,
                'status': 'already_installed',
                'category': category,
                'timestamp': datetime.now().isoformat()
            })
            return True
            
        except ImportError:
            print(f"📦 Installing {package_name} ({category})...")
            
            # Special handling for some packages
            install_name = package_name
            if package_name == 'sastrawi':
                install_name = 'PySastrawi'
            elif package_name == 'pillow':
                install_name = 'Pillow'
            
            try:
                # Install with progress indication
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", install_name, "--upgrade"],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
                
                if result.returncode == 0:
                    print(f"✅ {package_name} - Installed successfully")
                    self.installation_log.append({
                        'package': package_name,
                        'status': 'installed',
                        'category': category,
                        'timestamp': datetime.now().isoformat()
                    })
                    return True
                else:
                    print(f"❌ {package_name} - Installation failed")
                    print(f"Error: {result.stderr}")
                    self.installation_log.append({
                        'package': package_name,
                        'status': 'failed',
                        'error': result.stderr,
                        'category': category,
                        'timestamp': datetime.now().isoformat()
                    })
                    return False
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ {package_name} - Installation timeout")
                return False
            except Exception as e:
                print(f"❌ {package_name} - Installation error: {e}")
                return False
    
    def install_packages(self):
        """Install all required packages with progress tracking"""
        print("\n🔧 PACKAGE INSTALLATION")
        print("-" * 50)
        
        total_packages = sum(len(packages) for packages in self.required_packages.values())
        installed_count = 0
        failed_packages = []
        
        for category, packages in self.required_packages.items():
            print(f"\n📋 Installing {category} packages...")
            
            for package in packages:
                if self.check_and_install_package(package, category):
                    installed_count += 1
                else:
                    failed_packages.append((package, category))
                
                # Progress indicator
                progress = (installed_count + len(failed_packages)) / total_packages * 100
                print(f"📊 Progress: {progress:.1f}% ({installed_count}/{total_packages} successful)")
        
        # Summary
        print(f"\n📊 INSTALLATION SUMMARY")
        print(f"✅ Successfully installed: {installed_count}/{total_packages}")
        
        if failed_packages:
            print(f"❌ Failed packages: {len(failed_packages)}")
            for package, category in failed_packages:
                print(f"   • {package} ({category})")
            
            # Check if critical packages failed
            critical_failed = [pkg for pkg, cat in failed_packages if cat == 'core']
            if critical_failed:
                print(f"⚠️  Critical packages failed: {', '.join(critical_failed)}")
                print("🔧 The application may not work properly")
                return False
            else:
                print("ℹ️  Only optional packages failed, application should work")
                return True
        else:
            print("🎉 All packages installed successfully!")
            return True
    
    def setup_nltk_data(self):
        """Enhanced NLTK setup with better error handling"""
        print("\n🔤 NLTK DATA SETUP")
        print("-" * 50)
        
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
            
            # Enhanced NLTK data list
            required_data = [
                ('punkt', 'Punkt tokenizer'),
                ('punkt_tab', 'Punkt tokenizer (tabular)'),
                ('stopwords', 'Stop words corpus'),
                ('wordnet', 'WordNet database'),
                ('averaged_perceptron_tagger', 'POS tagger'),
                ('omw-1.4', 'Open Multilingual Wordnet')
            ]
            
            successful_downloads = 0
            
            for data_name, description in required_data:
                try:
                    print(f"📥 Downloading {description}...")
                    
                    # Check if already exists
                    try:
                        nltk.data.find(f'tokenizers/{data_name}')
                        print(f"✅ {data_name} - Already exists")
                        successful_downloads += 1
                        continue
                    except LookupError:
                        pass
                    
                    # Download with timeout
                    download_success = nltk.download(data_name, quiet=False, raise_on_error=True)
                    
                    if download_success:
                        print(f"✅ {data_name} - Downloaded successfully")
                        successful_downloads += 1
                    else:
                        print(f"⚠️  {data_name} - Download completed with warnings")
                        successful_downloads += 1
                        
                except Exception as e:
                    print(f"❌ {data_name} - Failed: {e}")
            
            print(f"\n📊 NLTK DATA SUMMARY")
            print(f"✅ Successfully downloaded: {successful_downloads}/{len(required_data)}")
            
            if successful_downloads >= len(required_data) * 0.8:  # 80% success rate
                print("🎉 NLTK setup completed successfully!")
                return True
            else:
                print("⚠️  Some NLTK data failed, but application might still work")
                return True
                
        except ImportError:
            print("❌ NLTK not available - this will be handled during package installation")
            return False
        except Exception as e:
            print(f"❌ NLTK setup failed: {e}")
            return False
    
    def check_dataset(self):
        """Enhanced dataset validation"""
        print("\n📊 DATASET VALIDATION")
        print("-" * 50)
        
        dataset_path = Path('Dataset.csv')
        
        if not dataset_path.exists():
            print("❌ Dataset.csv not found!")
            print("📋 Expected file: Dataset.csv")
            print("📍 Current directory:", os.getcwd())
            
            # List CSV files in current directory
            csv_files = list(Path('.').glob('*.csv'))
            if csv_files:
                print("📁 Found CSV files:")
                for csv_file in csv_files:
                    size_mb = csv_file.stat().st_size / (1024 * 1024)
                    print(f"   • {csv_file.name} ({size_mb:.2f} MB)")
                print("💡 Rename one of these to 'Dataset.csv' if appropriate")
            
            return False
        
        try:
            # Quick dataset analysis
            import pandas as pd
            
            print("🔍 Analyzing dataset...")
            
            # Read just the header and a few rows for quick analysis
            df_sample = pd.read_csv(dataset_path, nrows=100)
            
            # Get file size
            file_size_mb = dataset_path.stat().st_size / (1024 * 1024)
            
            # Get total rows (approximately)
            with open(dataset_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f) - 1  # Subtract header
            
            print(f"✅ Dataset found: {dataset_path.name}")
            print(f"📏 File size: {file_size_mb:.2f} MB")
            print(f"📊 Estimated rows: {total_lines:,}")
            print(f"📋 Columns: {len(df_sample.columns)}")
            
            # Check required columns
            required_columns = ['content']
            recommended_columns = ['userName', 'score', 'at', 'appVersion']
            
            missing_required = [col for col in required_columns if col not in df_sample.columns]
            missing_recommended = [col for col in recommended_columns if col not in df_sample.columns]
            
            if missing_required:
                print(f"❌ Missing required columns: {missing_required}")
                print("📋 Available columns:", list(df_sample.columns))
                return False
            else:
                print("✅ All required columns present")
            
            if missing_recommended:
                print(f"⚠️  Missing recommended columns: {missing_recommended}")
                print("ℹ️  Application will work but some features may be limited")
            
            # Data quality check
            content_null_pct = df_sample['content'].isnull().mean() * 100
            if content_null_pct > 50:
                print(f"⚠️  High null rate in content column: {content_null_pct:.1f}%")
            else:
                print(f"✅ Content column quality: {100-content_null_pct:.1f}% valid")
            
            # Performance recommendation
            if file_size_mb > 100:
                print("💡 Large dataset detected - consider using sampling for faster processing")
            elif file_size_mb > 50:
                print("💡 Medium dataset - processing time may vary")
            else:
                print("✅ Dataset size is optimal for processing")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")
            return False
    
    def create_directories(self):
        """Enhanced directory creation with validation"""
        print("\n📁 DIRECTORY SETUP")
        print("-" * 50)
        
        directories = {
            'models': 'Store trained models',
            'dataset_sudah': 'Store processed datasets',
            'docs': 'Documentation and logs',
            'temp': 'Temporary files'
        }
        
        created_count = 0
        
        for dir_name, description in directories.items():
            dir_path = Path(dir_name)
            
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"✅ Created: {dir_name}/ - {description}")
                    created_count += 1
                except Exception as e:
                    print(f"❌ Failed to create {dir_name}/: {e}")
            else:
                print(f"📁 Exists: {dir_name}/ - {description}")
        
        # Check permissions
        test_file = Path('temp/test_write.txt')
        try:
            test_file.write_text('test')
            test_file.unlink()
            print("✅ Write permissions confirmed")
        except Exception as e:
            print(f"⚠️  Write permission issue: {e}")
        
        return True
    
    def save_system_info(self):
        """Save system information and installation log"""
        try:
            logs_dir = Path('docs')
            logs_dir.mkdir(exist_ok=True)
            
            # Save system info
            system_log = {
                'system_info': self.system_info,
                'installation_log': self.installation_log,
                'startup_time': time.time() - self.start_time,
                'status': 'completed'
            }
            
            log_file = logs_dir / f"startup_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(log_file, 'w') as f:
                json.dump(system_log, f, indent=2, default=str)
            
            print(f"📝 System log saved: {log_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save system log: {e}")
    
    def run_streamlit_app(self):
        """Enhanced Streamlit application runner"""
        print("\n🚀 STARTING APPLICATION")
        print("-" * 50)
        
        try:
            # Verify streamlit is available
            import streamlit
            print(f"✅ Streamlit {streamlit.__version__} ready")
            
            # Check if app file exists
            app_file = Path('streamlit_app.py')
            if not app_file.exists():
                print("❌ streamlit_app.py not found!")
                return False
            
            print("🌐 Starting web interface...")
            print("📱 The application will open in your default browser")
            print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
            print("⏹️  Press Ctrl+C to stop the application")
            print("-" * 50)
            
            # Run streamlit with optimized settings
            cmd = [
                sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                "--server.headless", "false",
                "--server.runOnSave", "true",
                "--server.maxUploadSize", "200"
            ]
            
            subprocess.run(cmd)
            return True
            
        except ImportError:
            print("❌ Streamlit not installed properly")
            return False
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
            return True
        except Exception as e:
            print(f"❌ Error running Streamlit: {e}")
            return False
    
    def run_complete_setup(self):
        """Run complete system setup and validation"""
        self.print_header()
        
        # Setup steps with validation
        setup_steps = [
            ("Python Version", self.check_python_version),
            ("Directories", self.create_directories),
            ("Packages", self.install_packages),
            ("NLTK Data", self.setup_nltk_data),
            ("Dataset", self.check_dataset)
        ]
        
        failed_steps = []
        
        for step_name, step_function in setup_steps:
            print(f"\n🔄 Step: {step_name}")
            try:
                if not step_function():
                    failed_steps.append(step_name)
                    print(f"⚠️  {step_name} completed with issues")
                else:
                    print(f"✅ {step_name} completed successfully")
            except Exception as e:
                failed_steps.append(step_name)
                print(f"❌ {step_name} failed: {e}")
        
        # Setup summary
        print("\n" + "=" * 70)
        print("📊 SETUP SUMMARY")
        print("=" * 70)
        
        total_time = time.time() - self.start_time
        print(f"⏱️  Total setup time: {total_time:.2f} seconds")
        print(f"✅ Successful steps: {len(setup_steps) - len(failed_steps)}/{len(setup_steps)}")
        
        if failed_steps:
            print(f"⚠️  Steps with issues: {', '.join(failed_steps)}")
            
            # Check if critical steps failed
            critical_steps = ["Python Version", "Packages"]
            critical_failed = [step for step in failed_steps if step in critical_steps]
            
            if critical_failed:
                print(f"❌ Critical failures: {', '.join(critical_failed)}")
                print("🔧 Please resolve these issues before continuing")
                
                choice = input("\nContinue anyway? (y/N): ").lower().strip()
                if choice not in ['y', 'yes']:
                    print("👋 Setup cancelled")
                    return False
            else:
                print("ℹ️  Non-critical issues - application should still work")
        else:
            print("🎉 All setup steps completed successfully!")
        
        # Save logs
        self.save_system_info()
        
        print("=" * 70)
        return True

def main():
    """Main function with enhanced error handling"""
    try:
        manager = OptimizedSystemManager()
        
        # Run complete setup
        if manager.run_complete_setup():
            # Run the application
            print("🚀 Launching Sentiment Analysis System...")
            success = manager.run_streamlit_app()
            
            if success:
                print("\n🎉 Application completed successfully!")
            else:
                print("\n❌ Application encountered issues")
                return 1
        else:
            print("\n❌ Setup failed - cannot start application")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        import traceback
        print("🔍 Debug information:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)