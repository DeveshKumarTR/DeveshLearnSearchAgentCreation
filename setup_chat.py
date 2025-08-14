#!/usr/bin/env python3
"""
Setup script for the Chat Agent with Django UI
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, check=True, shell=True):
    """Run a command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=shell, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor} detected")

def check_ollama():
    """Check if Ollama is installed and running"""
    print("\nChecking Ollama installation...")
    
    # Check if ollama command exists
    result = run_command("ollama --version", check=False)
    if result is None or result.returncode != 0:
        print("⚠ Ollama not found. Please install from https://ollama.ai")
        return False
    
    print("✓ Ollama is installed")
    
    # Check if Ollama server is running
    result = run_command("ollama list", check=False)
    if result is None or result.returncode != 0:
        print("⚠ Ollama server not running. Start with: ollama serve")
        return False
    
    print("✓ Ollama server is running")
    return True

def setup_django():
    """Setup Django application"""
    print("\nSetting up Django...")
    
    # Change to django_ui directory
    django_dir = Path(__file__).parent / "django_ui"
    if not django_dir.exists():
        print("Error: django_ui directory not found")
        return False
    
    os.chdir(django_dir)
    
    # Install requirements
    print("Installing Python dependencies...")
    result = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    if result is None:
        return False
    
    # Run migrations
    print("Running database migrations...")
    result = run_command(f"{sys.executable} manage.py migrate")
    if result is None:
        return False
    
    # Collect static files
    print("Collecting static files...")
    result = run_command(f"{sys.executable} manage.py collectstatic --noinput")
    if result is None:
        return False
    
    print("✓ Django setup complete")
    return True

def suggest_models():
    """Suggest Ollama models to install"""
    print("\nRecommended Ollama models:")
    models = [
        "llama2:7b - Good general purpose model",
        "mistral:7b - Fast and efficient",
        "codellama:7b - Specialized for code",
        "phi:2.7b - Lightweight model"
    ]
    
    for model in models:
        print(f"  - {model}")
    
    print("\nTo install a model, run: ollama pull <model-name>")
    print("Example: ollama pull llama2")

def create_env_file():
    """Create environment file template"""
    django_dir = Path(__file__).parent / "django_ui"
    env_file = django_dir / ".env"
    
    if env_file.exists():
        print(f"✓ Environment file already exists: {env_file}")
        return
    
    env_content = """# Django Settings
DEBUG=True
SECRET_KEY=django-insecure-change-this-in-production
ALLOWED_HOSTS=localhost,127.0.0.1

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama2

# Search Agent Configuration (optional)
SEARCH_ENABLED=False
PINECONE_API_KEY=your-pinecone-key-here
OPENAI_API_KEY=your-openai-key-here
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✓ Created environment file: {env_file}")
    print("  Please edit this file with your actual configuration")

def main():
    """Main setup function"""
    print("🚀 Setting up Chat Agent with Django UI")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check Ollama
    ollama_ok = check_ollama()
    
    # Setup Django
    django_ok = setup_django()
    
    # Create environment file
    create_env_file()
    
    # Suggest models
    if ollama_ok:
        suggest_models()
    
    # Final instructions
    print("\n" + "=" * 50)
    if django_ok:
        print("✅ Setup completed successfully!")
        print("\nTo start the application:")
        print("1. cd django_ui")
        if not ollama_ok:
            print("2. Start Ollama: ollama serve")
            print("3. Install a model: ollama pull llama2")
            print("4. python manage.py runserver")
        else:
            print("2. python manage.py runserver")
        print("5. Open http://localhost:8000 in your browser")
    else:
        print("❌ Setup encountered errors. Please check the output above.")
    
    print("\nFor more information, see chat_agent/README.md")

if __name__ == "__main__":
    main()
