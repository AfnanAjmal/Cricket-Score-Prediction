import os
from pathlib import Path

# Configuration settings for the ML Models Dashboard

# Models directory configuration
# You can set this via environment variable MODELS_DIR 
# or modify the DEFAULT_MODELS_DIR below
DEFAULT_MODELS_DIR = "models"
MODELS_DIR = os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR)

# Validate models directory exists
def validate_models_directory():
    """Validate that the models directory exists and contains required files"""
    models_path = Path(MODELS_DIR)
    
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")
    
    # Check for required files
    model_info_file = models_path / "cricket_model_info_list.pkl"
    if not model_info_file.exists():
        raise FileNotFoundError(f"Model info file not found: {model_info_file}")
    
    return True

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Frontend Configuration
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501") 