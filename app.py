"""
Flask web application for Music Genre Classification.
Provides a web-based HMI for uploading audio and getting genre predictions.
"""
import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, redirect, url_for
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    FLASK_CONFIG, SERVER_CONFIG, MODELS_DIR,
    STATIC_DIR, TEMPLATES_DIR, GENRES
)
from src.predict import GenrePredictor, get_available_models


# Initialize Flask app
app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR)
)

# Configure app
app.config["SECRET_KEY"] = FLASK_CONFIG["SECRET_KEY"]
app.config["MAX_CONTENT_LENGTH"] = FLASK_CONFIG["MAX_CONTENT_LENGTH"]

# Setup CORS for API access
CORS(app)

# Create upload folder
UPLOAD_FOLDER = STATIC_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = FLASK_CONFIG["ALLOWED_EXTENSIONS"]

# Global predictor cache
_predictors = {}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_predictor(model_type: str = "cnn") -> GenrePredictor:
    """Get or create predictor instance."""
    if model_type not in _predictors:
        try:
            _predictors[model_type] = GenrePredictor(model_type=model_type)
        except FileNotFoundError:
            return None
    return _predictors[model_type]


def cleanup_old_uploads(max_age_hours: int = 24):
    """Remove uploaded files older than max_age_hours."""
    now = datetime.now()
    for file_path in UPLOAD_FOLDER.glob("*"):
        if file_path.is_file():
            file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age.total_seconds() > max_age_hours * 3600:
                try:
                    file_path.unlink()
                except Exception:
                    pass


# Routes
@app.route("/")
def index():
    """Main page."""
    available_models = get_available_models()
    return render_template(
        "index.html",
        genres=GENRES,
        available_models=available_models
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    """API endpoint for genre prediction."""
    try:
        # Check if file was uploaded
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files["audio"]
        
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Get model type
        model_type = request.form.get("model", "cnn")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = UPLOAD_FOLDER / unique_filename
        file.save(str(file_path))
        
        try:
            # Get predictor
            predictor = get_predictor(model_type)
            if predictor is None:
                return jsonify({
                    "error": f"Model '{model_type}' not available. Please train models first."
                }), 400
            
            # Make prediction
            result = predictor.predict(str(file_path))
            result["filename"] = filename
            result["model_used"] = model_type
            
            return jsonify(result)
            
        finally:
            # Clean up uploaded file
            try:
                file_path.unlink()
            except Exception:
                pass
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/models")
def list_models():
    """Get list of available models."""
    available = get_available_models()
    return jsonify({
        "models": available,
        "default": "cnn" if "cnn" in available else (available[0] if available else None)
    })


@app.route("/api/genres")
def list_genres():
    """Get list of supported genres."""
    return jsonify({"genres": GENRES})


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_available": len(get_available_models()),
        "genres_supported": len(GENRES)
    })


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(str(STATIC_DIR), filename)


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


def create_app():
    """Application factory."""
    return app


if __name__ == "__main__":
    # Cleanup old uploads on startup
    cleanup_old_uploads()
    
    # Check for available models
    available = get_available_models()
    if not available:
        print("\n" + "=" * 60)
        print("WARNING: No trained models found!")
        print("Please run the preprocessing and training scripts first:")
        print("  1. python src/preprocess.py")
        print("  2. python src/train.py")
        print("=" * 60 + "\n")
    else:
        print(f"\nAvailable models: {', '.join(available)}")
    
    # Run server
    print(f"\nStarting server at http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    app.run(
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        debug=SERVER_CONFIG["debug"]
    )
