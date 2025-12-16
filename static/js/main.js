/**
 * Music Genre Classifier - Frontend JavaScript
 * Handles file uploads, API calls, and UI updates
 */

// Genre icons mapping
const GENRE_ICONS = {
    blues: '🎷',
    classical: '🎻',
    country: '🤠',
    disco: '🪩',
    hiphop: '🎤',
    jazz: '🎺',
    metal: '🤘',
    pop: '🎵',
    reggae: '🌴',
    rock: '🎸'
};

// Model display names
const MODEL_NAMES = {
    cnn: 'CNN (Deep Learning)',
    random_forest: 'Random Forest',
    logistic_regression: 'Logistic Regression',
    knn: 'k-Nearest Neighbors'
};

// State
let selectedFile = null;
let selectedModel = 'cnn';
let availableModels = [];

// DOM Elements
const dropZone = document.getElementById('dropZone');
const audioInput = document.getElementById('audioInput');
const selectedFileEl = document.getElementById('selectedFile');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFile');
const modelOptions = document.getElementById('modelOptions');
const predictBtn = document.getElementById('predictBtn');
const resultsSection = document.getElementById('resultsSection');
const statusIndicator = document.getElementById('statusIndicator');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkAvailableModels();
});

function initializeEventListeners() {
    // Drop zone click
    dropZone.addEventListener('click', () => audioInput.click());
    
    // File input change
    audioInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    
    // Remove file
    removeFileBtn.addEventListener('click', clearFile);
    
    // Model selection
    modelOptions.querySelectorAll('.model-btn').forEach(btn => {
        btn.addEventListener('click', () => selectModel(btn.dataset.model));
    });
    
    // Predict button
    predictBtn.addEventListener('click', classify);
}

async function checkAvailableModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        availableModels = data.models || [];
        
        // Update UI to show available/unavailable models
        modelOptions.querySelectorAll('.model-btn').forEach(btn => {
            const model = btn.dataset.model;
            if (!availableModels.includes(model)) {
                btn.classList.add('disabled');
                btn.title = 'Model not trained yet';
            }
        });
        
        // Select first available model
        if (availableModels.length > 0 && !availableModels.includes(selectedModel)) {
            selectModel(availableModels[0]);
        }
        
        updateStatus(availableModels.length > 0 ? 'Ready' : 'No models available');
    } catch (error) {
        console.error('Error checking models:', error);
        updateStatus('Error connecting to server', 'error');
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        setFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && isValidAudioFile(file)) {
        setFile(file);
    } else {
        showToast('Please drop a valid audio file', 'error');
    }
}

function isValidAudioFile(file) {
    const validExtensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a'];
    const fileName = file.name.toLowerCase();
    return validExtensions.some(ext => fileName.endsWith(ext));
}

function setFile(file) {
    selectedFile = file;
    
    // Update UI
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    dropZone.style.display = 'none';
    selectedFileEl.style.display = 'flex';
    
    // Enable predict button if model is available
    predictBtn.disabled = !availableModels.includes(selectedModel);
}

function clearFile(e) {
    if (e) e.stopPropagation();
    
    selectedFile = null;
    audioInput.value = '';
    
    dropZone.style.display = 'block';
    selectedFileEl.style.display = 'none';
    predictBtn.disabled = true;
    resultsSection.style.display = 'none';
}

function selectModel(model) {
    if (!availableModels.includes(model)) {
        showToast('This model is not available. Please train it first.', 'warning');
        return;
    }
    
    selectedModel = model;
    
    // Update UI
    modelOptions.querySelectorAll('.model-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.model === model);
    });
    
    // Update predict button state
    if (selectedFile) {
        predictBtn.disabled = false;
    }
}

async function classify() {
    if (!selectedFile || !availableModels.includes(selectedModel)) {
        return;
    }
    
    // Show loading state
    setLoading(true);
    resultsSection.style.display = 'none';
    
    try {
        const formData = new FormData();
        formData.append('audio', selectedFile);
        formData.append('model', selectedModel);
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
            showToast('Classification complete!', 'success');
        } else {
            throw new Error(result.error || 'Classification failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast(error.message, 'error');
    } finally {
        setLoading(false);
    }
}

function displayResults(result) {
    // Update genre display
    const genre = result.predicted_genre;
    document.getElementById('genreIcon').textContent = GENRE_ICONS[genre] || '🎵';
    document.getElementById('genreName').textContent = genre;
    
    // Update confidence
    const confidence = (result.confidence * 100).toFixed(1);
    document.getElementById('confidenceFill').style.width = `${confidence}%`;
    document.getElementById('confidenceValue').textContent = `${confidence}%`;
    
    // Update model name
    document.getElementById('resultModel').textContent = MODEL_NAMES[selectedModel] || selectedModel;
    
    // Update top predictions
    const predictionList = document.getElementById('predictionList');
    predictionList.innerHTML = '';
    
    if (result.top_3) {
        result.top_3.forEach((pred, index) => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.innerHTML = `
                <span class="prediction-rank">#${index + 1}</span>
                <span class="prediction-genre">${GENRE_ICONS[pred.genre] || ''} ${pred.genre}</span>
                <span class="prediction-prob">${(pred.probability * 100).toFixed(1)}%</span>
            `;
            predictionList.appendChild(item);
        });
    }
    
    // Update all probabilities
    const probabilityGrid = document.getElementById('probabilityGrid');
    probabilityGrid.innerHTML = '';
    
    if (result.probabilities) {
        // Sort by probability
        const sorted = Object.entries(result.probabilities)
            .sort((a, b) => b[1] - a[1]);
        
        sorted.forEach(([genre, prob]) => {
            const item = document.createElement('div');
            item.className = 'probability-item';
            item.innerHTML = `
                <span class="genre">${genre}</span>
                <span class="prob">${(prob * 100).toFixed(1)}%</span>
            `;
            probabilityGrid.appendChild(item);
        });
    }
    
    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setLoading(loading) {
    predictBtn.disabled = loading;
    
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoader = predictBtn.querySelector('.btn-loader');
    
    if (loading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'block';
        document.body.classList.add('loading');
        updateStatus('Processing...', 'loading');
    } else {
        btnText.style.display = 'block';
        btnLoader.style.display = 'none';
        document.body.classList.remove('loading');
        updateStatus('Ready');
    }
}

function updateStatus(text, type = 'success') {
    const dot = statusIndicator.querySelector('.status-dot');
    const textEl = statusIndicator.querySelector('.status-text');
    
    textEl.textContent = text;
    
    // Update dot color
    dot.style.background = type === 'error' ? 'var(--error)' : 
                          type === 'loading' ? 'var(--warning)' : 
                          'var(--success)';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function showToast(message, type = 'info') {
    // Remove existing toasts
    document.querySelectorAll('.toast').forEach(t => t.remove());
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 10);
    
    // Remove after delay
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
