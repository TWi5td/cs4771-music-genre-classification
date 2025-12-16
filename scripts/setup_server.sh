#!/bin/bash
# =============================================================================
# Music Genre Classifier - Server Setup Script
# For Ubuntu Server (headless)
# =============================================================================

set -e

echo "=============================================="
echo "  Music Genre Classifier - Server Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

# Configuration
VENV_DIR="$PROJECT_DIR/venv"
SERVICE_NAME="music-genre-classifier"
PORT=${PORT:-5000}
HOST=${HOST:-0.0.0.0}

echo -e "\n${YELLOW}[1/6] Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Found $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.9+${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[2/6] Installing system dependencies...${NC}"
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-pip python3-venv libsndfile1 ffmpeg
    echo -e "${GREEN}✓ System dependencies installed${NC}"
else
    echo -e "${YELLOW}! Not a Debian-based system, skipping apt-get${NC}"
fi

echo -e "\n${YELLOW}[3/6] Creating virtual environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

echo -e "\n${YELLOW}[4/6] Installing Python dependencies...${NC}"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$PROJECT_DIR/requirements.txt" -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo -e "\n${YELLOW}[5/6] Creating necessary directories...${NC}"
mkdir -p "$PROJECT_DIR/data/raw"
mkdir -p "$PROJECT_DIR/data/processed"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/static/uploads"
echo -e "${GREEN}✓ Directories created${NC}"

echo -e "\n${YELLOW}[6/6] Creating systemd service file...${NC}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Music Genre Classification Web Service
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$VENV_DIR/bin"
Environment="HOST=$HOST"
Environment="PORT=$PORT"
ExecStart=$VENV_DIR/bin/gunicorn --bind $HOST:$PORT --workers 2 --timeout 120 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}✓ Systemd service created${NC}"

echo -e "\n=============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "=============================================="
echo ""
echo "To start the service:"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl start $SERVICE_NAME"
echo "  sudo systemctl enable $SERVICE_NAME  # Auto-start on boot"
echo ""
echo "To check status:"
echo "  sudo systemctl status $SERVICE_NAME"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "For development (without systemd):"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "The web interface will be available at:"
echo "  http://<your-server-ip>:$PORT"
echo ""
echo -e "${YELLOW}Note: Download the GTZAN dataset and run preprocessing before use.${NC}"
echo "  python src/preprocess.py"
echo "  python src/train.py"
