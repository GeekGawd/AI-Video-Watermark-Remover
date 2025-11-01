# Install ffmpeg
brew install ffmpeg

# Create Python virtual environment
cd backend
python3 -m venv venv
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt
