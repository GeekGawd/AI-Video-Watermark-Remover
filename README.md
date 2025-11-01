# AI Watermark Remover


<div style="text-align: center;">
  <img src="assets/logo.png" alt="logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>
<br>
A web application for removing watermarks from videos using AI-powered inpainting with LaMa (Large Mask Inpainting).

## Results

| Original                                                                                                                                                                                                 | Output                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![sora_example.mp4](assets/sora_example.mp4)](assets/sora_example.mp4?raw=true) | [![watermark_removed_video.mp4](assets/watermark_removed_video.mp4)](assets/watermark_removed_video.mp4?raw=true) |


## Features

- Upload video files (MP4, AVI, MOV, MKV)
- Interactive video player with frame-by-frame navigation
- Manual watermark region marking with drawing tool
- AI-powered inpainting using LaMa model
- Download processed videos

## Prerequisites

- Python 3.8+
- ffmpeg (with VideoToolbox support for Apple Silicon)

## Installation

### Setup on MacOS

```bash
chmod +x setup_macos.sh
./setup_macos.sh
```

## Usage

### Start Backend

```bash
chmod +x start_backend.sh
./start_backend.sh
```

Backend runs on `http://localhost:8000`

### Start Frontend

Open index.html 

### Usage


<div style="text-align: center;">
  <img src="assets/web_app_usage.png" alt="web app usage" style="display: block; margin-left: auto; margin-right: auto;">
</div>
<br>

