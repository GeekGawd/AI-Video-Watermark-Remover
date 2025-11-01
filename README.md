# AI Watermark Remover

<div>
  <p align="center" width="100%">
    <img src="assets/logo.png" />
  </p>
</div>

A web application for removing watermarks from videos using AI-powered inpainting with LaMa (Large Mask Inpainting).

This can be used to remove watermarks from AI generated videos. Supports:
- Sora
- Veo 3
- Kling
- Hailuo

and more...

## Results

| Original                                                                                                                                                                                                 | Output                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <video src="https://github.com/user-attachments/assets/4a13606c-cf6d-4391-b421-af03a892905b" width="352" height="720"> | <video src="https://github.com/user-attachments/assets/3f51cce7-4ccc-41f8-8d99-1266aed2e373" width="352" height="720"> |
  
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

