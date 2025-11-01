import os
# Enable MPS fallback to CPU for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import shutil
import tempfile
from pathlib import Path
import uuid
import asyncio
import json
from services import VideoProcessingService


app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize lama model with MPS fallback enabled
# MPS will be used where supported, and automatically fall back to CPU for FFT operations
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device} (with MPS fallback to CPU for unsupported ops)")

# Use tmpfs for ephemeral storage
TMP_DIR = Path(tempfile.gettempdir()) / "watermark_remover"
TMP_DIR.mkdir(exist_ok=True)

# Store video metadata and processing progress in memory
video_store: Dict[str, dict] = {}
processing_progress: Dict[str, dict] = {}

# Cleanup temp artifacts
def _cleanup_tmp_dir() -> None:
    """Remove leftover uploads/outputs from previous runs."""
    if not TMP_DIR.exists():
        return
    for path in TMP_DIR.iterdir():
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"Failed to remove {path}: {exc}")
    video_store.clear()
    processing_progress.clear()

# Process could have been stopped ungracefully, so ensure cleanup before startup
_cleanup_tmp_dir()

actual_device = VideoProcessingService.configure(device=device)
device = actual_device

@app.on_event("shutdown")
async def _on_shutdown_cleanup() -> None:
    _cleanup_tmp_dir()

class MaskRegion(BaseModel):
    x: int
    y: int
    width: int
    height: int

class ProcessRequest(BaseModel):
    video_id: str
    regions: List[MaskRegion]


def _prepare_mask(height: int, width: int, regions: List[MaskRegion]) -> np.ndarray:
    """Create a binary mask from rectangular regions for inpainting.

    Args:
        height: Frame height.
        width: Frame width.
        regions: List of rectangles to be removed (set to 255).
    Returns:
        Numpy uint8 mask (H, W) with 255 in regions to inpaint.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        x1 = max(0, region.x)
        y1 = max(0, region.y)
        x2 = min(width, region.x + region.width)
        y2 = min(height, region.y + region.height)
        mask[y1:y2, x1:x2] = 255
    return mask


def _process_frames(cap: cv2.VideoCapture, mask: np.ndarray, temp_dir: Path, processed_frames: list):
    """Read frames from capture, inpaint using mask, and write PNGs to temp_dir.

    Args:
        cap: OpenCV VideoCapture already opened on input video.
        mask: Binary mask used for inpainting.
        temp_dir: Directory to store output frames.
        processed_frames: List to append written frame paths.
    """
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = VideoProcessingService.inpaint(frame_rgb, mask)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        frame = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        frame_path = temp_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        processed_frames.append(frame_path)
        frame_idx += 1


def _assemble_video(fps: float, temp_dir: Path, output_path: Path):
    """Assemble temp_dir PNG frames into an MP4 at given fps using ffmpeg."""
    VideoProcessingService.assemble(fps, temp_dir, output_path)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file to tmpfs"""
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    video_id = str(uuid.uuid4())
    video_path = TMP_DIR / f"{video_id}.mp4"
    
    # Write to tmpfs in chunks for memory efficiency
    with open(video_path, "wb") as buffer:
        while chunk := await file.read(8192):  # 8KB chunks
            buffer.write(chunk)
    
    # Get video metadata
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    # Store metadata in memory
    video_store[video_id] = {
        "path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }
    
    return {
        "video_id": video_id,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }

@app.get("/video/{video_id}")
async def get_video(video_id: str):
    """Stream uploaded video from tmpfs"""
    if video_id not in video_store:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = video_store[video_id]["path"]
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    def iterfile():
        with open(video_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk
    
    return StreamingResponse(iterfile(), media_type="video/mp4")

@app.post("/process")
async def process_video(request: ProcessRequest):
    """Process video with lama inpainting and track progress"""
    if request.video_id not in video_store:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_info = video_store[request.video_id]
    video_path = video_info["path"]
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    output_id = str(uuid.uuid4())
    temp_dir = TMP_DIR / f"frames_{output_id}"
    temp_dir.mkdir(exist_ok=True)
    output_path = TMP_DIR / f"{output_id}.mp4"
    
    # Initialize progress tracking
    processing_progress[output_id] = {
        "status": "processing",
        "progress": 0,
        "total_frames": video_info["frame_count"],
        "current_frame": 0,
        "stage": "inpainting"
    }
    
    # Process in background
    asyncio.create_task(_process_video_async(
        video_path, output_path, temp_dir, output_id, request.regions,
        video_info["fps"], video_info["width"], video_info["height"],
        video_info["frame_count"]
    ))
    
    return {
        "output_id": output_id,
        "message": "Processing started"
    }

@app.get("/progress/{output_id}")
async def get_progress(output_id: str):
    """Get processing progress via Server-Sent Events"""
    if output_id not in processing_progress:
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    async def event_generator():
        while True:
            if output_id not in processing_progress:
                break
            
            progress_data = processing_progress[output_id]
            yield f"data: {json.dumps(progress_data)}\n\n"
            
            if progress_data["status"] in ["completed", "error"]:
                break
            
            await asyncio.sleep(0.5)  # Update every 500ms
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/output/{output_id}")
async def get_output(output_id: str):
    """Stream processed video from tmpfs"""
    output_path = TMP_DIR / f"{output_id}.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    
    def iterfile():
        with open(output_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk
    
    return StreamingResponse(iterfile(), media_type="video/mp4", headers={
        "Content-Disposition": "attachment; filename=processed_video.mp4"
    })

@app.get("/health")
async def health_check():
    return {"status": "ok", "device": device}

async def _process_video_async(video_path, output_path, temp_dir, output_id, regions, fps, width, height, total_frames):
    """Process video asynchronously with progress tracking"""
    try:
        # Read video
        cap = cv2.VideoCapture(str(video_path))
        
        # Create combined mask for all regions
        mask = _prepare_mask(height, width, regions)
        
        # Process frames with progress tracking
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                result = VideoProcessingService.inpaint(frame_rgb, mask)
            except Exception as exc:
                processing_progress[output_id]["status"] = "error"
                processing_progress[output_id]["error"] = str(exc)
                raise
            if not isinstance(result, np.ndarray):
                result = np.array(result)
            frame = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
            frame_path = temp_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            
            frame_idx += 1
            processing_progress[output_id]["current_frame"] = frame_idx
            processing_progress[output_id]["progress"] = int((frame_idx / total_frames) * 90)  # 90% for inpainting
            
            # Yield control to event loop every 10 frames
            if frame_idx % 10 == 0:
                await asyncio.sleep(0)
        
        cap.release()
        
        # Delete uploaded video after processing
        try:
            video_path.unlink()
        except:
            pass
        
        # Reassemble video
        processing_progress[output_id]["stage"] = "assembling"
        processing_progress[output_id]["progress"] = 95
        
        _assemble_video(fps, temp_dir, output_path)
        
        processing_progress[output_id]["status"] = "completed"
        processing_progress[output_id]["progress"] = 100
        
    except Exception as e:
        processing_progress[output_id]["status"] = "error"
        processing_progress[output_id]["error"] = str(e)
    
    finally:
        # Cleanup temp frames
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
