import subprocess
from pathlib import Path
from typing import List
import numpy as np
from simple_lama_inpainting import SimpleLama
from fastapi import HTTPException


class VideoProcessingService:
    lama_model = None
    device = "cpu"

    @classmethod
    def configure(cls, device: str) -> str:
        """Initialize the shared Lama model on the specified device.

        Args:
            device: Preferred torch device string (e.g., 'mps' or 'cpu').

        Returns:
            The device actually used to load the model.
        """
        try:
            cls.lama_model = SimpleLama(device=device)
            cls.device = str(device)
        except Exception as exc:
            if device != "cpu":
                print(
                    f"Failed to initialize SimpleLama on device '{device}': {exc}. "
                    "Falling back to CPU."
                )
                cls.lama_model = SimpleLama(device="cpu")
                cls.device = "cpu"
            else:
                raise
        return cls.device

    @staticmethod
    def inpaint(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint a single RGB frame using the provided mask.

        Args:
            frame_rgb: RGB image as numpy array (H, W, 3).
            mask: Binary mask (H, W) where 255 indicates regions to inpaint.
        Returns:
            Inpainted RGB image as numpy array.
        """
        try:
            result = VideoProcessingService.lama_model(frame_rgb, mask)
        except RuntimeError as exc:
            # Some TorchScript ops fail on MPS; retry once on CPU if needed.
            if VideoProcessingService.device != "cpu":
                print(
                    f"Inpainting failed on device '{VideoProcessingService.device}': {exc}. "
                    "Reinitializing model on CPU and retrying."
                )
                VideoProcessingService.configure("cpu")
                result = VideoProcessingService.lama_model(frame_rgb, mask)
            else:
                raise
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        return result

    @staticmethod
    def assemble(fps: float, frames_dir: Path, output_path: Path) -> None:
        """Assemble PNG frames into an MP4 using hardware-accelerated ffmpeg.

        Args:
            fps: Frames per second for the output video.
            frames_dir: Directory containing frame_%06d.png files.
            output_path: Target MP4 file path.
        Raises:
            HTTPException: If ffmpeg returns a non-zero exit code.
        """
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            "5M",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {result.stderr}")
