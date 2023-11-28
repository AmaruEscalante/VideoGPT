import os
import subprocess
import shutil
import time

from tqdm import tqdm
import numpy as np
import torch
from torchvision.io import read_video, read_video_timestamps

from videogpt import VideoData
from videogpt.gpt import VideoGPT
from videogpt import download, load_vqvae, load_videogpt
from videogpt.data import preprocess
from videogpt.vqvae import VQVAE

# Download VQ-VAE
device = torch.device("cuda")
filepath = download("1FNWJtWDTX5CcVSSlINK1ZFFHuBgjBZfB", "ucf101_stride4x4x4")
vqvae = VQVAE.load_from_checkpoint(filepath).to(device)
# Download Pre-trained GPT
filepath = download("1c4CYL1joN5KDC5VYJIilFYWcDOmjWtgE", "ucf101_uncond_gpt")
gpt = VideoGPT.load_from_checkpoint(filepath).to(device)

# Generate samples
gpt.eval()

start_time = time.time()
# Generate samples
num_samples = 16  # number of samples to generate
samples = gpt.sample(num_samples)  # unconditional model does not require batch input

# Stop the timer
end_time = time.time()

# Calculate the time taken
time_taken = end_time - start_time
print(f"Time taken to generate {num_samples} samples: {time_taken} seconds")

# Process samples
b, c, t, h, w = samples.shape  # batch, channels, frames, height, width
rsamples = samples.permute(0, 2, 3, 4, 1)  # reshape to batch, frames, height, width, channels
rsamples = (rsamples.cpu().numpy() * 255).astype('uint8')  # scale to 0-255 and convert to uint8

# Directory to save frames
frames_dir = "frames"
videos_dir = "output_videos"
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)

# Save each video
for video_idx in range(b):
    video_frames_dir = os.path.join(frames_dir, f"video_{video_idx}")
    os.makedirs(video_frames_dir, exist_ok=True)

    # Save each frame of the video as an image
    for frame_idx in range(t):
        frame_path = os.path.join(video_frames_dir, f"frame_{frame_idx:05d}.png")
        frame = rsamples[video_idx, frame_idx]
        plt.imsave(frame_path, frame)

# Compile each video using ffmpeg
for video_idx in range(b):
    video_frames_dir = os.path.join(frames_dir, f"video_{video_idx}")
    output_video_path = os.path.join(videos_dir, f"output_video_{video_idx}.mp4")

    subprocess.run([
        "ffmpeg", "-y",  # Overwrite output file if it exists
        "-framerate", "5",  # Frame rate, change as needed
        "-i", os.path.join(video_frames_dir, "frame_%05d.png"),  # Input frames
        "-c:v", "libx264",  # Video codec
        "-pix_fmt", "yuv420p",  # Pixel format
        output_video_path
    ])

    # Optional: Cleanup - remove the individual video frames directory
    shutil.rmtree(video_frames_dir)

print("Videos saved.")