import argparse
import os
import subprocess
import time

import matplotlib.pyplot as plt
import torch

from videogpt.gpt import VideoGPT
from videogpt import download
from videogpt.vqvae import VQVAE

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=16, help="Batch size for processing")
parser.add_argument(
    "--vqvae", type=str, default="ucf101_stride4x4x4", help="path to vqvae ckpt"
)
parser.add_argument(
    "--gpt", type=str, default="ucf101_uncond_gpt", help="path to gpt ckpt"
)
args = parser.parse_args()


def main(num_samples=16, vqvae="ucf101_stride4x4x4", gpt="ucf101_uncond_gpt"):
    # Download VQ-VAE
    device = torch.device("cuda")
    if vqvae == "ucf101_stride4x4x4":
        filepath = download("1FNWJtWDTX5CcVSSlINK1ZFFHuBgjBZfB", "ucf101_stride4x4x4")
        vqvae = VQVAE.load_from_checkpoint(filepath).to(device)
    else:
        vqvae = VQVAE.load_from_checkpoint(vqvae).to(device)
    # Download Pre-trained GPT
    if gpt == "ucf101_uncond_gpt":
        filepath = download("1c4CYL1joN5KDC5VYJIilFYWcDOmjWtgE", "ucf101_uncond_gpt")
        gpt = VideoGPT.load_from_checkpoint(filepath).to(device)
    else:
        gpt = VideoGPT.load_from_checkpoint(gpt).to(device)

    # Generate samples
    gpt.eval()

    start_time = time.time()
    # Generate samples
    # num_samples = 16  # number of samples to generate
    samples = gpt.sample(
        num_samples
    )  # unconditional model does not require batch input

    # Stop the timer
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time
    print(f"Time taken to generate {num_samples} samples: {time_taken} seconds")

    # Process samples
    b, c, t, h, w = samples.shape  # batch, channels, frames, height, width
    rsamples = samples.permute(
        0, 2, 3, 4, 1
    )  # reshape to batch, frames, height, width, channels
    rsamples = (rsamples.cpu().numpy() * 255).astype(
        "uint8"
    )  # scale to 0-255 and convert to uint8

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

        subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-framerate",
                "5",  # Frame rate, change as needed
                "-i",
                os.path.join(video_frames_dir, "frame_%05d.png"),  # Input frames
                "-c:v",
                "libx264",  # Video codec
                "-pix_fmt",
                "yuv420p",  # Pixel format
                output_video_path,
            ]
        )

        # Optional: Cleanup - remove the individual video frames directory
        # shutil.rmtree(video_frames_dir)

    print("Videos saved.")


if __name__ == "__main__":
    main(
        num_samples=args.samples,
        vqvae=args.vqvae,
        gpt=args.gpt,
    )
