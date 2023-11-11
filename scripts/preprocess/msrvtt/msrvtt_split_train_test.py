import os
import json
import shutil
import argparse

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--split", help="Path to the JSON file with train test splits", required=True
)
args = parser.parse_args()

# Path to the JSON file
json_file_path = args.split

# Base directory where videos are stored
base_dir = "datasets/MSRVTT/videos/all"

# Directories for train and test splits
train_dir = "datasets/msrvtt/train"
test_dir = "datasets/msrvtt/test"

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read the JSON file
with open(json_file_path, "r") as file:
    data = json.load(file)

# Iterate through the video entries
for video in data["videos"]:
    # Construct the source file path
    source_file = os.path.join(
        base_dir, video["video_id"] + ".mp4"
    )  # Assuming the videos are in .mp4 format

    # Check the split and set the destination directory
    if video["split"] == "train":
        dest_dir = train_dir
    elif video["split"] == "validate":
        dest_dir = test_dir
    else:
        continue  # Skip if split is neither train nor test

    # Construct the destination file path
    dest_file = os.path.join(dest_dir, video["video_id"] + ".mp4")

    # Move the file
    if os.path.exists(source_file):
        shutil.move(source_file, dest_file)
    else:
        print(f"File not found: {source_file}")
