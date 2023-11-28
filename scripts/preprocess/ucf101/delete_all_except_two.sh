#!/bin/bash

# Check if a directory path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory-path>"
    exit 1
fi

# Directory containing the folders
dir=$1

# Check if the provided directory exists
if [ ! -d "$dir" ]; then
    echo "Error: Directory $dir does not exist."
    exit 1
fi

# Loop through each folder in the directory
for folder in "$dir"/*; do
    # Extract the folder name
    foldername=$(basename "$folder")

    # Check if the folder name is not 'BreastStroke' or 'BaseballPitch'
    if [ "$foldername" != "BreastStroke" ] && [ "$foldername" != "BaseballPitch" ]; then
        # Remove the folder
        echo "Deleting $foldername"
        rm -rf "$folder"
    fi
done

echo "Deletion complete."
