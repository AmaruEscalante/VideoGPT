#!/bin/sh

mkdir -p ${1}

# Function to check if unrar is installed and install it locally if not
install_unrar_if_needed() {
    if ! command -v unrar &> /dev/null; then
        echo "unrar not found, installing locally..."
        wget https://www.rarlab.com/rar/unrarsrc-6.1.7.tar.gz
        tar -xzvf unrarsrc-6.1.7.tar.gz
        cd unrarsrc-6.1.7
        make
        cd ..
        export UNRAR="$PWD/unrarsrc-6.1.7/unrar"
    else
        export UNRAR="unrar"
    fi
}

# Install unrar locally if needed
install_unrar_if_needed

# Download UCF-101 video files
wget --no-check-certificate -P ${1} https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

# Unrar UCF-101 dataset
$UNRAR x ${1}/UCF101.rar ${1}
rm ${1}/UCF101.rar

# Download UCF-101 train/test splits
wget --no-check-certificate -P ${1} https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip ${1}/UCF101TrainTestSplits-RecognitionTask.zip -d ${1}
rm ${1}/UCF101TrainTestSplits-RecognitionTask.zip

# Move video files into train / test directories based on train/test split
python scripts/preprocess/ucf101/ucf_split_train_test.py ${1} 1

# Delete leftover files
rm -r ${1}/UCF-101
