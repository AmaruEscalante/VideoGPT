#! /bin/sh
mkdir -p ${1}

# Check if gdown and wget are installed, if not, install them
if ! python -c "import gdown" &> /dev/null
then
    echo "gdown could not be found, installing..."
    pip install gdown
fi

if ! python -c "import wget" &> /dev/null
then
    echo "wget could not be found, installing..."
    pip install wget
fi

# Download MSRVTT video files
wget --no-check-certificate -P ${1} https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
unzip ${1}/MSRVTT.unzip ${1}
rm ${1}/MSRVTT.zip

# Download train/test splits
python scripts/preprocess/download.py -gfile https://drive.google.com/file/d/1uo1mNbhDLNB46Wps5rrhuo7QyGBS1w_l/view\?usp\=sharing

# Move video files into train / test directories based on train/test split
# python scripts/preprocess/msrvtt/msrvtt_split_train_test.py ${1} 1

# Delete leftover files
rm -r ${1}/MSRVTT