#!/bin/bash
# Download and extract N-MNIST dataset
set -e

echo "Downloading N-MNIST dataset..."
mkdir -p data/nmnis_t

# Download train set
echo "Downloading train set..."
curl -L -o data/nmnis_t/train.zip https://www.garrickorchard.com/wp-content/uploads/2021/05/Train.zip

# Download test set
echo "Downloading test set..."
curl -L -o data/nmnis_t/test.zip https://www.garrickorchard.com/wp-content/uploads/2021/05/Test.zip

# Extract
echo "Extracting..."
cd data/nmnis_t
unzip -q train.zip
unzip -q test.zip

echo "N-MNIST dataset ready in data/nmnis_t/"
echo "Run with: NMNIST_REAL_DATA=1 cargo run --example nmnis_t_train --release"
