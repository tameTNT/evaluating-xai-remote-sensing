#!/usr/bin/env bash

subdir="sen12ms"
download_path=$DATASET_ROOT$subdir

echo "This will download the SEN12MS dataset files to $download_path with wget."

read -p "Do you want to continue with the download? (y/n) " -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo "Downloading sen12ms dataset to $download_path with wget..."
  # from https://torchgeo.readthedocs.io/en/stable/api/datasets.html#torchgeo.datasets.SEN12MS
  for season in 1158_spring 1868_summer 1970_fall 2017_winter
  do
    for source in lc s1 s2
    do
      # -nc flag skips if file already exists. -P flag specifies the directory to save to.
      wget "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs${season}_${source}.tar.gz" -nc -P "$download_path"
      # tar xvzf "ROIs${season}_${source}.tar.gz" - we handle decompression in Python
    done
  done

  for split in train test
  do
    wget "https://raw.githubusercontent.com/schmitt-muc/SEN12MS/3a41236a28d08d253ebe2fa1a081e5e32aa7eab4/splits/${split}_list.txt" -nc -P "$download_path"
  done
fi