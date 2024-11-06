#!/usr/bin/env bash

subdir="reBEN"
download_path=$DATASET_ROOT$subdir

# urls to download from. See https://bigearth.net/ for more information.
urls=(
  "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst"
  "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst"
  "https://zenodo.org/records/10891137/files/Reference_Maps.tar.zst"
  "https://zenodo.org/records/10891137/files/metadata.parquet"
  "https://zenodo.org/records/10891137/files/metadata_for_patches_with_snow_cloud_or_shadow.parquet"
)

echo "This will download the reBEN dataset files to $download_path with wget."
echo "These files are of the following sizes:"
for url in "${urls[@]}"
do
  echo "- $url"
  echo -n "  " # -n flag to not append a newline
  wget --spider "$url" 2>&1 | grep "Length"
done

read -p "Do you want to continue with the download? (y/n) " -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo "Downloading reBEN dataset to $download_path with wget..."
  for url in "${urls[@]}"
  do
      wget "$url" -nc -P "$download_path"  # -nc flag skips if file already exists
  done
fi
