#!/usr/bin/env python
# coding: utf-8

import os
import requests
import zipfile

def download_and_unzip(url: str, download_dir: str):
    # Extract the filename from the URL
    filename = os.path.basename(url)
    filepath = os.path.join(download_dir, filename)

    # Check if file already exists
    if not os.path.exists(filepath):
        # Download the file
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}.")
    else:
        print(f"{filename} already exists.")

    # Check if the file is a ZIP file and unzip it
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        print(f"Unzipped {filename}.")
    else:
        print(f"{filename} is not a ZIP file, no extraction needed.")

download_and_unzip("https://www.hagen.de/web/media/files/fb/stadtplaene/wahlen_und_statistik/Wohnbezirke_Hagen.zip", "../data/raw")
download_and_unzip("https://www.hagen.de/web/media/files/fb/stadtplaene/wahlen_und_statistik/Statistische_Bezirke_Hagen.zip", "../data/raw")
download_and_unzip("https://www.hagen.de/web/media/files/fb/stadtplaene/wahlen_und_statistik/Stadtbezirke.zip", "../data/raw")
download_and_unzip("http://www.stadtplan.hagen.de/StrVz/Hauskoordinaten.csv", "../data/raw")

# In[ ]:

