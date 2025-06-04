# download.py
import os
import requests
import zipfile
from requests.exceptions import RequestException

from concurrent.futures import ThreadPoolExecutor

import cdsapi

def download_month(client, dataset, request_template, year, month, download_dir, force):
    month_str = f"{month:02}"
    target = os.path.join(download_dir, f"ERA5-{year}-{month_str}.grib")
    
    if not force and os.path.exists(target):
        print(f"File for {year}-{month_str} already exists. Skipping...")
        return
    
    request = request_template.copy()
    request["year"] = str(year)
    request["month"] = month_str
    
    print(f"Downloading data for {year}-{month_str}...")
    client.retrieve(dataset, request, target)
    print(f"Saved {target}")

def download_era5_parallel(download_dir: str = None, force: bool = False):
    if download_dir is None:
        download_dir = os.getcwd()

    os.makedirs(download_dir, exist_ok=True)

    dataset = "reanalysis-era5-land"
    request_template = {
        "variable": [
            "2m_temperature",
            "snow_cover",
            "snowfall",
            "2m_dewpoint_temperature",
            "total_precipitation"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "grib",
        "area": [51.5, 7.3, 51.2, 7.7]
    }

    client = cdsapi.Client()

    # Prepare a list of tasks
    tasks = []
    for year, months in [(2017, [12]), (2018, range(1, 13)), (2019, range(1, 13)), (2020, range(1, 13)), (2021, range(1, 13)), (2022, range(1, 13)), (2023, range(1, 13)),(2024, range(1, 13)),(2025, [1])]:
        for month in months:
            tasks.append((client, dataset, request_template, year, month, download_dir, force))

    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=64) as executor:  # Adjust max_workers as needed
        executor.map(lambda args: download_month(*args), tasks)

    print("All downloads completed.")


def download_era5(download_dir: str = None, force: bool = False):
    import cdsapi
    import os

    if download_dir is None:
        download_dir = os.getcwd()
    
    # Ensure the directory exists
    os.makedirs(download_dir, exist_ok=True)

    dataset = "reanalysis-era5-land"
    request_template = {
        "variable": [
            "2m_temperature",
            "snow_cover",
            "snowfall",
            "2m_dewpoint_temperature",
            "total_precipitation"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "grib",
        "area": [51.5, 7.3, 51.2, 7.7]
    }

    client = cdsapi.Client()

    # Include December 2022 and all months of 2023
    for year, months in [(2017, [12]), (2018, range(1, 13)), (2019, range(1, 13)), (2020, range(1, 13)), (2021, range(1, 13)), (2022, range(1, 13)), (2023, range(1, 13))]:
        for month in months:
            month_str = f"{month:02}"  # Format month as two digits (01, 02, ..., 12)
            target = os.path.join(download_dir, f'ERA5-{year}-{month_str}.grib')
            
            # Skip check if force=True
            if not force and os.path.exists(target):
                print(f"File for {year}-{month_str} already exists. Skipping download.")
                continue

            # Update the request with the current year and month
            request = request_template.copy()
            request["year"] = str(year)
            request["month"] = month_str

            print(f"Downloading data for {year}-{month_str}...")
            client.retrieve(dataset, request, target)

    print("Download completed for all months.")
    

def download_file(url: str, download_dir: str) -> str:
    """
    Downloads a file from a given URL and saves it in the specified directory.
    Returns the path to the downloaded file.
    """
    filename = os.path.basename(url)
    filepath = os.path.join(download_dir, filename)

    if not os.path.exists(filepath):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

            print(f"Downloaded {filename}.")
        except RequestException as e:
            print(f"Failed to download {url}: {e}")
            raise
    else:
        print(f"{filename} already exists.")

    return filepath


def unzip_file(filepath: str, extract_to: str):
    """
    Unzips the given ZIP file to the specified directory.
    """
    if filepath.endswith('.zip'):
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Unzipped {os.path.basename(filepath)}.")
        except zipfile.BadZipFile as e:
            print(f"Failed to unzip {filepath}: {e}")
            raise
    else:
        print(f"{os.path.basename(filepath)} is not a ZIP file, no extraction needed.")


def download_and_unzip(identifier: str, download_dir: str = None):
    """
    Downloads a file from the given URL or nickname and unzips it if it's a ZIP file.
    """
    if download_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        download_dir = os.path.join(os.path.dirname(current_dir), "data", "raw")

    # Determine if the identifier is a URL or a nickname
    if identifier in DATASET_URLS:
        url = DATASET_URLS[identifier]
    else:
        url = identifier

    filepath = download_file(url, download_dir)
    unzip_file(filepath, download_dir)


DATASET_URLS = {
    "Wohnbezirke_Hagen": "https://www.hagen.de/web/media/files/fb/stadtplaene/wahlen_und_statistik/Wohnbezirke_Hagen.zip",
    "Statistische_Bezirke_Hagen": "https://www.hagen.de/web/media/files/fb/stadtplaene/wahlen_und_statistik/Statistische_Bezirke_Hagen.zip",
    "Stadtbezirke": "https://www.hagen.de/web/media/files/fb/stadtplaene/wahlen_und_statistik/Stadtbezirke.zip",
    "Hauskoordinaten": "http://www.stadtplan.hagen.de/StrVz/Hauskoordinaten.csv"
}


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(os.path.dirname(current_dir), "data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    for nickname in DATASET_URLS.keys():
        try:
            download_and_unzip(nickname, raw_data_dir)
        except Exception as e:
            print(f"Error processing {nickname}: {e}")


if __name__ == "__main__":
    main()
