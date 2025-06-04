#!/usr/bin/env python
# coding: utf-8

import os
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

def download_era5(
    download_dir: str = None,
    force: bool = False,
    parallel: bool = False,
    max_workers: int = 8
):
    """
    Download ERA5 data either in parallel or consecutively.

    :param download_dir: Directory to save downloaded files. Defaults to current working directory.
    :param force: If True, re-download even if file exists.
    :param parallel: If True, download months in parallel using ThreadPoolExecutor.
    :param max_workers: Number of threads to use when parallel=True.
    """
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
            f"{d:02}" for d in range(1, 32)
        ],
        "time": [
            f"{h:02}:00" for h in range(24)
        ],
        "data_format": "grib",
        # area = [north, west, south, east]
        "area": [51.5, 7.3, 51.2, 7.7]
    }

    client = cdsapi.Client()

    # Prepare a list of tasks (year, month) tuples
    tasks = []
    ranges = [
        (2024, range(1, 13)),
        (2025, [1])
    ]
    for year, months in ranges:
        for month in months:
            tasks.append((client, dataset, request_template, year, month, download_dir, force))

    if parallel:
        print(f"Starting parallel download with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(lambda args: download_month(*args), tasks)
    else:
        print("Starting sequential download...")
        for args in tasks:
            download_month(*args)

    print("All downloads completed.")

download_era5()

# In[ ]:

