#!/usr/bin/env python
# coding: utf-8

import requests
import pandas as pd
from icalendar import Calendar
from datetime import datetime
from pathlib import Path

# Define URLs and save paths
urls = {
    "feiertage": "https://ics.tools/Feiertage/nordrhein-westfalen.ics",
    "ferien": "https://ics.tools/Ferien/nordrhein-westfalen.ics",
}

save_dir = Path("../data/raw")
save_dir.mkdir(parents=True, exist_ok=True)

# Download and save files
for name, url in urls.items():
    response = requests.get(url)
    if response.status_code == 200:
        file_path = save_dir / f"{name}.ics"
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded and saved: {file_path}")
    else:
        print(f"Failed to download {url}")

# In[ ]:

