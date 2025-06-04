#!/usr/bin/env python
# coding: utf-8

import requests
import pandas as pd
from icalendar import Calendar
from datetime import datetime
from pathlib import Path
import os

# Function to parse ICS files
def parse_ics(file_path):
    with open(file_path, "rb") as f:
        cal = Calendar.from_ical(f.read())

    events = []
    for component in cal.walk():
        if component.name == "VEVENT":
            event_name = str(component.get("summary"))
            start = component.get("dtstart").dt
            end = component.get("dtend").dt if component.get("dtend") else start
            events.append({"event": event_name, "start": start, "end": end})

    return events

save_dir = Path("../data/raw")
# Read and parse both files

feiertage = []
feiertage.extend(parse_ics(save_dir / "feiertage.ics"))

# Convert to DataFrame
feiertage = pd.DataFrame(feiertage)
feiertage

ferien = []
ferien.extend(parse_ics(save_dir / "ferien.ics"))

# Convert to DataFrame
ferien = pd.DataFrame(ferien)

ferien

date_range = pd.date_range(start="2018-01-01", end="2024-12-31", freq="D")

dates_df = pd.DataFrame({"date": date_range.date})

dates_df["Feiertag"] = dates_df["date"].isin(feiertage["start"])

dates_df

# Function to safely extract date from ICS files
def extract_date(dt):
    """Ensures the extracted date is a date object."""
    if isinstance(dt, datetime):
        return dt.date()  # Convert datetime to date
    return dt  # Already a date object

# Function to parse ICS files with date ranges
def parse_ferien_ics(file_path):
    with open(file_path, "rb") as f:
        cal = Calendar.from_ical(f.read())

    events = []
    for component in cal.walk():
        if component.name == "VEVENT":
            event_name = str(component.get("summary"))
            start = extract_date(component.get("dtstart").dt)
            end = extract_date(component.get("dtend").dt)  # End date is exclusive
            events.append({"event": event_name, "start": start, "end": end})

    return events

# Read and parse the Ferien file
save_dir = Path("../data/raw")
ferien_data = parse_ferien_ics(save_dir / "ferien.ics")

# Convert to DataFrame
ferien_df = pd.DataFrame(ferien_data)
ferien_df

# Initialize the "ferien" column in dates_df
dates_df["Ferien"] = False

# Iterate over the Ferien dataset and mark the correct date range in dates_df
for _, row in ferien_df.iterrows():
    mask = (dates_df["date"] >= row["start"]) & (dates_df["date"] < row["end"])  # Exclude end date
    dates_df.loc[mask, "Ferien"] = True

dates_df.head(20)

output_path = os.path.join("../data/interim", "Feiertage-Ferien.parquet")
dates_df.to_parquet(output_path, index=False)

