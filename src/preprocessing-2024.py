
"""
Single-script preprocessing pipeline that:
  1. Loads raw Einsatzdaten 2024, combines them.
  2. Adds Rettungswache coordinates (FZ_lat, FZ_lon).
  3. Calculates distances/durations with ORS (saved in an intermediate Parquet).
  4. Merges with ERA5 weather data.
  5. Merges with Feiertage/Ferien data.
  6. Performs final cleaning and filtering.
  7. Saves the final dataset as selected-data.parquet.
"""

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from pathlib import Path
import openrouteservice
from datetime import datetime
from icalendar import Calendar

# ------------------------------------------------------------------------------
# CONFIG & PATHS
# ------------------------------------------------------------------------------
BASE_DATA_DIR = Path("data")  
RAW_DIR      = BASE_DATA_DIR / "raw"
INTERIM_DIR  = BASE_DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Input raw data
EINSATZ_2024_FILE       = RAW_DIR / "2025-03-31 einsatzdaten24.xlsx"
ERA5_COMBINED_FILE      = INTERIM_DIR / "ERA5_combined.csv"  
ICS_FEIERTAGE_FILE      = RAW_DIR / "feiertage.ics"
ICS_FERIEN_FILE         = RAW_DIR / "ferien.ics"

# Outputs
COMBINED_EINSATZ_PARQUET       = INTERIM_DIR / "combined_einsatz_data_2024.parquet"
COMBINED_EINSATZ_ORS_PARQUET   = INTERIM_DIR / "combined_einsatz_ORS_2024.parquet"   
COMBINED_EINSATZ_WEATHER_PARQUET = INTERIM_DIR / "combined_einsatz_weather_2024.parquet"
COMBINED_EINSATZ_WEATHER_DATES_PARQUET = INTERIM_DIR / "combined_einsatz_weather_dates_2024.parquet"
FEIERTAGE_FERIEN_PARQUET       = INTERIM_DIR / "Feiertage-Ferien_2024.parquet"
ALL_DATA_PARQUET               = INTERIM_DIR / "all-data_2024.parquet"
FINAL_DATA_PARQUET             = INTERIM_DIR / "selected-data_2024.parquet"

# ------------------------------------------------------------------------------
# STEP 1: COMBINE EINSATZDATEN (2018-2022 and 2023)
# ------------------------------------------------------------------------------
# Common date columns
DATE_COLUMNS = [
    "MELDUNGSEINGANG", "ALARMZEIT", "Status 3", "Status 4",
    "Status 7", "Status 8", "Status 1", "Status 2"
]

def _safe_float_conversion(value):
    """Handles float conversion from German decimal-comma."""
    if pd.isna(value) or value == '':
        return None
    return float(value.replace(",", "."))

def _transform_coordinates(rechtswert, hochwert, transformer):
    """Transform coordinates from EPSG:32632 to EPSG:4326."""
    try:
        lon, lat = transformer.transform(rechtswert, hochwert)
        return pd.Series([lat, lon])
    except:
        return pd.Series([None, None])

def load_raw_einsatzdaten(file_path, 
                          date_cols, 
                          has_einsatzstichwort_1=False, 
                          has_bezeichnung_zielort=False, 
                          has_zielort=False):
    """
    Loads a single CSV of Einsatzdaten with consistent transformations:
      - Parse dates
      - Convert to categories where applicable
      - Convert coordinate columns from EPSG:32632 -> EPSG:4326
    """
    transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    df = pd.read_csv(
        file_path,
        sep=';',
        parse_dates=date_cols,
        dayfirst=True,
        converters={
            "EINSATZORT_X": _safe_float_conversion,
            "EINSATZORT_Y": _safe_float_conversion,
            "ZIELORT_X": _safe_float_conversion,
            "ZIELORT_Y": _safe_float_conversion
        }
    )
    # Floor date columns to minute
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].dt.floor('min')

    # Convert some columns to category
    if 'EINSATZSTICHWORT' in df.columns:
        df['EINSATZSTICHWORT'] = df['EINSATZSTICHWORT'].astype('category')
    if has_einsatzstichwort_1 and 'EINSATZSTICHWORT_1' in df.columns:
        df['EINSATZSTICHWORT_1'] = df['EINSATZSTICHWORT_1'].astype('category')
    if 'EINSATZNUMMER' in df.columns:
        df['EINSATZNUMMER'] = df['EINSATZNUMMER'].astype(pd.Int64Dtype())
    if 'HAUPTEINSATZNUMMER' in df.columns:
        df['HAUPTEINSATZNUMMER'] = df['HAUPTEINSATZNUMMER'].astype(pd.Int64Dtype())

    # Drop fully empty rows
    df.dropna(how='all', inplace=True)

    # Convert SONDERSIGNAL (J/N) to booleans if present
    if 'SONDERSIGNAL' in df.columns:
        df['SONDERSIGNAL'] = df['SONDERSIGNAL'].apply(lambda x: x if x in ['J', 'N'] else np.nan)
        df['SONDERSIGNAL'] = df['SONDERSIGNAL'].map({'J': True, 'N': False}).astype(pd.BooleanDtype())

    # Convert typical category columns if present
    for cat_col in ['Standort FZ', 'Einsatzmittel-Typ', 'FUNKRUFNAME']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype('category')
    if has_bezeichnung_zielort and 'Bezeichnung Zielort' in df.columns:
        df['Bezeichnung Zielort'] = df['Bezeichnung Zielort'].astype('category')
    if has_zielort and 'ZIELORT' in df.columns:
        df['ZIELORT'] = df['ZIELORT'].astype('category')

    # Add WGS84 lat/lon for EINSATZORT
    if 'EINSATZORT_X' in df.columns and 'EINSATZORT_Y' in df.columns:
        df[['EINSATZORT_lat', 'EINSATZORT_lon']] = df.apply(
            lambda row: _transform_coordinates(row['EINSATZORT_X'], row['EINSATZORT_Y'], transformer), axis=1
        )

    return df

def combine_einsatzdaten():
    """Combine the two Einsatzdaten CSV files and save to Parquet."""
    print("Loading Einsatzdaten 2023...")
    df_2023 = load_raw_einsatzdaten(
        EINSATZ_2023_FILE,
        date_cols=DATE_COLUMNS,
        has_einsatzstichwort_1=True,
        has_bezeichnung_zielort=True,
        has_zielort=True
    )

    print("Loading Einsatzdaten 2018-22...")
    df_2018_22 = load_raw_einsatzdaten(
        EINSATZ_2018_22_FILE,
        date_cols=DATE_COLUMNS,
        has_einsatzstichwort_1=False,
        has_bezeichnung_zielort=False,
        has_zielort=False
    )

    print("Combining datasets...")
    combined = pd.concat([df_2023, df_2018_22], ignore_index=True)
    print(f"Combined shape: {combined.shape}")

    print(f"Saving to {COMBINED_EINSATZ_PARQUET}...")
    combined.to_parquet(COMBINED_EINSATZ_PARQUET, index=False)
    print("Done.")

# ------------------------------------------------------------------------------
# STEP 2: ADD RESSOURCENSTANDORT COORDINATES (fz_lat/fz_lon)
# ------------------------------------------------------------------------------
def add_fz_coordinates():
    print("Adding 'Standort FZ' coordinates...")
    df = pd.read_parquet(COMBINED_EINSATZ_PARQUET)

    # Mapping dictionary
    fz_coordinates = {
        "Rettungswache Mitte": (51.35738328512606, 7.463011574675159),
        "Rettungswache Ost": (51.37354069735159, 7.546357855336432),
        "Rettungswache Ost -NEF-": (51.37354069735159, 7.546357855336432),
        "Rettungswache Ev.Krhs. Haspe": (51.347649595944866, 7.410824948225933),
        "Rettungswache Vorhalle": (51.3857053741779, 7.4349754931565855),
        "Rettungswache Allgemeine Krankenhaus Hagen": (51.35392279941694, 7.466584396788664),
        "Rettungswache HaTüWe": (51.35146472534295, 7.432385861915431),
        "Rettungswache St. Johannes Hospital": (51.39586711277197, 7.473996499279848),
        "Rettungswache Dahl": (51.3055443277608, 7.53220077856884)
    }
    # Separate lat/lon mappings
    fz_lat_mapping = {key: coords[0] for key, coords in fz_coordinates.items()}
    fz_lon_mapping = {key: coords[1] for key, coords in fz_coordinates.items()}

    # Add columns
    df['FZ_lat'] = df['Standort FZ'].map(fz_lat_mapping)
    df['FZ_lon'] = df['Standort FZ'].map(fz_lon_mapping)

    # Save updated
    df.to_parquet(COMBINED_EINSATZ_PARQUET, index=False)
    print("Done.")

# ------------------------------------------------------------------------------
# STEP 3: CALCULATE ORS DISTANCE & DURATION, SAVE INTERMEDIATE
# ------------------------------------------------------------------------------
def calculate_ors_distances():
    print("Calculating ORS distances/durations (intermediate save)...")
    df = pd.read_parquet(COMBINED_EINSATZ_PARQUET)

    # Extract unique combos
    unique_combos = df[['FZ_lon', 'FZ_lat', 'EINSATZORT_lon', 'EINSATZORT_lat']].drop_duplicates()
    unique_combos_clean = unique_combos.dropna(subset=['FZ_lon', 'FZ_lat', 'EINSATZORT_lon', 'EINSATZORT_lat'])

    print(f"Unique coordinate combinations: {len(unique_combos_clean)}")
    # ORS client
    client = openrouteservice.Client(base_url='http://localhost:8080/ors')
    
    def calc_route(row):
        start_coords = (row['FZ_lon'], row['FZ_lat'])
        end_coords   = (row['EINSATZORT_lon'], row['EINSATZORT_lat'])
        try:
            route = client.directions(
                coordinates=[start_coords, end_coords],
                profile='driving-car',
                format='geojson'
            )
            dist_km = route['features'][0]['properties']['segments'][0]['distance'] / 1000
            dur_min = route['features'][0]['properties']['segments'][0]['duration'] / 60
            return pd.Series([dist_km, dur_min])
        except:
            return pd.Series([None, None])

    unique_combos_clean[['distance','duration']] = unique_combos_clean.apply(calc_route, axis=1)

    # Merge results back
    df = df.merge(
        unique_combos_clean,
        on=['FZ_lon','FZ_lat','EINSATZORT_lon','EINSATZORT_lat'],
        how='left'
    )
    df.to_parquet(COMBINED_EINSATZ_ORS_PARQUET, index=False)
    print(f"Saved intermediate results to {COMBINED_EINSATZ_ORS_PARQUET}.")

# ------------------------------------------------------------------------------
# STEP 4: MERGE WITH WEATHER (ERA5) DATA
# ------------------------------------------------------------------------------
def merge_with_weather():
    print("Merging Einsatz+ORS with ERA5 weather data...")
    df = pd.read_parquet(COMBINED_EINSATZ_ORS_PARQUET)

    # We floor ALARMZEIT to hour for merging
    if "ALARMZEIT" not in df.columns:
        raise ValueError("No 'ALARMZEIT' column found. Cannot merge with weather.")
    df["ALARMZEIT_floor"] = df["ALARMZEIT"].dt.floor("h")

    # Load combined ERA5 data
    wdf = pd.read_csv(ERA5_COMBINED_FILE, parse_dates=["valid_time"])
    wdf["valid_time_floor"] = wdf["valid_time"].dt.floor("h")

    # Sort for merge_asof or direct merge on same hour
    # We'll do a plain merge on the floored hour
    merged = pd.merge(
        df,
        wdf,
        left_on="ALARMZEIT_floor",
        right_on="valid_time_floor",
        how="left"
    )
    merged.to_parquet(COMBINED_EINSATZ_WEATHER_PARQUET, index=False)
    print(f"Saved to {COMBINED_EINSATZ_WEATHER_PARQUET}.")

# ------------------------------------------------------------------------------
# STEP 5: PARSE FEIERTAGE & FERIEN, MERGE
# ------------------------------------------------------------------------------
def parse_ics(file_path):
    """
    Parse ICS for single-day events (e.g. Feiertage).
    Ensures 'start' and 'end' are Pandas Timestamps for consistent comparisons.
    """
    with open(file_path, "rb") as f:
        cal = Calendar.from_ical(f.read())

    events = []
    for component in cal.walk():
        if component.name == "VEVENT":
            event_name = str(component.get("summary"))

            start_dt = component.get("dtstart").dt  # might be date or datetime
            end_dt   = component.get("dtend").dt or start_dt

            # Convert to Pandas Timestamps
            start_ts = pd.to_datetime(start_dt)  # safe if it's date or datetime
            end_ts   = pd.to_datetime(end_dt)

            events.append({"event": event_name, "start": start_ts, "end": end_ts})
    return pd.DataFrame(events)


def parse_ferien_ics(file_path):
    """
    Parse ICS for multi-day events (Ferien). The 'end' date is exclusive.
    Also ensures 'start' and 'end' are Pandas Timestamps.
    """
    df = parse_ics(file_path)  # re-use parse_ics logic
    # For ferien, each row is a range [start, end), already as Timestamps
    return df


def build_feiertage_ferien():
    """
    1. Parse feiertage.ics into Timestamps => 'start','end'.
    2. Parse ferien.ics similarly.
    3. Construct daily date range from 2018-01-01 to 2024-12-31 (dtype datetime64).
    4. Mark each row with Feiertag / Ferien columns.
    """
    print("Creating Feiertage-Ferien dataset...")

    # 1. Feiertage
    feiertage_df = parse_ics("data/raw/feiertage.ics")
    # convert each 'start' to Timestamp in parse_ics => done above

    # 2. Ferien
    ferien_df = parse_ferien_ics("data/raw/ferien.ics")

    # 3. A daily DataFrame from 2018 to 2024
    all_dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq="D")
    df_dates = pd.DataFrame({"date": all_dates})  # dtype=datetime64[ns]

    # Mark Feiertage
    # feiertage_df["start"] is now pd.Timestamp, so create a set of Timestamps
    feiertage_dates = set(feiertage_df["start"])
    df_dates["Feiertag"] = df_dates["date"].isin(feiertage_dates)

    # Mark Ferien
    df_dates["Ferien"] = False
    for _, row in ferien_df.iterrows():
        start_ = row["start"]  # a Timestamp
        end_   = row["end"]    # a Timestamp, exclusive
        mask   = (df_dates["date"] >= start_) & (df_dates["date"] < end_)
        df_dates.loc[mask, "Ferien"] = True

    # Save
    out_path = FEIERTAGE_FERIEN_PARQUET
    df_dates.to_parquet(out_path, index=False)
    print(f"Saved Feiertage-Ferien to {out_path}")

def merge_with_feiertage_ferien():
    print("Merging Einsatz/Weather data with Feiertage-Ferien info...")
    df = pd.read_parquet(COMBINED_EINSATZ_WEATHER_PARQUET)

    # We'll floor ALARMZEIT to day
    df["date"] = df["ALARMZEIT"].dt.floor("D")
    dates_df = pd.read_parquet(FEIERTAGE_FERIEN_PARQUET)
    merged = pd.merge(
        df,
        dates_df,
        on="date",
        how="left"
    )
    merged.to_parquet(COMBINED_EINSATZ_WEATHER_DATES_PARQUET, index=False)
    print(f"Saved to {COMBINED_EINSATZ_WEATHER_DATES_PARQUET}.")

# ------------------------------------------------------------------------------
# STEP 6: FINAL TRANSFORMATIONS & FILTERING
# ------------------------------------------------------------------------------
def final_transformations():
    print("Loading merged data for final transformations...")
    df = pd.read_parquet(COMBINED_EINSATZ_WEATHER_DATES_PARQUET)

    # Extract "Einsatzniveau" from last digit in EINSATZSTICHWORT / EINSATZSTICHWORT_1
    df['Einsatzniveau'] = df['EINSATZSTICHWORT'].str.extract(r' (\d)$')
    df['Einsatzniveau'] = df['Einsatzniveau'].astype('Int64')  # can hold NA

    df['Einsatzniveau_1'] = df['EINSATZSTICHWORT_1'].str.extract(r' (\d)$')
    df['Einsatzniveau_1'] = df['Einsatzniveau_1'].astype('Int64')  # can hold NA
    
    # Compute Dispositionszeit (Status 4 - MELDUNGSEINGANG) in minutes
    if 'ALARMZEIT' in df.columns and 'MELDUNGSEINGANG' in df.columns:
        df['Dispositionszeit'] = (df['ALARMZEIT'] - df['MELDUNGSEINGANG']).dt.total_seconds() / 60.0
        
    # Compute Ausrückzeit (Status 3 - ALARMZEIT) in minutes
    if 'ALARMZEIT' in df.columns and 'MELDUNGSEINGANG' in df.columns:
        df['Ausrückzeit'] = (df['Status 3'] - df['ALARMZEIT']).dt.total_seconds() / 60.0

    # Compute Eintreffzeit (Status 4 - MELDUNGSEINGANG) in minutes
    if 'Status 4' in df.columns and 'MELDUNGSEINGANG' in df.columns:
        df['Eintreffzeit'] = (df['Status 4'] - df['MELDUNGSEINGANG']).dt.total_seconds() / 60.0

    # Compute Fahrzeit (Status 4 - Status 3) in minutes
    if 'Status 4' in df.columns and 'Status 3' in df.columns:
        df['Fahrzeit'] = (df['Status 4'] - df['Status 3']).dt.total_seconds() / 60.0

    # Add Wochentag, Monat, Uhrzeit
    df['Wochentag'] = df['MELDUNGSEINGANG'].dt.strftime('%A').astype('category')
    df['Monat']     = df['MELDUNGSEINGANG'].dt.strftime('%B').astype('category')
    df['Uhrzeit']   = df['MELDUNGSEINGANG'].dt.hour \
                      + df['MELDUNGSEINGANG'].dt.minute / 60.0 \
                      + df['MELDUNGSEINGANG'].dt.second / 3600.0
    df['Jahr'] = df['MELDUNGSEINGANG'].dt.year.astype('Int64')

    # Normalize one "Standort FZ" label
    df["Standort FZ"] = df["Standort FZ"].astype(str).replace(
        {"Rettungswache Ost -NEF-": "Rettungswache Ost"}
    ).astype("category")

    # Convert Feiertag/Ferien to booleans
    if "Feiertag" in df.columns:
        df["Feiertag"] = df["Feiertag"].astype('boolean')
    if "Ferien" in df.columns:
        df["Ferien"] = df["Ferien"].astype('boolean')

    # Save intermediate "all-data"
    df.to_parquet(ALL_DATA_PARQUET, index=False)
    print(f"Saved full dataset to {ALL_DATA_PARQUET} for final filtering.")

def final_filter_and_save():
    print("Filtering data and saving final 'selected-data.parquet'...")
    df = pd.read_parquet(ALL_DATA_PARQUET)

    # 1) Drop unneeded columns
    columns_to_drop = [
        "EINSATZNUMMER", "HAUPTEINSATZNUMMER",
        "MELDUNGSEINGANG","ALARMZEIT","Status 3","Status 4","Status 7","Status 8","Status 1","Status 2",
        "EINSATZSTICHWORT_1","EINSATZSTICHWORT","EINSATZORT_X","EINSATZORT_Y","ZIELORT","ZIELORT_X","ZIELORT_Y",
        "Bezeichnung Zielort","FUNKRUFNAME","Einsatzmittel-Typ","FZ_lat","FZ_lon",
        "ALARMZEIT_floor","valid_time","valid_time_floor","date","ZIELORT_ADRESSE", "total_precipitation", "snowfall"
    ]
    df.drop(columns=columns_to_drop, errors="ignore", inplace=True)
    
    # 2) Filter rows: SONDERSIGNAL == True, and (Einsatzniveau >=1 or Einsatzniveau_1 >=1)
    print("Initial rows:", len(df))
    df = df[
        (df['SONDERSIGNAL'] == True) &
        (
            (df['Einsatzniveau'].notna() & (df['Einsatzniveau'] >= 1)) |
            (df['Einsatzniveau_1'].notna() & (df['Einsatzniveau_1'] >= 1))
        )
    ]
    print("After SONDERSIGNAL/Einsatzniveau filter:", len(df))

    # Drop the columns for SONDERSIGNAL/Einsatzniveau if no longer needed
    df.drop(columns=['Einsatzniveau','Einsatzniveau_1','SONDERSIGNAL'], errors="ignore", inplace=True)

    # 3) Drop rows with NA
    df.dropna(inplace=True)
    print("After dropping NA rows:", len(df))

    # 4) Keep only Eintreffzeit >= 0 and Fahrzeit >= 0
    df = df[(df['Eintreffzeit'] >= 0) & (df['Fahrzeit'] >= 0)]
    print("After non-negative time filtering:", len(df))

    # 5) Remove outliers in Eintreffzeit (1.5 * IQR rule)
    Q1 = df['Eintreffzeit'].quantile(0.25)
    Q3 = df['Eintreffzeit'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df = df[df['Eintreffzeit'] <= upper_bound]
    print("After outlier removal:", len(df))

    # 6) Save final
    df.to_parquet(FINAL_DATA_PARQUET, index=False)
    print(f"Final data saved to {FINAL_DATA_PARQUET} with shape {df.shape}.")

# ------------------------------------------------------------------------------
# RUN EVERYTHING
# ------------------------------------------------------------------------------
def main():
    # 1) Combine raw data
    #combine_einsatzdaten()
    # 2) Add Rettungswache coordinates
    #add_fz_coordinates()
    # 3) Calculate ORS distances => intermediate
    #calculate_ors_distances()
    # 4) Merge with weather
    #merge_with_weather()
    # 5) Parse & build Feiertage/Ferien, then merge
    #build_feiertage_ferien()
    #merge_with_feiertage_ferien()
    # 6) Final transformations
    final_transformations()
    # 7) Final filter & save
    final_filter_and_save()
    print("Preprocessing pipeline complete.")

if __name__ == "__main__":
    main()
