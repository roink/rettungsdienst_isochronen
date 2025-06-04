# ors.py
import os
import subprocess
import requests
import time

def check_ors_status():
    try:
        response = requests.get("http://localhost:8080/ors/v2/status")
        if response.status_code == 200:
            print("OpenRouteService is running.")
            return True
    except requests.ConnectionError:
        pass
    print("OpenRouteService is not running.")
    return False

def start_docker_compose():
    try:
        subprocess.run(["docker", "compose", "up", "-d"], check=True)
        print("Docker Compose started successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Failed to start Docker Compose.")
        return False

def download_osm_map_data(base_path):
    osm_map_url = "http://download.geofabrik.de/europe/germany/nordrhein-westfalen/arnsberg-regbez-latest.osm.pbf"
    osm_map_path = os.path.join(base_path, "ors-docker/files/arnsberg-regbez-latest.osm.pbf")

    # Ensure the directory exists
    osm_map_dir = os.path.dirname(osm_map_path)
    os.makedirs(osm_map_dir, exist_ok=True)
    
    if os.path.exists(osm_map_path):
        print(f"OSM map data already exists at {osm_map_path}.")
        return
    
    try:
        response = requests.get(osm_map_url)
        response.raise_for_status()
        with open(osm_map_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded OSM map data to {osm_map_path}.")
    except requests.RequestException as e:
        print(f"Failed to download OSM map data: {e}")

def start(restart=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ors_dir = os.path.join(os.path.dirname(current_dir), "ors")
    
    if restart:
        os.chdir(ors_dir)
        subprocess.run(["docker", "compose", "up", "-d"], check=True)
        start_docker_compose
    
    # Check if ORS is running
    if check_ors_status():
        return
    
    # Try to start Docker Compose from the "ors" directory
 

    if os.path.exists(ors_dir):
        os.chdir(ors_dir)
        # Download the OSM map data if necessary
        download_osm_map_data(ors_dir)

        # Attempt to start Docker Compose
        if start_docker_compose():
            time.sleep(5)  # Give some time for ORS to start
            check_ors_status()
    else:
        print(f"The expected 'ors' directory does not exist at {ors_dir}.")

if __name__ == "__main__":
    start()
