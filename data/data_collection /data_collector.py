import os
import pandas as pd
import requests
from io import StringIO
from mp_api.client import MPRester

# Set output directory
OUTPUT_DIR = "/Users/eddypoon/Desktop/electrolyte_loop/data/raw/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set your new Materials Project API key here
MP_API_KEY = "HPuAc6UU9ByceOAeEpidWM4RXFD4CBL6"

def fetch_materials_project_data():
    print("Fetching data from Materials Project (new API)...")
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            elements=["Li", "F", "P"],
            num_elements=(2, 5),
            deprecated=False,
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "energy_above_hull",
                "density"
            ]
        )
        df = pd.DataFrame([d.dict() for d in docs])
        output_path = os.path.join(OUTPUT_DIR, "materials_project.csv")
        df.to_csv(output_path, index=False)
        print(f"✔ Saved {len(df)} entries to {output_path}.")

def fetch_nomad_data():
    print("Fetching data from NOMAD...")
    # Placeholder — you can add actual logic later
    df = pd.DataFrame(columns=["material_id", "property_1", "property_2"])
    output_path = os.path.join(OUTPUT_DIR, "nomad_data.csv")
    df.to_csv(output_path, index=False)
    print(f"✔ Saved NOMAD data with 0 entries to {output_path}.")

def fetch_freesolv_data():
    print("Fetching FreeSolv data...")
    url = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"

    try:
        df = pd.read_csv(url, sep="\t", on_bad_lines='skip')
        output_path = os.path.join(OUTPUT_DIR, "freesolv.csv")
        df.to_csv(output_path, index=False)
        print(f"✔ Saved FreeSolv data to {output_path}.")
    except Exception as e:
        print(f"✘ Failed to fetch FreeSolv data: {e}")
        raise

if __name__ == "__main__":
    fetch_materials_project_data()
    fetch_nomad_data()
    fetch_freesolv_data()