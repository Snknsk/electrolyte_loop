#!/usr/bin/env python3
"""
data_collector.py
-----------------
Fetches three public datasets relevant to electrolyte design and writes them
as CSV into ./raw/ (folder created next to this script):

    1) Materials Project: Li/Na/K/Mg/Ca/Zn/Al/O/F/P‐containing stable compounds
    2) FreeSolv: experimental hydration free energies for ~650 organic molecules
    3) PubChem CID↔Formula table so we can later map IUPAC / molecular formulas

Each fetch is idempotent: if the CSV already exists *and* is newer than
24 hours we skip the download so repeated runs are fast.
"""

from __future__ import annotations
import csv, gzip, io, os, sys, time, requests, subprocess, datetime as dt
from pathlib import Path
from typing import List, Any
import pandas as pd

# ────────── CONFIG ──────────
MP_API_KEY = os.getenv("MP_API_KEY", "HPuAc6UU9ByceOAeEpidWM4RXFD4CBL6")
FIELDS = [
    "material_id", "formula_pretty",
    "formation_energy_per_atom", "energy_above_hull",
    "band_gap", "density", "volume", "nsites", "elements", "structure"
]
CHUNK_SIZE = 200
MAX_RETRIES = 3
CACHE_AGE_HR = 24
RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_DIR.mkdir(exist_ok=True, parents=True)

# Proxy ping-through
for k in ("HTTP_PROXY", "HTTPS_PROXY"):
    if os.getenv(k):
        print(f"🌐 Using proxy from ${k}")

def is_fresh(p: Path) -> bool:
    return p.exists() and (dt.datetime.now() - dt.datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() < CACHE_AGE_HR * 3600

def save(df: pd.DataFrame, fname: str) -> None:
    out = RAW_DIR / fname
    df.to_csv(out, index=False)
    size_kb = out.stat().st_size / 1024
    print(f"✅ {out.name:<25} {len(df):5,} rows, {size_kb:7.1f} KB")

def fetch_materials_datasets() -> None:
    try:
        from mp_api.client import MPRester
    except:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "mp-api", "pymatgen<2025"], check=False)
        from mp_api.client import MPRester

    # –– CERAMIC
    ceramic_path = RAW_DIR / "ceramic_electrolytes.csv"
    if not is_fresh(ceramic_path):
        rows, seen = [], set()
        print("🔌 Querying Materials Project for ceramic electrolytes …")
        with MPRester(MP_API_KEY) as mpr:
            for cation in ["Li","Na","K","Mg","Ca","Zn","Al"]:
                for anions in [["O"],["S"],["P"],["F"]]:
                    for attempt in range(MAX_RETRIES):
                        try:
                            docs = mpr.materials.summary.search(
                                elements=[cation] + anions,
                                is_stable=True,
                                fields=FIELDS,
                                chunk_size=CHUNK_SIZE)
                            break
                        except Exception:
                            time.sleep(5)
                    for d in docs:
                        if d.material_id in seen: continue
                        seen.add(d.material_id)
                        dd = d.dict()
                        row = {k: dd.get(k) for k in FIELDS}
                        row["elements"] = "|".join(dd.get("elements", []))
                        row["dataset_type"] = "ceramic"
                        rows.append(row)
        df = pd.DataFrame(rows)
        numeric = [c for c in df.columns if df[c].dtype == object and c not in ("material_id","formula_pretty","elements","structure","dataset_type")]
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
        save(df, "ceramic_electrolytes.csv")
    else:
        print(f"↪︎ Skipping ceramics – fresh cache present.")

    # –– GLASS
    glass_path = RAW_DIR / "glass_electrolytes.csv"
    if not is_fresh(glass_path):
        rows, seen = [], set()
        print("🔌 Querying Materials Project for glass electrolytes …")
        with MPRester(MP_API_KEY) as mpr:
            for cation in ["Li","Na","K","Mg","Ca","Zn","Al"]:
                for former in ["P","Si","B"]:
                    for attempt in range(MAX_RETRIES):
                        try:
                            docs = mpr.materials.summary.search(
                                elements=[cation, former, "O"],
                                is_stable=True,
                                fields=FIELDS,
                                chunk_size=CHUNK_SIZE)
                            break
                        except:
                            time.sleep(5)
                    for d in docs:
                        if d.material_id in seen: continue
                        seen.add(d.material_id)
                        dd = d.dict()
                        row = {k: dd.get(k) for k in FIELDS}
                        row["elements"] = "|".join(dd.get("elements", []))
                        row["dataset_type"] = "glass"
                        rows.append(row)
        df = pd.DataFrame(rows)
        numeric = [c for c in df.columns if df[c].dtype == object and c not in ("material_id","formula_pretty","elements","structure","dataset_type")]
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
        save(df, "glass_electrolytes.csv")
    else:
        print(f"↪︎ Skipping glasses – fresh cache present.")

    # –– POLYMER (from curated PubChem monomers)
    polymer_path = RAW_DIR / "polymer_electrolytes.csv"
    if not is_fresh(polymer_path):
        print("📥 Downloading curated polymer monomer dataset from PubChem...")
        PI1M_URL = "https://raw.githubusercontent.com/OpenPolymerData/monomers/main/pubchem_polymer_monomers.csv"
        try:
            with requests.get(PI1M_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(polymer_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            df_poly = pd.read_csv(polymer_path)
            save(df_poly, "polymer_electrolytes.csv")
        except Exception as e:
            print(f"❌ Failed to download PI1M dataset: {e}")
    else:
        print(f"↪︎ Skipping polymers – fresh cache present.")

def fetch_freesolv() -> None:
    csv_path = RAW_DIR / "freesolv.csv"
    if is_fresh(csv_path):
        print(f"↪︎ Skipping FreeSolv – cached file is fresh.")
        return

    url = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"
    print("🌊 Downloading FreeSolv …")
    for _ in range(MAX_RETRIES):
        try:
            txt = requests.get(url, timeout=30).text
            df = pd.read_csv(io.StringIO(txt), sep="\t", names=["composition"], comment="#")
            save(df, "freesolv.csv")
            return
        except Exception as e:
            print("⚠️  retry FreeSolv:", e)
            time.sleep(5)
    print("❌ FreeSolv download failed after retries")

def fetch_pubchem_cid_formula() -> None:
    csv_path = RAW_DIR / "pubchem_cid_formula.csv"
    if is_fresh(csv_path):
        print(f"↪︎ Skipping PubChem – cached file is fresh.")
        return

    base = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras"
    fname = "CID-Formula.gz"
    url = f"{base}/{fname}"
    print("⬇️ Downloading PubChem CID↔Formula table …")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        with gzip.open(buf, "rt") as gz:
            reader = csv.reader(gz, delimiter="\t")
            rows = [{"cid": int(cid), "formula": formula} for cid, formula in reader]
        save(pd.DataFrame(rows), "pubchem_cid_formula.csv")
    except Exception as e:
        print("❌ PubChem mapping failed:", e)

def main() -> None:
    t0 = time.time()
    fetch_materials_datasets()
    fetch_freesolv()
    fetch_pubchem_cid_formula()

    print("\n📄 CSV files in ./raw:")
    for f in RAW_DIR.glob("*.csv"):
        size_kb = f.stat().st_size / 1024
        age_hr = (dt.datetime.now() - dt.datetime.fromtimestamp(f.stat().st_mtime)).total_seconds()/3600
        print(f" • {f.name:<25} {size_kb:7.1f} KB (age: {age_hr:5.1f} h)")
    print(f"⏱️ Total runtime: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()