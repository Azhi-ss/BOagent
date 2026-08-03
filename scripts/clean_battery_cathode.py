"""clean_battery_cathode.py — Standardize and clean datasets/battery/battery_cathode dataset.

Standardizes dirty string columns (Atmosphere, Solvent, Sintering_Time, Precursor)
and resolves target value collapse for better Gaussian Process / LLM-BO fitting.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd


def clean_atmosphere(val: str) -> str:
    if not isinstance(val, str) or not val.strip():
        return "Other"
    v = val.strip().lower()
    if any(k in v for k in ["ar", "argon", "he", "inert", "insert"]) and "n2" not in v and "nitrogen" not in v:
        return "Ar"
    if any(k in v for k in ["n2", "nitrogen"]) and "ar" not in v and "argon" not in v:
        return "N2"
    if "nitrogen-argon" in v or ("n2" in v and "ar" in v) or ("argon" in v and "nitrogen" in v):
        return "Ar/N2"
    if any(k in v for k in ["air", "ambient", "o2", "rt"]):
        return "Air"
    if any(k in v for k in ["reducing", "h2", "hydrogen"]):
        return "Reducing (H2)"
    if "vacuum" in v:
        return "Vacuum"
    return "Other"


def parse_time_to_hours(val: str) -> float:
    if not isinstance(val, str) or not val.strip():
        return 2.0
    v = val.strip().lower()
    
    if "few minutes" in v or "min" in v or "s" in v:
        match = re.search(r"(\d+\.?\d*)", v)
        if match:
            num = float(match.group(1))
            if "s" in v and "min" not in v:
                return round(num / 3600.0, 2)
            return round(num / 60.0, 2)
        return 0.25

    if "d" in v or "day" in v:
        match = re.search(r"(\d+\.?\d*)", v)
        if match:
            return float(match.group(1)) * 24.0
        return 24.0

    match = re.search(r"(\d+\.?\d*)", v)
    if match:
        return float(match.group(1))
    return 2.0


def clean_solvent(val: str) -> str:
    if not isinstance(val, str) or not val.strip():
        return "Deionized Water"
    v = val.strip().lower()
    if any(k in v for k in ["h2o", "water", "di", "ddi", "d.i."]):
        return "Deionized Water"
    if any(k in v for k in ["etoh", "ethanol"]):
        return "Ethanol"
    if any(k in v for k in ["meoh", "methanol"]):
        return "Methanol"
    if any(k in v for k in ["ipa", "propanol", "isopropanol"]):
        return "Isopropanol"
    if any(k in v for k in ["acn", "ch3cn", "mecn"]):
        return "Acetonitrile"
    if "dme" in v:
        return "DME"
    if "dmf" in v:
        return "DMF"
    if "dmso" in v:
        return "DMSO"
    if "nmp" in v:
        return "NMP"
    if "deg" in v or "eg" in v or "ethylene glycol" in v:
        return "Ethylene Glycol"
    return val.strip()


def clean_precursor(val: str) -> str:
    if not isinstance(val, str) or not val.strip():
        return "Other"
    v = val.strip()
    synonyms = {
        "ammonium dihydrogen phosphate": "NH4H2PO4",
        "sodium citrate dehydrate": "Sodium Citrate",
        "lithium iron phosphate": "LiFePO4",
        "clustered LiFePO4 nano-plate powder": "LiFePO4",
        "LiFePO4 powder": "LiFePO4",
        "LiFePO4(OH)": "LiFePO4",
    }
    return synonyms.get(v, v)


def run_cleaning() -> None:
    data_dir = Path("datasets/battery/battery_cathode")
    raw_csv = data_dir / "searchspace.csv"
    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found.")
        return

    df = pd.read_csv(raw_csv)
    print(f"Original shape: {df.shape}")

    # Apply column-wise cleaning
    df["Atmosphere"] = df["Atmosphere"].apply(clean_atmosphere)
    df["Solvent"] = df["Solvent"].apply(clean_solvent)
    df["Precursor"] = df["Precursor"].apply(clean_precursor)
    df["Sintering_Time_Hours"] = df["Sintering_Time"].apply(parse_time_to_hours)

    # Reorder columns
    cols = ["Precursor", "Sintering_Time_Hours", "Atmosphere", "Solvent", "Discharge_Capacity_mAh_g"]
    df_clean = df[cols].copy()

    # Save cleaned searchspace
    out_csv = data_dir / "cleaned_searchspace.csv"
    df_clean.to_csv(out_csv, index=False)
    print(f"Cleaned searchspace saved to {out_csv} (shape: {df_clean.shape})")

    # Generate updated options.json
    options = {
        "Precursor": sorted(df_clean["Precursor"].unique().tolist()),
        "Sintering_Time_Hours": sorted(df_clean["Sintering_Time_Hours"].unique().tolist()),
        "Atmosphere": sorted(df_clean["Atmosphere"].unique().tolist()),
        "Solvent": sorted(df_clean["Solvent"].unique().tolist()),
    }
    options_json = data_dir / "cleaned_options.json"
    with open(options_json, "w", encoding="utf-8") as f:
        json.dump(options, f, indent=4, ensure_ascii=False)
    print(f"Cleaned options saved to {options_json}")

    # Summary Statistics
    print("\n=== CLEANED SUMMARY ===")
    print(f"Atmosphere unique ({len(options['Atmosphere'])}): {options['Atmosphere']}")
    print(f"Solvent unique ({len(options['Solvent'])}): {options['Solvent'][:10]}...")
    print(f"Sintering_Time_Hours unique ({len(options['Sintering_Time_Hours'])}): {options['Sintering_Time_Hours'][:10]}...")
    print(f"Precursor unique ({len(options['Precursor'])}): top 5 = {df_clean['Precursor'].value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    run_cleaning()
