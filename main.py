## SNAPSHOTS EXTRACTION 

# Racine Opendata officielle (fourni par Vélib' Métropole)
VELIB_BASE = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole"

# Dossiers de sortie
OUT_DIR = Path("data/raw/velib")
OUT_DIR_SNAP = OUT_DIR / "snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_SNAP.mkdir(parents=True, exist_ok=True)

STATION_INFO_CSV = OUT_DIR / "station_information.csv"
SYSTEM_INFO_JSON = OUT_DIR / "system_information.json"

print("Sortie:", OUT_DIR.resolve())
