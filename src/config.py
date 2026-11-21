"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 10

for _d in (DATA_PROCESSED, FIG_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
