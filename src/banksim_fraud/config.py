from pathlib import Path

# directory that holds the saved boosters + threshold.json
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ensemble settings
NUM_MODELS        = 5
UNDERSAMPLE_RATIO = 0.05          # 1 fraud : 50 normal

# threshold chosen on PR curve
DEFAULT_THRESHOLD = 0.306
