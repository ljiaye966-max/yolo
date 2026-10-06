import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings
from services.dataset_service import convert_construction_ppe

if __name__ == "__main__": print(convert_construction_ppe(get_settings()))
