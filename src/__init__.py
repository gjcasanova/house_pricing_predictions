"""Contains data, scripts and other tools about this project."""

import sys
from pathlib import Path

current_path = Path(__file__).parent.resolve()
sys.path.append(str(current_path))
