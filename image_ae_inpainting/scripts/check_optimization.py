"""Quick smoke check for AE-only and TV+AE optimization."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_main_experiments import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--images",
        "cameraman",
        "--masks",
        "irregular",
        "--methods",
        "ae_only,tv_ae",
        "--ae_source",
        "auto",
        "--ae_epochs",
        "20",
        "--max_cases",
        "1",
        "--steps",
        "100",
    ]
    main()
