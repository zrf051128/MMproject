"""Quick smoke check for objective ablation and lambda_AE sweep."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_ablation_experiments import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--image",
        "cameraman",
        "--mask",
        "irregular",
        "--ae_source",
        "auto",
        "--ae_epochs",
        "20",
        "--steps",
        "100",
        "--lambda_sweep",
        "1e-3,1e-2",
    ]
    main()
