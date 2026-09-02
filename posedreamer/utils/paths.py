from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
WEIGHTS_DIR = PACKAGE_ROOT / "weights"
