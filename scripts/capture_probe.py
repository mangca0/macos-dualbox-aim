import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from macos_dualbox_aim.core.capture_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
