from pathlib import Path
import sys


def ensure_mage_flow_importable(source_path: str | None = None) -> None:
    root = Path(source_path or Path(__file__).resolve().parents[4] / "third_party" / "Mage").expanduser().resolve()
    if not (root / "mage_flow").is_dir():
        raise FileNotFoundError(f"Mage-Flow source tree not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
