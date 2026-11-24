"""Repository paths shared by commands and applications."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = ROOT / "resources" / "openarm"
ROBOT_XML = ROBOT_DIR / "arm.xml"
