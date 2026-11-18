"""Make the existing standalone scripts importable from the package entry points."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from learn import DATA, digest
