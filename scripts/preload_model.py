from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from handler import get_ocr


def main() -> None:
    get_ocr()
    print("Nemotron OCR v2 loaded")


if __name__ == "__main__":
    main()
