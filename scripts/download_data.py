"""Download the Bosch Production Line Performance dataset via kagglehub.

Requires:
1. `pip install kagglehub`
2. Kaggle credentials — either at ~/.kaggle/kaggle.json (chmod 600)
   or as env vars KAGGLE_USERNAME + KAGGLE_KEY.
3. You must have accepted the competition rules ONCE at:
   https://www.kaggle.com/c/bosch-production-line-performance/rules

After download, the script also stream-reencodes CSVs -> Parquet for fast chunked reads.
"""

from __future__ import annotations

import logging
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"

sys.path.insert(0, str(REPO_ROOT / "src"))
from data_loader import reencode_to_parquet  # noqa: E402


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def check_kaggle_auth() -> None:
    import os
    kaggle_dir = Path.home() / ".kaggle"
    legacy_json = kaggle_dir / "kaggle.json"          # old format: {"username":"...","key":"..."}
    new_token = kaggle_dir / "access_token"           # new format: single "KGAT_..." string
    env_new = "KAGGLE_API_TOKEN" in os.environ
    env_old = "KAGGLE_KEY" in os.environ and "KAGGLE_USERNAME" in os.environ
    if not (legacy_json.exists() or new_token.exists() or env_new or env_old):
        logger.error(
            "Kaggle credentials missing. Provide ONE of:\n"
            "  a) ~/.kaggle/access_token containing your KGAT_... token (chmod 600), OR\n"
            "  b) ~/.kaggle/kaggle.json with {\"username\":..., \"key\":...} (chmod 600), OR\n"
            "  c) env var KAGGLE_API_TOKEN=KGAT_..., OR\n"
            "  d) env vars KAGGLE_USERNAME and KAGGLE_KEY."
        )
        sys.exit(1)
    if new_token.exists() and "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = new_token.read_text().strip()
        logger.info("loaded KAGGLE_API_TOKEN from %s", new_token)


def download_via_kagglehub(competition: str) -> Path:
    import kagglehub

    logger.info("downloading competition '%s' via kagglehub (this may take a while, ~14 GB)", competition)
    path = kagglehub.competition_download(competition)
    logger.info("kagglehub returned path: %s", path)
    return Path(path)


def copy_files_to_project(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        target = dst_dir / f.name
        if target.exists():
            logger.info("  %s already present, skipping", f.name)
            continue
        logger.info("  copying %s -> %s", f.name, target)
        shutil.copy2(f, target)


def unzip_all_archives(dst_dir: Path) -> None:
    """kagglehub returns .csv.zip files — unzip them in place and remove the archives."""
    for zip_path in sorted(dst_dir.glob("*.zip")):
        logger.info("  unzipping %s", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dst_dir)
        zip_path.unlink()


def reencode_all_csvs(raw_dir: Path, parquet_dir: Path, chunksize: int) -> None:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for csv in sorted(raw_dir.glob("*.csv")):
        pq_path = parquet_dir / (csv.stem + ".parquet")
        if pq_path.exists():
            logger.info("%s already exists, skipping reencode", pq_path.name)
            continue
        reencode_to_parquet(csv, pq_path, chunksize=chunksize)


if __name__ == "__main__":
    cfg = load_config()
    check_kaggle_auth()

    kagglehub_path = download_via_kagglehub(cfg["data"]["kaggle_competition"])

    raw_dir = REPO_ROOT / cfg["data"]["raw_dir"]
    copy_files_to_project(kagglehub_path, raw_dir)
    unzip_all_archives(raw_dir)

    parquet_dir = REPO_ROOT / cfg["data"]["parquet_dir"]
    reencode_all_csvs(raw_dir, parquet_dir, chunksize=cfg["loading"]["chunksize"])

    logger.info("all done. Raw at %s, Parquet at %s", raw_dir, parquet_dir)
