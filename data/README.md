# Data directory

Raw Bosch Production Line Performance data lives here after running:

```bash
python scripts/download_data.py
```

Downloaded contents (~14 GB uncompressed):

- `bosch-production-line-performance/train_numeric.csv`  (~2.1 GB)
- `bosch-production-line-performance/train_categorical.csv`  (~2.1 GB)
- `bosch-production-line-performance/train_date.csv`  (~2.9 GB)
- `bosch-production-line-performance/test_numeric.csv`  (~2.1 GB)
- `bosch-production-line-performance/test_date.csv`  (~2.9 GB)

The download script reencodes these to Parquet in `data/processed/` for fast chunked reads.

**Nothing in this directory is committed to git** — see `.gitignore`.
