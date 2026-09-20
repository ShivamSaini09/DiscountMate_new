# Catalogue AU Coles OCR scraper

This command-line scraper accepts either one Catalogue AU URL or a CSV/text
dataset of URLs. It reads the site's public catalogue metadata, downloads the
catalogue page images, runs local Tesseract OCR, and produces CSV and JSON
datasets.

It extracts:

- retailer and region
- catalogue title, start date, and end date
- page number and source links
- product name and pack size
- special price, regular/actual price, and save amount
- calculated discount percentage and promotion type
- unit-price text where visible
- OCR confidence, `needs_review`, and a review reason
- price provenance, independent-evidence count, and verification status

## Requirements

- Python 3.10+
- Tesseract OCR
- Pillow (`pip install -r requirements.txt`)

Install Tesseract:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

## Run the target catalogue

From this folder:

```bash
python catalogue_scraper.py \
  'https://www.catalogueau.com/coles/#catalogue=coles/coles-catalogue-december-31-2025-january-6-2026-vic-metro/' \
  --output coles_2025_12_31_vic_metro \
  --pages all
```

This creates:

- `coles_2025_12_31_vic_metro.csv`
- `coles_2025_12_31_vic_metro.json`

## Run every catalogue in the link dataset

The included link dataset contains 53 VIC Metro catalogues. Process every
unique `viewer_url` with:

```bash
python catalogue_scraper.py \
  --links-file coles_2025_vic_catalogue_links.csv \
  --output coles_2025_all_catalogues \
  --pages all
```

Batch output is structured as:

```text
coles_2025_all_catalogues/
├── all_catalogue_products.csv
├── all_catalogue_products.json
├── trusted_catalogue_products.csv
├── trusted_catalogue_products.json
├── quality_report.json
├── batch_manifest.csv
├── batch_manifest.json
└── catalogues/
    ├── coles__coles-catalogue-january-1-7-2025-vic-metro.csv
    ├── coles__coles-catalogue-january-1-7-2025-vic-metro.json
    └── ...
```

Each catalogue is saved as soon as it finishes. If the command is interrupted,
run the same command again: existing per-catalogue JSON outputs are reused and
only unfinished catalogues are scraped. Add `--force` when you intentionally
want to rebuild every catalogue using updated extraction logic.

The manifest records `completed`, `resumed`, and `failed` catalogues, including
the error and trusted-row count for each failure. A failure does not stop the
remaining links. The command exits with status 1 after completing the batch if
any link failed.

Use `trusted_catalogue_products.csv` for precision-first analysis. A row enters
that file only when its name is independently recognised by both OCR variants,
its values pass all arithmetic and promotion checks, and its special price has
at least two independent supports (SAVE/WAS arithmetic, displayed price, or
unit-price calculation). `all_catalogue_products.csv` retains every detected
row—including uncertain rows—for auditing against `source_page_image`.

The `price_source`, `price_evidence`, `evidence_count`,
`verification_status`, and `review_reason` columns explain each decision. The
quality report summarises why rows were held out. Missing unit-price text by
itself is not an error, but without another independent price signal the row is
not placed in the trusted file.

For a quick test, process only pages 2–3:

```bash
python catalogue_scraper.py 'PASTE_CATALOGUE_URL_HERE' \
  --output test_output --pages 2-3 --keep-images
```

The page-image cache is reused if a catalogue is interrupted. By default,
images are removed after a successful extraction; pass `--keep-images` to
retain them. A full run covers 2,499 high-resolution pages, so it can
take a long time and use substantial disk space if images are retained.

## Accuracy notes

Catalogue pages are promotional images rather than structured product records.
The scraper may calculate `special_price = regular_price - save_amount` when
both printed values are recognised, but arithmetic alone is not considered
verification. Rows without corroborating evidence are flagged with
`needs_review` and excluded from the trusted output.

Page layouts vary, and some promotions do not print both `SAVE` and `WAS`.
Those offers may be incomplete or absent. No OCR-only workflow can guarantee
ground truth; manually compare trusted rows with `source_page_image` when
absolute accuracy is required. The provenance and review columns make that
audit reproducible.

Rows that cannot represent a usable priced offer are omitted from every CSV:
missing/nonpositive special prices, nonpositive regular prices, special prices
at or above a populated regular price, negative savings, and broken
`special + save = regular` arithmetic. A `unit_price_conflict` alone is retained
because multi-buy promotions show a bundle total while the printed unit price
describes one item.

Use a respectful download delay, follow the website's terms, and avoid running
multiple aggressive copies of the scraper at the same time.
