#!/usr/bin/env python3
"""Scrape Catalogue AU catalogue images and extract advertised products with OCR.

The site exposes catalogue metadata as JSON and catalogue pages as JPEG images.
This script downloads those public assets, runs local Tesseract OCR, extracts
SAVE/WAS offer blocks, and writes tidy CSV plus JSON.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import math
import re
import shutil
import subprocess
import sys
import time
from io import BytesIO
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

USER_AGENT = "CatalogueResearchScraper/1.0 (+educational price research)"
API_URL = "https://www.catalogueau.com/api/web/catalogue/v1.php"
IMAGE_ROOT = (
    "https://caau.syd1.cdn.digitaloceanspaces.com/"
    "wp-content/uploads/catalogue"
)
VALIDATION_VERSION = 4


@dataclass
class Word:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    block: int
    paragraph: int
    line: int
    source: int = 0

    @property
    def cx(self) -> float:
        return self.left + self.width / 2

    @property
    def cy(self) -> float:
        return self.top + self.height / 2


@dataclass
class OcrLine:
    text: str
    confidence: float
    left: int
    top: int
    right: int
    bottom: int
    source_count: int = 1
    variant_agreement: float = 1.0

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class Product:
    retailer: str
    region: str
    catalogue_title: str
    catalogue_start_date: str
    catalogue_end_date: str
    page_number: int
    product_name: str
    special_price: float | None
    regular_price: float | None
    save_amount: float | None
    discount_percent: float | None
    promo_type: str
    pack_size: str
    unit_price_text: str
    offer_text: str
    ocr_confidence: float
    needs_review: bool
    review_reason: str
    source_page_image: str
    source_catalogue_url: str
    price_source: str = ""
    price_evidence: str = ""
    evidence_count: int = 0
    verification_status: str = "unvalidated"


def request_bytes(url: str, timeout: int = 45, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last_error}")


def parse_catalogue_url(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url)
    fragment = urllib.parse.parse_qs(parsed.fragment).get("catalogue", [])
    if not fragment:
        raise ValueError("URL must contain #catalogue=store/catalogue-slug")
    bits = fragment[0].strip("/").split("/", 1)
    if len(bits) != 2 or not all(bits):
        raise ValueError("Invalid catalogue fragment; expected store/catalogue-slug")
    return bits[0], bits[1]


def load_catalogue_urls(path: Path) -> list[str]:
    """Load and de-duplicate Catalogue AU viewer URLs from a CSV/text file.

    The supplied link dataset uses a ``viewer_url`` column, but scanning all
    cells also supports renamed columns and simple one-URL-per-line files.
    """
    if not path.is_file():
        raise ValueError(f"Link dataset was not found: {path}")
    urls: list[str] = []
    seen: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            for cell in row:
                for match in re.finditer(r"https?://[^\s\]\[()<>{}\"']+", cell):
                    candidate = match.group(0).rstrip(".,;")
                    if "#catalogue=" not in candidate:
                        continue
                    try:
                        key = parse_catalogue_url(candidate)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid catalogue URL on row {row_number}: {candidate} ({exc})"
                        ) from exc
                    if key not in seen:
                        urls.append(candidate)
                        seen.add(key)
    if not urls:
        raise ValueError(f"No Catalogue AU viewer URLs found in {path}")
    return urls


def catalogue_output_base(output_dir: Path, store: str, slug: str) -> Path:
    """Return a stable, filesystem-safe per-catalogue output path."""
    safe_store = re.sub(r"[^a-zA-Z0-9._-]+", "-", store).strip("-_") or "store"
    safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-_") or "catalogue"
    return output_dir / "catalogues" / f"{safe_store}__{safe_slug}"


def get_metadata(store: str, slug: str) -> dict:
    query = urllib.parse.urlencode(
        {"get": "catalogues2", "brand": store, "slug": slug}
    )
    data = json.loads(request_bytes(f"{API_URL}?{query}"))
    if not data or not data.get("page_count"):
        raise RuntimeError("Catalogue metadata was not returned by the site")
    return data


def iso_date(unix_timestamp: str | int | None) -> str:
    if not unix_timestamp:
        return ""
    return datetime.fromtimestamp(int(unix_timestamp), tz=timezone.utc).date().isoformat()


def infer_region(title: str) -> str:
    match = re.search(
        r"\b(NSW|VIC|QLD|SA|WA|TAS|NT|ACT)(?:\s+(METRO|NORTH|SOUTH))?\b",
        title.upper(),
    )
    return " ".join(x for x in match.groups() if x) if match else ""


def image_url(store: str, slug: str, page: int) -> str:
    safe_store = urllib.parse.quote(store, safe="-")
    safe_slug = urllib.parse.quote(slug, safe="-")
    return f"{IMAGE_ROOT}/{safe_store}/{safe_slug}/{page}.jpg"


def tesseract_words(
    tesseract_input: str | bytes, scale: int = 1, source: int = 0,
    psm: int = 11, whitelist: str = "",
) -> list[Word]:
    """Run one OCR pass and retain which image variant produced each word."""
    command = [
        "tesseract", "stdin" if isinstance(tesseract_input, bytes) else tesseract_input,
        "stdout", "-l", "eng", "--psm", str(psm), "tsv"
    ]
    if whitelist:
        command.extend(["-c", f"tessedit_char_whitelist={whitelist}"])
    result = subprocess.run(
        command,
        input=tesseract_input if isinstance(tesseract_input, bytes) else None,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "Tesseract failed: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    words: list[Word] = []
    # Tesseract TSV is not RFC CSV: OCR text may contain an unmatched quote.
    # Parse the fixed 12 fields directly so a quote cannot consume later rows.
    tsv_lines = result.stdout.decode("utf-8", errors="replace").splitlines()
    for raw_row in tsv_lines[1:]:
        values = raw_row.split("\t", 11)
        if len(values) != 12:
            continue
        (level, _page_num, block_num, par_num, line_num, _word_num,
         left, top, width, height, conf, text_value) = values
        if level != "5" or not text_value.strip():
            continue
        try:
            words.append(
                Word(
                    text=html.unescape(text_value.strip()),
                    confidence=float(conf),
                    left=round(int(left) / scale),
                    top=round(int(top) / scale),
                    width=round(int(width) / scale),
                    height=round(int(height) / scale),
                    block=int(block_num), paragraph=int(par_num),
                    line=int(line_num), source=source,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return words


def missing_price_circle_words(image_path: Path, words: list[Word]) -> list[Word]:
    """OCR dollar digits missed inside red EVERY DAY price circles.

    Whole-page sparse OCR occasionally sees only superscript cents. Detect the
    red ring, then run a single-character OCR crop only when no tall numeric
    word already exists there. Restricting this pass to detected unresolved
    rings keeps packaging artwork out of the price candidates.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        step = 2
        width, height = image.size
        pixels = image.load()
        red = {
            (x // step, y // step)
            for y in range(250, height, step)
            for x in range(0, width, step)
            if (
                pixels[x, y][0] > 180
                and pixels[x, y][1] < 90
                and pixels[x, y][2] < 100
            )
        }
        seen: set[tuple[int, int]] = set()
        recovered: list[Word] = []
        source = 2
        for start in red:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            component: list[tuple[int, int]] = []
            while stack:
                point = stack.pop()
                component.append(point)
                x, y = point
                for neighbour in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if neighbour in red and neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            xs = [point[0] for point in component]
            ys = [point[1] for point in component]
            left, top = min(xs) * step, min(ys) * step
            right = (max(xs) + 1) * step
            bottom = (max(ys) + 1) * step
            box_width, box_height = right - left, bottom - top
            area = len(component) * step * step
            if not (
                100 <= box_width <= 175
                and 95 <= box_height <= 175
                and area >= 3000
            ):
                continue
            existing_numeric = [
                word for word in words
                if re.search(r"\d", word.text)
                and left <= word.cx <= right
                and top <= word.cy <= bottom
            ]
            existing_main = [
                word for word in existing_numeric
                if word.height >= box_height * 0.25
            ]
            existing_cents = [
                word for word in existing_numeric
                if word.height < box_height * 0.25
                and word.cx >= left + box_width * 0.53
                and word.cy <= top + box_height * 0.60
            ]

            def targeted_digits(
                relative_box: tuple[float, float, float, float],
                min_digits: int, max_digits: int,
            ) -> str:
                region_left = max(0, round(left + box_width * relative_box[0]))
                region_top = max(0, round(top + box_height * relative_box[1]))
                region_right = min(width, round(left + box_width * relative_box[2]))
                region_bottom = min(height, round(top + box_height * relative_box[3]))
                if region_right <= region_left or region_bottom <= region_top:
                    return ""
                region = image.crop(
                    (region_left, region_top, region_right, region_bottom)
                ).resize(
                    ((region_right - region_left) * 6,
                     (region_bottom - region_top) * 6),
                    Image.Resampling.LANCZOS,
                )
                region_buffer = BytesIO()
                region.save(region_buffer, format="PNG")
                candidates: list[tuple[float, str]] = []
                for region_psm in (7, 11, 13):
                    for candidate in tesseract_words(
                        region_buffer.getvalue(), scale=6, source=source,
                        psm=region_psm, whitelist="$0123456789",
                    ):
                        candidate_digits = re.sub(r"\D", "", candidate.text)
                        if min_digits <= len(candidate_digits) <= max_digits:
                            candidates.append((candidate.confidence, candidate_digits))
                return max(candidates, default=(-1.0, ""))[1]

            # Re-OCR the main dollars and superscript cents separately. Whole-
            # circle OCR often fuses "$2.75" into "97" or sees only "$2".
            # Fixed subregions preserve the typographic distinction.
            suspicious_fused_main = any(
                word.confidence < 20
                and "$" in word.text
                and len(re.sub(r"\D", "", word.text)) >= 2
                for word in existing_main
            )
            if suspicious_fused_main:
                main_digits = targeted_digits((0.05, 0.12, 0.70, 0.85), 1, 1)
                if main_digits:
                    recovered.append(Word(
                        text=main_digits, confidence=90.0,
                        left=left + round(box_width * 0.18),
                        top=top + round(box_height * 0.23),
                        width=round(box_width * 0.48),
                        height=round(box_height * 0.50),
                        block=1, paragraph=1, line=1, source=source,
                    ))
                    source += 1
                    existing_main = [recovered[-1]]
            # Only attempt cents when the main OCR token visibly contains an
            # unresolved glyph ("1?" / "6?"). Blindly re-reading every small
            # top-right region creates convincing but false cents from nearby
            # packaging artwork.
            unresolved_main = any("?" in word.text for word in existing_main)
            if (unresolved_main or suspicious_fused_main) and not existing_cents:
                cents_digits = targeted_digits((0.40, 0.05, 1.00, 0.65), 1, 2)
                if cents_digits:
                    recovered.append(Word(
                        text=cents_digits, confidence=90.0,
                        left=left + round(box_width * 0.58),
                        top=top + round(box_height * 0.27),
                        width=round(box_width * 0.24),
                        height=round(box_height * 0.18),
                        block=1, paragraph=1, line=1, source=source,
                    ))
                    source += 1
            if existing_main:
                continue
            crop_left, crop_top = max(0, left + 5), max(0, top + 3)
            crop_right, crop_bottom = min(width, right - 5), min(height, bottom - 3)
            crop = image.crop((crop_left, crop_top, crop_right, crop_bottom)).resize(
                ((crop_right - crop_left) * 4, (crop_bottom - crop_top) * 4),
                Image.Resampling.LANCZOS,
            )
            buffer = BytesIO()
            crop.save(buffer, format="PNG")
            crop_words = tesseract_words(
                buffer.getvalue(), scale=4, source=source,
                psm=13, whitelist="$0123456789",
            )
            digit_words = [
                word for word in crop_words
                if len(re.sub(r"\D", "", word.text)) == 1
            ]
            if not digit_words:
                crop_words = tesseract_words(
                    buffer.getvalue(), scale=4, source=source,
                    psm=11, whitelist="$0123456789",
                )
                digit_words = [
                    word for word in crop_words
                    if len(re.sub(r"\D", "", word.text)) == 1
                ]
            source += 1
            if digit_words:
                word = max(digit_words, key=lambda candidate: candidate.height)
                digits = re.sub(r"\D", "", word.text)
                # PSM 13 may return the surrounding ring as the glyph box.
                # Normalise it to the large dollar-digit area so an existing
                # superscript-cents line can be paired geometrically.
                word.text = digits
                word.left = left + round(box_width * 0.18)
                word.top = top + round(box_height * 0.25)
                word.width = round(box_width * 0.55)
                word.height = round(box_height * 0.46)
                word.confidence = max(word.confidence, 80.0)
                recovered.append(word)
        return recovered
    except Exception:
        return []


def run_tesseract(image_path: Path) -> list[Word]:
    # Red text on yellow price labels and small black product text respond
    # differently to preprocessing. Keep both passes: the original often finds
    # SAVE while the enlarged/sharpened variant often finds WAS and names.
    words = tesseract_words(str(image_path), source=0)
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
        with Image.open(image_path) as source:
            scale = 2 if source.width < 1800 else 1
            processed = source.convert("RGB")
            if scale > 1:
                processed = processed.resize(
                    (source.width * scale, source.height * scale), Image.Resampling.LANCZOS
                )
            processed = ImageEnhance.Contrast(processed).enhance(1.08)
            processed = processed.filter(ImageFilter.SHARPEN)
            buffer = BytesIO()
            processed.save(buffer, format="PNG")
            words.extend(tesseract_words(buffer.getvalue(), scale=scale, source=1))
    except Exception:
        # The original pass is still usable if Pillow/preprocessing fails.
        pass
    words.extend(missing_price_circle_words(image_path, words))
    return words


def make_lines(words: list[Word]) -> list[OcrLine]:
    grouped: dict[tuple[int, int, int, int], list[Word]] = {}
    for word in words:
        grouped.setdefault(
            (word.source, word.block, word.paragraph, word.line), []
        ).append(word)
    lines: list[tuple[int, OcrLine]] = []
    for (source, _block, _paragraph, _line), original_group in grouped.items():
        # Sparse-text mode occasionally places two catalogue columns in one OCR
        # line. Split on a large horizontal gap before building OcrLine objects.
        original_group.sort(key=lambda w: w.left)
        segments: list[list[Word]] = [[]]
        previous_right: int | None = None
        for word in original_group:
            if previous_right is not None and word.left - previous_right > 30:
                segments.append([])
            segments[-1].append(word)
            previous_right = word.left + word.width
        for group in segments:
            if not group:
                continue
            weights = [max(1, len(w.text)) for w in group]
            confidence = sum(w.confidence * n for w, n in zip(group, weights)) / sum(weights)
            lines.append(
                (source, OcrLine(
                    text=" ".join(w.text for w in group), confidence=confidence,
                    left=min(w.left for w in group), top=min(w.top for w in group),
                    right=max(w.left + w.width for w in group),
                    bottom=max(w.top + w.height for w in group),
                ))
            )
    return merge_line_variants(lines)


def lines_match(first: OcrLine, second: OcrLine) -> bool:
    """Return true when two OCR variants describe the same printed line."""
    vertical_overlap = max(0, min(first.bottom, second.bottom) - max(first.top, second.top))
    horizontal_overlap = max(0, min(first.right, second.right) - max(first.left, second.left))
    first_height = max(1, first.bottom - first.top)
    second_height = max(1, second.bottom - second.top)
    first_width = max(1, first.right - first.left)
    second_width = max(1, second.right - second.left)
    return (
        vertical_overlap >= 0.55 * min(first_height, second_height)
        and horizontal_overlap >= 0.55 * min(first_width, second_width)
        and min(first_width, second_width) / max(first_width, second_width) >= 0.50
        and abs(first.cx - second.cx) <= max(12, 0.25 * max(first_width, second_width))
    )


def line_quality(line: OcrLine) -> float:
    """Prefer parseable offer lines, then complete and confident text."""
    normalised = normalise_offer(line.text)
    parsed_prices = sum(
        money_after(label, line.text) is not None for label in ("SAVE", "WAS")
    )
    offer_label = int("SAVE" in normalised or "WAS" in normalised)
    useful_length = min(80, len(re.sub(r"\W", "", line.text)))
    return parsed_prices * 1000 + offer_label * 100 + useful_length * 2 + max(0, line.confidence)


def merge_line_variants(lines: list[tuple[int, OcrLine]]) -> list[OcrLine]:
    """Deduplicate matching lines from original and enhanced OCR passes."""
    merged: list[tuple[set[int], OcrLine]] = []
    for source, line in sorted(lines, key=lambda item: (item[1].top, item[1].left)):
        match_index = next(
            (
                idx for idx, (sources, existing) in enumerate(merged)
                if source not in sources and lines_match(existing, line)
            ),
            None,
        )
        if match_index is None:
            merged.append(({source}, line))
            continue
        sources, existing = merged[match_index]
        sources.add(source)
        first_text = re.sub(r"\W+", " ", existing.text.casefold()).strip()
        second_text = re.sub(r"\W+", " ", line.text.casefold()).strip()
        agreement = difflib.SequenceMatcher(
            None, first_text, second_text
        ).ratio()
        if line_quality(line) > line_quality(existing):
            merged[match_index] = (sources, line)
            chosen = line
        else:
            chosen = existing
        chosen.source_count = len(sources)
        chosen.variant_agreement = min(
            existing.variant_agreement, line.variant_agreement, agreement
        )
    return sorted((line for _sources, line in merged), key=lambda line: (line.top, line.left))


def normalise_offer(text: str) -> str:
    value = text.upper().replace("§", "$").replace("¢", "")
    value = value.replace("SAVES", "SAVE$").replace("WASS", "WAS$")
    value = value.replace("SAVE*", "SAVE$").replace("WAS*", "WAS$")
    # Superscript/condensed 1s in prices are also commonly recognised as a
    # square bracket (SAVE$]2.50 is the printed SAVE $12.50).
    value = re.sub(r"(?<=\$)\](?=\d(?:[.,]\d{2})\b)", "1", value)
    value = re.sub(r"\b(SAVE|WAS)\$S(?=[.,]\d{2}\b)", r"\1$5", value)
    value = re.sub(r"(?<=\d)[IL](?=[.,]\d{2}\b)", "1", value)
    # A narrow printed 1 is often OCR'd as I/l in four-digit price labels:
    # WAS$I750 represents WAS $17.50.
    value = re.sub(
        r"\b(SAVE|WAS)\$[IL](\d)(\d{2})\b",
        lambda match: (
            f"{match.group(1)}$1{match.group(2)}.{match.group(3)}"
        ),
        value,
    )
    value = re.sub(
        r"\b(SAVE|WAS)\$[IL](\d{1,2})\b",
        lambda match: f"{match.group(1)}$1{match.group(2)}",
        value,
    )
    value = re.sub(r"\b(SAVE|WAS)\s+5(?=\d)", r"\1$", value)
    value = re.sub(
        r"\b(SAVE|WAS)[IL]{2}(\d{2})\b",
        lambda match: f"{match.group(1)}$11.{match.group(2)}",
        value,
    )
    return re.sub(r"\s+", " ", value)


def money_after(label: str, text: str) -> float | None:
    cleaned = normalise_offer(text)
    match = re.search(
        rf"\b{label}[^0-9]{{0,4}}([0-9]+(?:[.,][0-9]{{1,2}})?)", cleaned
    )
    if not match:
        return None
    try:
        token = match.group(1).replace(",", ".")
        value = float(token)
        # Yellow labels omit the decimal point surprisingly often. Four
        # digits on these compact labels represent dollars plus cents (3750
        # is $37.50), not a four-thousand-dollar grocery shelf price.
        if "." not in token and len(token) == 4 and value >= 1000:
            value /= 100
        return round(value, 2)
    except ValueError:
        return None


def pair_offer_lines(lines: list[OcrLine]) -> list[tuple[OcrLine, OcrLine | None]]:
    # Require an actual label boundary. Substring matching incorrectly treated
    # product words such as "Dishwashing" and "Life Savers" as offer labels.
    save_lines = [
        line for line in lines
        if re.match(r"^[^A-Z0-9]{0,3}SAVE(?=[^A-Z]|$)", normalise_offer(line.text))
    ]
    was_lines = [
        line for line in lines
        if re.match(r"^[^A-Z0-9]{0,3}WAS(?=[^A-Z]|$)", normalise_offer(line.text))
    ]
    pairs: list[tuple[OcrLine, OcrLine | None]] = []
    used: set[int] = set()
    for save in save_lines:
        candidates = [
            (idx, was) for idx, was in enumerate(was_lines)
            if idx not in used and -5 <= was.top - save.bottom <= 55
            and abs(was.cx - save.cx) <= 135
        ]
        if candidates:
            idx, was = min(candidates, key=lambda item: (
                abs(item[1].cx - save.cx) + 2 * abs(item[1].top - save.bottom)
            ))
            used.add(idx)
            pairs.append((save, was))
        else:
            pairs.append((save, None))
    # Recover offers where SAVE was missed but WAS was recognised.
    for idx, was in enumerate(was_lines):
        if idx not in used:
            pairs.append((was, was))
    return sorted(pairs, key=lambda pair: (pair[0].top, pair[0].left))


def nearby_product_text(
    anchor: OcrLine, was: OcrLine | None, lines: list[OcrLine], image_width: int,
    horizontal_radius: float | None = None,
) -> tuple[str, str, float]:
    cell_width = image_width / 3
    radius = horizontal_radius if horizontal_radius is not None else cell_width * 0.52
    left = max(0, anchor.cx - radius)
    right = min(image_width, anchor.cx + radius)
    # Red EVERY DAY labels can sit only 3-4 pixels above the description.
    start_y = max(anchor.bottom, was.bottom if was else anchor.bottom) + 2
    end_y = start_y + 125
    candidates = [
        line for line in lines
        if start_y <= line.top <= end_y and left <= line.cx <= right
        and not re.search(r"\b(SAVE|WAS)\b", normalise_offer(line.text))
    ]
    candidates.sort(key=lambda line: (line.top, line.left))
    selected: list[OcrLine] = []
    for candidate_index, line in enumerate(candidates):
        text = re.sub(r"\s+", " ", line.text).strip(" |")
        if not text or text.startswith("†"):
            continue
        if re.search(
            r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+20\d{2}\b",
            text, flags=re.IGNORECASE,
        ) or re.fullmatch(r"[\[| ]*(?:EVERY DAY|DOWN DOWN)[\]| ]*", text,
                          flags=re.IGNORECASE):
            continue
        # Keep bold description text that shares a line with unit-price/T&C
        # text. Previously the early `per litre` check discarded the complete
        # line, including descriptions such as "Drink 1.25 Litre".
        product_part = re.split(
            r"\s+(?=\$\d|\*?See\b|On sale\b|per\s+(?:100|litre|kg|each)\b)",
            text, maxsplit=1, flags=re.IGNORECASE,
        )[0].strip()
        boundary_found = product_part != text
        if re.match(r"^(?:\$\s*\d|\*?See\s+page|On sale\b|per\s+(?:100|litre|kg|each)\b)",
                    product_part, flags=re.IGNORECASE):
            product_part = ""
            boundary_found = True
        leading_letters = re.sub(r"[^A-Za-z]", "", product_part)
        if not selected and product_part.upper() == "MADE":
            continue
        if (
            product_part
            and not selected
            and len(leading_letters) <= 2
            and line.confidence < 75
        ):
            continue
        if product_part and not re.fullmatch(r"\$?[\d.,]+", product_part):
            selected.append(OcrLine(
                product_part, line.confidence, line.left, line.top,
                line.right, line.bottom, line.source_count,
                line.variant_agreement,
            ))
        if boundary_found and selected:
            break
        # Most descriptions use at most two lines. Continue to a third only
        # when line two visibly ends with a conjunction (for example,
        # "Sourdough Vienna or" / "Pane di Casa").
        needs_continuation = bool(
            selected and re.search(r"\b(?:or|and)\s*$", selected[-1].text,
                                   flags=re.IGNORECASE)
        )
        next_text = (
            candidates[candidate_index + 1].text
            if candidate_index + 1 < len(candidates) else ""
        )
        next_adds_pack_details = bool(
            len(selected) in {2, 3}
            and re.search(
                r"\b\d+(?:\.\d+)?\s*(?:mL|L|g|kg|pack|pk)\b",
                next_text, flags=re.IGNORECASE,
            )
        )
        if len(selected) >= 4 or (
            len(selected) >= 2 and not needs_continuation and not next_adds_pack_details
        ) or (selected and line.top - selected[0].top > 55):
            break
    while len(selected) > 1:
        leading_letters = re.sub(r"[^A-Za-z]", "", selected[0].text)
        if len(leading_letters) > 2 or selected[0].confidence >= 75:
            break
        selected.pop(0)
    product_name = " ".join(line.text for line in selected)
    product_name = re.sub(r"\s+", " ", product_name).strip(" .,-")
    product_name = re.sub(r"\borigiral\b\s*", "", product_name,
                          flags=re.IGNORECASE)
    product_name = re.sub(r"\b1k\b", "1kg", product_name, flags=re.IGNORECASE)
    product_name = re.sub(
        r"\b[IiIl]kg\b", "1kg", product_name, flags=re.IGNORECASE
    )
    product_name = re.sub(r"\bSofft\b", "Soft", product_name)
    product_name = re.sub(
        r"\s+[xX]\s+(?=\d+(?:\.\d+)?\s*(?:g|kg)\b)",
        " ", product_name,
    )
    product_name = re.sub(
        r"\s+per\s+100g\s+(?=\d+(?:\.\d+)?g\b)", " ", product_name,
        flags=re.IGNORECASE,
    )
    raw_nearby = " | ".join(line.text for line in candidates[:6])
    confidence = (
        sum(line.confidence for line in selected) / len(selected) if selected else 0.0
    )
    # High Tesseract confidence is not the same as independent confirmation.
    # Cap name confidence below the trusted threshold unless both whole-page
    # OCR variants found the same printed description lines.
    if selected and any(
        line.source_count < 2 or line.variant_agreement < 0.94
        for line in selected
    ):
        confidence = min(confidence, 79.0)
    return product_name, raw_nearby, confidence


def extract_pack_size(text: str) -> str:
    matches = re.findall(
        r"\b(?:\d+\s*[xX]\s*)?\d+(?:\.\d+)?\s*"
        r"(?:mL|L|litres?|liters?|g|kg|pack|pk)\b",
        text, flags=re.IGNORECASE,
    )
    return "; ".join(dict.fromkeys(match.strip() for match in matches))


def extract_unit_price(text: str) -> str:
    match = re.search(
        r"\$\s*\d+(?:\.\d{1,2})?\s+per\s+"
        r"(?:100(?:\s*[|:/-]?\s*(?:g|mL|each))?|kg|litre|each)",
        text, flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", match.group(0)) if match else ""


def displayed_special_price(
    anchor: OcrLine, lines: list[OcrLine], regular_price: float | None = None
) -> float | None:
    """Recover the large price printed immediately above a SAVE/WAS label.

    Catalogue price circles commonly split dollars and superscript cents into
    separate OCR lines (for example, ``2`` and ``10``). This is deliberately a
    fallback: SAVE/WAS arithmetic remains the primary, less ambiguous source.
    """
    nearby = [
        line for line in lines
        if 8 <= anchor.top - line.bottom <= 105
        and anchor.top - 145 <= line.top
        and abs(line.cx - anchor.cx) <= 85
        and re.search(r"\d", line.text)
        and not re.search(r"\b(?:SAVE|WAS)\b", normalise_offer(line.text))
        and not re.search(r"\b(?:ANY\s*)?\d+\s*FOR\b", line.text,
                          flags=re.IGNORECASE)
    ]
    if not nearby:
        return None

    def digits(line: OcrLine) -> str:
        return "".join(re.findall(r"\d", line.text))

    # The dollar amount uses the tallest type in the circle. Prefer height,
    # then closeness to the offer label and OCR confidence.
    main = max(
        nearby,
        key=lambda line: (
            line.bottom - line.top,
            -(anchor.top - line.bottom),
            line.confidence,
        ),
    )
    main_digits = digits(main)
    if not main_digits or len(main_digits) > 3:
        return None
    dollars = int(main_digits)
    main_height = max(1, main.bottom - main.top)

    cents_candidates = [
        line for line in nearby
        if line is not main
        and 1 <= len(digits(line)) <= 2
        and not re.search(r"[A-Za-z]", line.text)
        and line.bottom - line.top <= main_height * 0.65
        and line.cx > main.cx
        and abs(line.top - main.top) <= 28
    ]
    cents = 0
    if cents_candidates:
        cents_line = min(
            cents_candidates,
            key=lambda line: (abs(line.left - main.right), abs(line.top - main.top)),
        )
        cents_text = digits(cents_line)
        cents = int(cents_text) * (10 if len(cents_text) == 1 else 1)

    value = round(dollars + cents / 100, 2)
    if value <= 0 or (regular_price is not None and value >= regular_price):
        return None
    return value


def unit_price_estimate(
    product_name: str, nearby_text: str
) -> tuple[float, float] | None:
    """Return a shelf-price estimate and error bound from printed unit pricing."""
    unit_text = extract_unit_price(nearby_text)
    if not unit_text:
        return None
    unit_match = re.search(
        r"\$\s*(\d+(?:\.\d{1,2})?)\s+per\s+"
        r"(100(?:\s*[|:/-]?\s*(?:g|mL|each))?|kg|litre|each)",
        unit_text, flags=re.IGNORECASE,
    )
    if not unit_match:
        return None
    unit_price = float(unit_match.group(1))
    basis = re.sub(r"[\s|:/-]+", "", unit_match.group(2)).lower()
    if basis == "100":
        # OCR sometimes separates the unit from "$9.50 per 100" as another
        # token. Infer only the dimension (never the quantity) from the name.
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:mL|L|litres?|liters?)\b",
                     product_name, flags=re.IGNORECASE):
            basis = "100ml"
        elif re.search(r"\b\d+(?:\.\d+)?\s*(?:g|kg)\b",
                       product_name, flags=re.IGNORECASE):
            basis = "100g"
        else:
            return None

    # Tesseract commonly reads a printed 1 as I/l in compact pack sizes.
    normalised_name = re.sub(
        r"\b[IiIl](?=\s*(?:kg|g|mL|L|litres?|liters?)\b)",
        "1", product_name, flags=re.IGNORECASE,
    )
    normalised_name = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*k\b", r"\1kg", normalised_name,
        flags=re.IGNORECASE,
    )
    normalised_name = re.sub(
        r"\bper\s+(?:100\s*(?:g|mL)|kg|litre|each)\b",
        "", normalised_name, flags=re.IGNORECASE,
    )

    pack_matches: list[tuple[str, str]] = []
    for match in re.finditer(
        r"\b(?:(\d+)\s*[xX]\s*)?(\d+(?:\.\d+)?)\s*"
        r"(mL|L|litres?|liters?|g|kg)\b",
        normalised_name, flags=re.IGNORECASE,
    ):
        count_text, quantity_text, unit_text_value = match.groups()
        quantity = float(quantity_text) * (int(count_text) if count_text else 1)
        pack_matches.append((f"{quantity:g}", unit_text_value))
    multiplier: float
    assumed_direct_unit = False

    if basis == "litre":
        if len(pack_matches) != 1:
            return None
        quantity_text, pack_unit_text = pack_matches[0]
        quantity = float(quantity_text)
        pack_unit = pack_unit_text.lower()
        if pack_unit == "ml":
            quantity /= 1000
        elif pack_unit not in {"l", "litre", "litres", "liter", "liters"}:
            return None
        multiplier = quantity
    elif basis == "100ml":
        if len(pack_matches) != 1:
            return None
        quantity_text, pack_unit_text = pack_matches[0]
        quantity = float(quantity_text)
        pack_unit = pack_unit_text.lower()
        if pack_unit in {"l", "litre", "litres", "liter", "liters"}:
            quantity *= 1000
        elif pack_unit != "ml":
            return None
        multiplier = quantity / 100
    elif basis == "kg":
        if not pack_matches:
            # Fresh meat, deli and produce prices are advertised directly per
            # kilogram and commonly have no separate pack weight in the name.
            multiplier = 1.0
            assumed_direct_unit = True
        elif len(pack_matches) != 1:
            return None
        else:
            quantity_text, pack_unit_text = pack_matches[0]
            quantity = float(quantity_text)
            pack_unit = pack_unit_text.lower()
            if pack_unit == "g":
                quantity /= 1000
            elif pack_unit != "kg":
                return None
            multiplier = quantity
    elif basis == "100g":
        if len(pack_matches) != 1:
            return None
        quantity_text, pack_unit_text = pack_matches[0]
        quantity = float(quantity_text)
        pack_unit = pack_unit_text.lower()
        if pack_unit == "kg":
            quantity *= 1000
        elif pack_unit != "g":
            return None
        multiplier = quantity / 100
    elif basis == "each":
        pack_counts = list(dict.fromkeys(re.findall(
            r"\b(\d+)\s*(?:pack|pk)\b", normalised_name, flags=re.IGNORECASE
        )))
        multiplier = float(pack_counts[0]) if len(pack_counts) == 1 else 1.0
    elif basis == "100each":
        pack_counts = list(dict.fromkeys(re.findall(
            r"\b(\d+)\s*(?:pack|pk)\b", normalised_name, flags=re.IGNORECASE
        )))
        if len(pack_counts) != 1:
            return None
        multiplier = float(pack_counts[0]) / 100
    else:
        return None

    raw_price = multiplier * unit_price
    # Printed unit prices are rounded to cents. Their maximum accumulated
    # rounding error grows with pack quantity and is used by callers to reject
    # unsafe estimates (for example, 100 bags at $0.05 each).
    uncertainty = (
        0.51 if assumed_direct_unit else multiplier * 0.005 + 0.005
    )
    return raw_price, uncertainty


def price_from_unit_text(product_name: str, nearby_text: str) -> float | None:
    """Estimate the advertised price, rounded to a normal five-cent price."""
    estimate = unit_price_estimate(product_name, nearby_text)
    if estimate is None:
        return None
    raw_price, uncertainty = estimate
    five_cent_price = round(math.floor(raw_price * 20 + 0.5) / 20, 2)
    # Most shelf prices use five-cent increments, but some EVERY DAY prices
    # genuinely end in odd cents. Reverse unit-price arithmetic can identify
    # those when no five-cent value is possible within the printed unit price's
    # rounding uncertainty (for example, 1.25 L at $1.30/L -> $1.62).
    if abs(five_cent_price - raw_price) > uncertainty:
        return round(raw_price, 2)
    return five_cent_price


def half_price_page(lines: list[OcrLine], products: list[Product]) -> bool:
    """Recognise a half-price page from its heading or established offers."""
    if any(
        re.search(r"(?:1|I|V)\s*/\s*2\s*PRICE", normalise_offer(line.text))
        for line in lines
    ):
        return True
    known = [
        product for product in products
        if product.save_amount is not None and product.regular_price
    ]
    return (
        len(known) >= 3
        and sum(
            abs(product.save_amount / product.regular_price - 0.5) <= 0.03
            for product in known
        ) / len(known) >= 0.8
    )


def excluded_page_reason(lines: list[OcrLine]) -> str:
    """Identify promotional pages that are outside grocery-product scope."""
    for line in lines:
        upper = line.text.upper()
        compact = re.sub(r"[^A-Z]", "", upper)
        if "LIQUORLAND" in compact:
            return "liquorland_page"
        if "COLESMOBILE" in compact:
            return "mobile_sim_page"
        if line.top < 400 and re.search(r"\bGIFT\s*(?:CARD|VOUCHER)S?\b", upper):
            return "gift_card_page"
    return ""


def excluded_offer_reason(product_name: str, offer_text: str) -> str:
    """Identify standalone SIM and gift-card offers on otherwise mixed pages."""
    # Limit OCR context to the offer's labels and first description/spec lines;
    # later text may contain page-wide terms such as "SIMs" or "gift cards".
    leading_offer_text = " | ".join(offer_text.split("|")[:5])
    description = f"{product_name} | {leading_offer_text}"
    sim_match = re.search(
        r"\b(?:PRE[\s-]*PAID\s+SIM|SIM\s+(?:CARD|KIT|STARTER\s+KIT|PACK|PLAN|ONLY))\b",
        description, flags=re.IGNORECASE,
    )
    hardware_name = re.search(
        r"\b(?:I?PHONE|HANDSET|MODEM|ROUTER|\dG|SCREEN|DISPLAY|CAMERA)\b",
        product_name, flags=re.IGNORECASE,
    )
    if sim_match and not hardware_name:
        return "sim_offer"
    description = re.sub(
        r"\b(?:EXCLUDES?|EXCLUDING)\s+GIFT\s*(?:CARD|VOUCHER)S?\b",
        "", description, flags=re.IGNORECASE,
    )
    if re.search(
        r"\bGIFT\s*(?:CARD|VOUCHER)S?\b", description, flags=re.IGNORECASE
    ):
        return "gift_card_offer"
    return ""


def product_quality(product: Product) -> tuple[int, int, int, int, int, int, float]:
    """Rank duplicate OCR rows by price integrity and completeness."""
    price_values = (
        product.special_price, product.regular_price, product.save_amount
    )
    complete_fields = sum(value is not None for value in price_values)
    valid_arithmetic = int(
        all(value is not None for value in price_values)
        and product.special_price > 0
        and product.save_amount >= 0
        and product.regular_price > product.special_price
        and abs(
            product.special_price + product.save_amount - product.regular_price
        ) <= 0.011
    )
    return (
        valid_arithmetic,
        complete_fields,
        int(not product.needs_review),
        int(bool(product.pack_size)),
        int(bool(product.unit_price_text)),
        len(re.sub(r"[^A-Za-z0-9]", "", product.product_name)),
        product.ocr_confidence,
    )


def deduplicate_products(products: list[Product]) -> list[Product]:
    """Keep the best row when OCR creates the same named offer twice."""
    best_by_name: list[tuple[int, Product]] = []
    for index, product in enumerate(products):
        if not product.product_name.strip():
            continue
        match_index = next(
            (
                candidate_index
                for candidate_index, (_original_index, existing) in enumerate(best_by_name)
                if names_overlap(product.product_name, existing.product_name)
            ),
            None,
        )
        if match_index is None:
            best_by_name.append((index, product))
        elif product_quality(product) > product_quality(best_by_name[match_index][1]):
            original_index, _existing = best_by_name[match_index]
            best_by_name[match_index] = (original_index, product)
    return [
        product for _index, product in sorted(best_by_name, key=lambda item: item[0])
    ]


def promotion_marker_kind(line: OcrLine) -> str:
    """Recognise compact red EVERY DAY and DOWN DOWN offer labels."""
    if line.right - line.left > 200 or line.bottom - line.top > 60:
        return ""
    upper = re.sub(r"[^A-Z]+", " ", line.text.upper()).strip()
    if "EVERY DAY" in upper:
        return "everyday"
    if "DOWN DOWN" in upper or re.search(r"\bWN DOWN\b", upper):
        return "down_down"
    return ""


def names_overlap(first: str, second: str) -> bool:
    """Detect exact names and obvious OCR descriptions containing one another."""
    normalise = lambda value: re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    def trim_noise(value: str) -> str:
        tokens = normalise(value).split()
        while tokens and len(tokens[0]) <= 2:
            tokens.pop(0)
        return " ".join(tokens)

    first_key, second_key = trim_noise(first), trim_noise(second)
    if not first_key or not second_key:
        return False
    shorter, longer = sorted((first_key, second_key), key=len)
    if shorter == longer or (len(shorter) >= 8 and shorter in longer):
        return True
    first_tokens, second_tokens = set(first_key.split()), set(second_key.split())
    overlap = first_tokens & second_tokens
    return (
        len(overlap) >= 4
        and len(overlap) / min(len(first_tokens), len(second_tokens)) >= 0.85
    )


def plausible_recovered_name(name: str, confidence: float) -> bool:
    """Reject footer fragments, pack-only text, and other non-product names."""
    cleaned = re.sub(r"\s+", " ", name).strip(" .,-")
    if confidence < 70 or len(re.sub(r"[^A-Za-z]", "", cleaned)) < 6:
        return False
    if re.match(r"^(?:on sale|bonus|save|was|every day|down down)\b", cleaned,
                flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"(?:\d+\s*)?(?:pack|pk|each)", cleaned,
                    flags=re.IGNORECASE):
        return False
    if re.search(
        r"(?:ADVERTISER\s+PROMOTION|TRADITIONAL\s+CUSTODIANS|"
        r"THIS\s+(?:MEDICINE|PRODUCT)\s+MAY\s+NOT\s+BE\s+RIGHT|"
        r"READ\s+THE\s+(?:LABEL|WARNINGS)|\bIS\s+DON\.?\s+IS\s+GOOD\b|"
        r"\bEXCELL\s+XCELLENC\b)",
        cleaned, flags=re.IGNORECASE,
    ):
        return False
    return True


def choose_recovered_special(
    displayed: float | None, product_name: str, nearby_text: str
) -> float | None:
    """Reconcile a price-circle reading with rounded unit-price arithmetic."""
    estimate = unit_price_estimate(product_name, nearby_text)
    if estimate is None:
        return displayed
    raw_unit_price, uncertainty = estimate
    raw_unit_price = round(raw_unit_price, 2)
    # A three-digit OCR token often contains dollars and superscript cents as
    # one value ("450" for $4.50, "360" for $3.60).
    if displayed is not None and 100 <= displayed <= 999:
        combined = displayed / 100
        if abs(combined - raw_unit_price) <= max(0.15, uncertainty * 2):
            return round(combined, 2)
    # OCR can recognise superscript cents but miss the large dollar digit. Try
    # neighbouring dollar values from unit-price arithmetic before rejecting a
    # large apparent disagreement. The neighbour accounts for rounded unit
    # prices (for example, 50 at $0.10 each versus an actual price of $4.80).
    if (
        displayed is not None
        and 20 <= displayed <= 99
    ):
        cents = math.floor(displayed) / 100
        base_dollars = math.floor(raw_unit_price)
        candidates = [
            dollars + cents
            for dollars in range(max(0, base_dollars - 1), base_dollars + 2)
        ]
        combined = min(candidates, key=lambda value: abs(value - raw_unit_price))
        if abs(combined - raw_unit_price) <= max(0.15, uncertainty * 2):
            return round(combined, 2)
    if (
        displayed is not None
        and (displayed > raw_unit_price * 3 or displayed * 3 < raw_unit_price)
        and uncertainty > 0.06
    ):
        return None
    if uncertainty > 0.50 and displayed is not None:
        if (
            displayed.is_integer()
            and math.floor(raw_unit_price) == displayed
            and raw_unit_price - displayed < 1
        ):
            return price_from_unit_text(product_name, nearby_text)
        return displayed
    if uncertainty > 0.06:
        return displayed
    unit_five_cent = price_from_unit_text(product_name, nearby_text)
    if displayed is None:
        return raw_unit_price
    if abs(displayed - raw_unit_price) <= max(0.10, uncertainty * 2):
        return displayed
    return unit_five_cent


def multibuy_near(anchor: OcrLine, lines: list[OcrLine]) -> bool:
    """Return whether a price circle is explicitly labelled ``N for``.

    The quantity is normally a small line immediately above the large total.
    It must not be mistaken for the dollar amount, and unit-price arithmetic
    must not replace the advertised multi-buy total with a per-item value.
    """
    return any(
        re.search(r"\b(?:ANY\s*)?\d+\s*FOR\b", line.text, flags=re.IGNORECASE)
        and abs(line.cx - anchor.cx) <= 105
        and -20 <= anchor.top - line.bottom <= 190
        for line in lines
    )


def reconcile_special_price(
    special: float | None,
    displayed: float | None,
    product_name: str,
    nearby_text: str,
    is_multibuy: bool = False,
) -> float | None:
    """Choose the headline price supported by independent visual evidence.

    SAVE/WAS arithmetic is useful but a single lost decimal can still produce
    a mathematically valid, wildly wrong result. Printed unit pricing gives an
    independent cross-check. Multi-buy totals are deliberately exempt because
    their unit price describes one item rather than the advertised bundle.
    """
    if is_multibuy and displayed is not None:
        return displayed

    estimate = unit_price_estimate(product_name, nearby_text)
    if estimate is None or estimate[1] > 0.50:
        return displayed if special is None and displayed is not None else special

    raw_unit_price, uncertainty = estimate
    tolerance = max(0.011, uncertainty)
    if displayed is not None and abs(displayed - raw_unit_price) <= tolerance:
        return displayed
    if special is not None and abs(special - raw_unit_price) <= tolerance:
        return special

    # A fused OCR token can encode dollars and cents without a decimal. Retain
    # it if scaling produces a value supported by the printed unit price.
    for candidate_source in (displayed, special):
        if candidate_source is None:
            continue
        for divisor in (10, 100):
            candidate = round(candidate_source / divisor, 2)
            if abs(candidate - raw_unit_price) <= max(0.15, uncertainty):
                return candidate

    return price_from_unit_text(product_name, nearby_text)


def page_discount_rate(lines: list[OcrLine]) -> float | None:
    """Read a prominent top-of-page percentage/half-price heading."""
    heading = " ".join(
        normalise_offer(line.text) for line in lines if line.top < 250
    )
    if re.search(r"(?:1|I|V)\s*/\s*2\s*PRICE", heading):
        return 0.50
    match = re.search(r"\b(\d{1,2})\s*%\s*OFF\b", heading)
    if match:
        rate = int(match.group(1)) / 100
        return rate if rate > 0 else None
    if any(
        re.fullmatch(r"PRICE", re.sub(r"[^A-Z]", "", line.text.upper()))
        and line.bottom - line.top >= 25
        for line in lines
    ):
        return 0.50
    return None


def classify_promo(save: float | None, regular: float | None, raw_text: str) -> str:
    normalised = normalise_offer(raw_text)
    explicit_half_price = bool(
        re.search(r"(?:1|I|V)\s*/\s*2\s*PRICE", normalised)
    )
    if explicit_half_price or (
        save is not None and regular and abs(save / regular - 0.5) <= 0.005
    ):
        return "half_price"
    if "40% OFF" in normalised:
        return "40_percent_off"
    if save is not None:
        return "save_amount"
    return "special"


def validate_product(product: Product) -> Product:
    """Attach conservative provenance and decide whether a row is trusted.

    Arithmetic consistency alone is not verification: two OCR mistakes can
    still form a plausible equation. A trusted row therefore needs at least
    two independent supports for the advertised special price, such as
    SAVE/WAS arithmetic plus the price circle or printed unit pricing.
    """
    reasons = [reason for reason in product.review_reason.split(";") if reason]

    def add_reason(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    evidence = {
        item for item in product.price_evidence.split(";") if item
    }
    special = product.special_price
    regular = product.regular_price
    save = product.save_amount

    raw_save = money_after("SAVE", product.offer_text)
    raw_regular = money_after("WAS", product.offer_text)
    if (
        raw_save is not None
        and raw_regular is not None
        and 0 < raw_save < raw_regular
        and special is not None
        and abs(special - (raw_regular - raw_save)) <= 0.011
        and regular is not None
        and abs(regular - raw_regular) <= 0.011
    ):
        evidence.add("save_was_arithmetic")

    unit_estimate = unit_price_estimate(product.product_name, product.offer_text)
    if unit_estimate is not None and special is not None:
        estimated_price, uncertainty = unit_estimate
        tolerance = max(0.10, uncertainty * 2)
        if uncertainty <= 0.06 and abs(special - estimated_price) <= tolerance:
            evidence.add("unit_price")
        elif uncertainty <= 0.06 and abs(special - estimated_price) > tolerance:
            add_reason("unit_price_conflict")

    if special is None:
        add_reason("special_price_missing")
    elif not 0 < special <= 1000:
        add_reason("special_price_implausible")
    if regular is not None and regular <= 0:
        add_reason("regular_price_implausible")
    if save is not None and save < 0:
        add_reason("save_amount_implausible")
    if special is not None and regular is not None and special >= regular:
        add_reason("special_not_below_regular")
    if None not in (special, regular, save):
        if abs(special + save - regular) > 0.011:
            add_reason("price_arithmetic_conflict")
        expected_discount = round(save / regular * 100, 2) if regular else None
        if (
            expected_discount is not None
            and product.discount_percent is not None
            and abs(product.discount_percent - expected_discount) > 0.011
        ):
            add_reason("discount_arithmetic_conflict")

    if product.promo_type == "half_price":
        explicit_half_price = bool(
            re.search(
                r"(?:1|I|V)\s*/\s*2\s*PRICE",
                normalise_offer(product.offer_text),
            )
        )
        exact_half = bool(
            save is not None
            and regular
            and abs(save / regular - 0.5) <= 0.005
        )
        if not explicit_half_price and not exact_half:
            add_reason("promo_type_conflict")

    if product.ocr_confidence < 80:
        add_reason("low_name_confidence_strict")
    if re.search(
        r"\b(?:EXCLUDES?|T&CS|BONUS)\b|(?:-|/|\bOR|\bAND)\s*$",
        product.product_name,
        flags=re.IGNORECASE,
    ):
        add_reason("product_name_suspicious")

    strong_evidence = evidence & {
        "save_was_arithmetic", "displayed_price", "unit_price"
    }
    weak_inference = bool(evidence & {"page_discount", "grid_inference"})
    if special is not None and len(strong_evidence) < 2:
        add_reason(
            "inferred_price_requires_review"
            if weak_inference else "single_price_evidence"
        )

    if "save_was_arithmetic" in evidence:
        price_source = "save_was_arithmetic"
    elif "displayed_price" in evidence:
        price_source = "displayed_price"
    elif "unit_price" in evidence:
        price_source = "unit_price"
    elif "page_discount" in evidence:
        price_source = "page_discount_inference"
    elif "grid_inference" in evidence:
        price_source = "grid_inference"
    else:
        price_source = "unknown"

    product.price_source = price_source
    product.price_evidence = ";".join(sorted(evidence))
    product.evidence_count = len(strong_evidence)
    product.review_reason = ";".join(reasons)
    product.needs_review = bool(reasons)
    product.verification_status = "verified" if not reasons else "review"
    return product


def invalid_product_reason(product: Product) -> str:
    """Return why a row cannot represent a usable advertised price.

    Unit-price disagreement is intentionally not a rejection condition. A
    multi-buy headline is a bundle total while its unit price describes one
    item, so both values can legitimately differ.
    """
    special = product.special_price
    regular = product.regular_price
    save = product.save_amount
    if special is None:
        return "special_price_missing"
    if not 0 < special <= 1000:
        return "special_price_implausible"
    if regular is not None and regular <= 0:
        return "regular_price_implausible"
    if save is not None and save < 0:
        return "save_amount_implausible"
    if regular is not None and special >= regular:
        return "special_not_below_regular"
    if (
        regular is not None
        and save is not None
        and abs(special + save - regular) > 0.011
    ):
        return "price_arithmetic_conflict"
    return ""


def usable_products(products: Iterable[Product]) -> list[Product]:
    """Keep valid priced offers, including otherwise valid unit conflicts."""
    return [product for product in products if not invalid_product_reason(product)]


def extract_products(
    words: list[Word], metadata: dict, page: int, page_url: str,
    catalogue_url: str, image_width: int,
) -> list[Product]:
    lines = make_lines(words)
    if excluded_page_reason(lines):
        return []
    products: list[Product] = []
    offer_pairs = pair_offer_lines(lines)
    heading_discount = page_discount_rate(lines)
    explicit_half_price_page = any(
        re.search(r"(?:1|I|V)\s*/\s*2\s*PRICE", normalise_offer(line.text))
        for line in lines
    )
    for save_line, was_line in offer_pairs:
        save = money_after("SAVE", save_line.text)
        regular = money_after("WAS", was_line.text) if was_line else None
        save_from_label = save is not None
        regular_from_label = regular is not None
        # If SAVE OCR failed but the line itself is WAS, retain the regular price.
        if save_line is was_line and regular is None:
            regular = money_after("WAS", save_line.text)
        # A frequent OCR substitution is SAVE$2.25 -> SAVE52.25. If the parsed
        # saving exceeds the printed WAS price, remove that spurious leading 5.
        if save is not None and regular is not None and save > regular:
            save_text = f"{save:g}"
            if save_text.startswith("5") and len(save_text) > 1:
                try:
                    corrected = float(save_text[1:])
                    if 0 < corrected <= regular:
                        save = round(corrected, 2)
                except ValueError:
                    pass
        special = round(regular - save, 2) if regular is not None and save is not None else None
        if special is not None and special <= 0:
            special = None
        if (
            special is not None and regular is not None and save is not None
            and save / regular < 0.10
        ):
            displayed_check = displayed_special_price(save_line, lines, regular)
            if (
                displayed_check is not None
                and displayed_check < regular
                and abs(displayed_check - special) > max(0.25, regular * 0.10)
            ):
                special = displayed_check
                save = round(regular - special, 2)
        labels_produced_price = special is not None
        # If one yellow-label value was missed, use the prominent price circle
        # above it and derive the remaining value. This also recovers offers
        # where only WAS survived OCR.
        if special is None:
            displayed = displayed_special_price(save_line, lines, regular)
            if displayed is not None:
                special = displayed
                labels_produced_price = True
                if regular is not None and save is None:
                    save = round(regular - special, 2)
                elif save is not None and regular is None:
                    regular = round(special + save, 2)
            elif explicit_half_price_page and save is not None and regular is None:
                special = save
                regular = round(save * 2, 2)
                labels_produced_price = True
        name, nearby, name_confidence = nearby_product_text(
            save_line, was_line, lines, image_width
        )
        if not name:
            continue
        if (
            name_confidence < 60
            and len(re.sub(r"[^A-Za-z]", "", name)) < 8
        ):
            continue
        displayed_evidence = displayed_special_price(save_line, lines, regular)
        is_multibuy = multibuy_near(save_line, lines)
        reconciliation_input = special
        unit_evidence = unit_price_estimate(name, nearby)
        if (
            save_from_label
            and not regular_from_label
            and save is not None
            and unit_evidence is not None
            and unit_evidence[1] <= 0.50
            and abs(save - unit_evidence[0]) <= max(0.011, unit_evidence[1])
        ):
            # On some half-price cards OCR labels the headline amount itself as
            # SAVE. If that amount agrees with the independent unit price, use
            # it instead of a fused circle reading such as 142.50.
            reconciliation_input = save
        reconciled = reconcile_special_price(
            reconciliation_input, displayed_evidence, name, nearby, is_multibuy
        )
        # A page heading is a useful fallback only when SAVE was not read. If
        # a row has its own SAVE value (mixed-promotion pages are common), that
        # row-specific evidence takes precedence over the page banner.
        heading_used = False
        if (
            heading_discount is not None
            and regular_from_label
            and not save_from_label
            and regular is not None
            and not is_multibuy
        ):
            heading_special = round(regular * (1 - heading_discount), 2)
            unit_estimate = unit_price_estimate(name, nearby)
            if (
                unit_estimate is None
                or abs(heading_special - unit_estimate[0])
                    <= max(0.15, unit_estimate[1] * 2)
            ):
                reconciled = heading_special
                heading_used = True
        if reconciled is not None and reconciled > 0:
            special = reconciled
            if regular_from_label and regular is not None and special < regular:
                save = round(regular - special, 2)
            elif save_from_label and save is not None:
                regular = round(special + save, 2)
        # When OCR found WAS but missed SAVE/the price circle, pack size and
        # unit price can reconstruct the advertised price. Only accept estimates
        # with low rounding uncertainty and sensible positive savings.
        if not labels_produced_price and regular is not None:
            unit_estimate = unit_price_estimate(name, nearby)
            unit_special = price_from_unit_text(name, nearby)
            if (
                unit_estimate is not None
                and unit_special is not None
                and unit_estimate[1] <= 0.06
                and 0 < unit_special < regular
            ):
                if special is None or abs(special - unit_special) > max(
                    0.10, unit_estimate[1] * 2
                ):
                    special = unit_special
                save = round(regular - special, 2)
        if re.search(r"\b(?:excludes|clearance|conditions apply)\b", name,
                     flags=re.IGNORECASE):
            continue
        combined = " | ".join(
            part for part in [save_line.text, was_line.text if was_line else "", nearby]
            if part
        )
        if excluded_offer_reason(name, combined):
            continue
        reasons: list[str] = []
        if not name or len(name) < 4:
            reasons.append("product_name_uncertain")
        if save is None:
            reasons.append("save_amount_missing")
        if regular is None:
            reasons.append("regular_price_missing")
        if special is None:
            reasons.append("special_price_missing")
        if name_confidence < 60:
            reasons.append("low_ocr_confidence")
        discount = round(save / regular * 100, 2) if save is not None and regular else None
        explicit_evidence: list[str] = []
        if (
            displayed_evidence is not None
            and special is not None
            and abs(displayed_evidence - special) <= 0.011
        ):
            explicit_evidence.append("displayed_price")
        if heading_used:
            explicit_evidence.append("page_discount")
        products.append(
            Product(
                retailer=metadata.get("store_name", ""),
                region=infer_region(metadata.get("title", "")),
                catalogue_title=metadata.get("title", ""),
                catalogue_start_date=iso_date(metadata.get("start_date")),
                catalogue_end_date=iso_date(metadata.get("end_date")),
                page_number=page,
                product_name=name,
                special_price=special,
                regular_price=regular,
                save_amount=save,
                discount_percent=discount,
                promo_type=classify_promo(save, regular, combined),
                pack_size=extract_pack_size(name),
                unit_price_text=extract_unit_price(nearby),
                offer_text=combined,
                ocr_confidence=round(name_confidence, 2),
                needs_review=bool(reasons),
                review_reason=";".join(reasons),
                source_page_image=page_url,
                source_catalogue_url=catalogue_url,
                price_evidence=";".join(explicit_evidence),
            )
        )

    existing_names = {product.product_name.casefold() for product in products}

    def already_recovered(name: str) -> bool:
        return any(names_overlap(name, existing) for existing in existing_names)

    # Red EVERY DAY/DOWN DOWN labels do not necessarily contain SAVE or WAS,
    # so treat each compact marker as an offer anchor in its own right.
    for marker in lines:
        marker_kind = promotion_marker_kind(marker)
        if not marker_kind:
            continue
        name, nearby, name_confidence = nearby_product_text(
            marker, None, lines, image_width, horizontal_radius=image_width / 8
        )
        if (
            not name
            or not plausible_recovered_name(name, name_confidence)
        ):
            continue
        displayed = displayed_special_price(marker, lines)
        special = choose_recovered_special(displayed, name, nearby)
        special = reconcile_special_price(
            special, displayed, name, nearby, multibuy_near(marker, lines)
        )
        if special is None or special <= 0:
            continue
        was_candidates = [
            line for line in lines
            if money_after("WAS", line.text) is not None
            and abs(line.cx - marker.cx) <= image_width / 8
            and marker.top <= line.top <= marker.bottom + 45
        ]
        regular = (
            money_after("WAS", min(was_candidates, key=lambda line: abs(line.cx - marker.cx)).text)
            if was_candidates else None
        )
        save = (
            round(regular - special, 2)
            if regular is not None and 0 < special < regular else None
        )
        if excluded_offer_reason(name, nearby):
            continue
        reasons: list[str] = []
        # Missing WAS/SAVE is expected for EVERY DAY pricing. DOWN DOWN should
        # normally expose its previous price, so retain a review flag if not.
        if marker_kind == "down_down" and regular is None:
            reasons.extend(["save_amount_missing", "regular_price_missing"])
        if name_confidence < 70:
            reasons.append("low_ocr_confidence")
        products.append(
            Product(
                retailer=metadata.get("store_name", ""),
                region=infer_region(metadata.get("title", "")),
                catalogue_title=metadata.get("title", ""),
                catalogue_start_date=iso_date(metadata.get("start_date")),
                catalogue_end_date=iso_date(metadata.get("end_date")),
                page_number=page,
                product_name=name,
                special_price=special,
                regular_price=regular,
                save_amount=save,
                discount_percent=(
                    round(save / regular * 100, 2) if save is not None and regular else None
                ),
                promo_type=marker_kind,
                pack_size=extract_pack_size(name),
                unit_price_text=extract_unit_price(nearby),
                offer_text=" | ".join(part for part in [marker.text, nearby] if part),
                ocr_confidence=round(name_confidence, 2),
                needs_review=bool(reasons),
                review_reason=";".join(reasons),
                source_page_image=page_url,
                source_catalogue_url=catalogue_url,
                price_evidence=(
                    "displayed_price"
                    if displayed is not None
                    and special is not None
                    and abs(displayed - special) <= 0.011
                    else ""
                ),
            )
        )
        existing_names.add(name.casefold())

    # Some red-on-yellow labels disappear completely in both OCR passes. On a
    # regular three-column offer grid, recover an empty cell only when at least
    # two other columns establish the row and confident product text plus a
    # usable displayed/unit price exists in the missing cell.
    cell_width = image_width / 3
    row_groups: list[list[OcrLine]] = []
    for anchor, _was in offer_pairs:
        group = next(
            (
                candidate for candidate in row_groups
                if abs(anchor.top - sum(line.top for line in candidate) / len(candidate)) <= 65
            ),
            None,
        )
        if group is None:
            row_groups.append([anchor])
        else:
            group.append(anchor)

    inferred_half_price = half_price_page(lines, products)
    existing_names = {product.product_name.casefold() for product in products}
    for row in row_groups:
        occupied_columns = {
            min(2, max(0, int(anchor.cx / cell_width))) for anchor in row
        }
        if len(occupied_columns) < 2:
            continue
        row_discounts: list[float] = []
        for row_anchor in row:
            matching_pair = next(
                (pair for pair in offer_pairs if pair[0] is row_anchor), None
            )
            if matching_pair is None:
                continue
            row_save = money_after("SAVE", matching_pair[0].text)
            row_regular = (
                money_after("WAS", matching_pair[1].text)
                if matching_pair[1] is not None else None
            )
            if row_save is not None and row_regular and 0 < row_save < row_regular:
                row_discounts.append(row_save / row_regular)
        row_is_half_price = (
            inferred_half_price
            and bool(row_discounts)
            and sum(abs(discount - 0.5) <= 0.03 for discount in row_discounts)
                / len(row_discounts) >= 0.8
        )
        row_top = round(sum(anchor.top for anchor in row) / len(row))
        # Rows containing red-label promotions are recovered from their marker
        # or price circle below. Grid interpolation can be horizontally offset
        # on mixed-width rows and splice the neighbouring product's pack size.
        if any(
            promotion_marker_kind(line)
            and abs(line.top - row_top) <= 100
            for line in lines
        ):
            continue
        # Catalogue cards sit slightly left of mathematical thirds. Estimate
        # that shared offset from the anchors which did survive OCR.
        offsets = [
            anchor.cx
            - min(2, max(0, int(anchor.cx / cell_width))) * cell_width
            for anchor in row
        ]
        base_center = sum(offsets) / len(offsets)
        for column in sorted({0, 1, 2} - occupied_columns):
            center = base_center + column * cell_width
            synthetic_anchor = OcrLine(
                text="", confidence=0.0,
                left=round(center - 40), top=row_top,
                right=round(center + 40), bottom=row_top + 30,
            )
            name, nearby, name_confidence = nearby_product_text(
                synthetic_anchor, None, lines, image_width,
                horizontal_radius=cell_width * 0.40,
            )
            if (
                not name
                or already_recovered(name)
                or name_confidence < 60
                or len(re.sub(r"[^A-Za-z]", "", name)) < 4
            ):
                continue
            displayed = displayed_special_price(synthetic_anchor, lines)
            unit_estimate = unit_price_estimate(name, nearby)
            # With no WAS value to constrain the estimate, preserve cent-level
            # prices (for example, $1.62 from 1.25 L at $1.30/L) rather than
            # forcing them to a five-cent promotion increment.
            unit_derived = (
                round(unit_estimate[0], 2) if unit_estimate is not None else None
            )
            if unit_estimate is not None and unit_estimate[1] > 0.06:
                unit_derived = None
            if displayed is not None and unit_derived is not None:
                special = (
                    displayed
                    if abs(displayed - unit_derived) <= max(0.10, unit_estimate[1] * 2)
                    else unit_derived
                )
            else:
                special = displayed if displayed is not None else unit_derived
            if special is None:
                continue
            if excluded_offer_reason(name, nearby):
                continue
            regular = round(special * 2, 2) if row_is_half_price else None
            save = special if row_is_half_price else None
            discount = 50.0 if row_is_half_price else None
            reasons = ["offer_labels_missing"]
            if row_is_half_price:
                reasons.append("regular_and_save_inferred_from_half_price")
            else:
                reasons.extend(["save_amount_missing", "regular_price_missing"])
            products.append(
                Product(
                    retailer=metadata.get("store_name", ""),
                    region=infer_region(metadata.get("title", "")),
                    catalogue_title=metadata.get("title", ""),
                    catalogue_start_date=iso_date(metadata.get("start_date")),
                    catalogue_end_date=iso_date(metadata.get("end_date")),
                    page_number=page,
                    product_name=name,
                    special_price=special,
                    regular_price=regular,
                    save_amount=save,
                    discount_percent=discount,
                    promo_type="half_price" if row_is_half_price else "special",
                    pack_size=extract_pack_size(name),
                    unit_price_text=extract_unit_price(nearby),
                    offer_text=nearby,
                    ocr_confidence=round(name_confidence, 2),
                    needs_review=True,
                    review_reason=";".join(reasons),
                    source_page_image=page_url,
                    source_catalogue_url=catalogue_url,
                    price_evidence=";".join(
                        part for part in [
                            "displayed_price"
                            if displayed is not None
                            and abs(displayed - special) <= 0.011 else "",
                            "grid_inference",
                        ] if part
                    ),
                )
            )
            existing_names.add(name.casefold())

    # Finally recover price-circle offers on rows with no usable labels at all
    # (common on EVERY DAY pages). Large, short numeric OCR lines are used only
    # when confident product text appears immediately below in the same narrow
    # column, which avoids treating packaging numbers as offers.
    top_banner_text = " ".join(line.text.upper() for line in lines if line.top < 250)
    page_marker_kind = (
        "everyday" if "EVERY DAY" in top_banner_text
        else "down_down" if "DOWN DOWN" in top_banner_text
        else ""
    )
    large_price_lines = [
        line for line in lines
        if line.top >= 250
        and line.bottom - line.top >= 35
        and len(re.sub(r"\s+", "", line.text)) <= 8
        and re.search(r"\d", line.text)
        and len(re.findall(r"[A-Za-z]", line.text)) <= 1
    ]
    for price_line in large_price_lines:
        synthetic_anchor = OcrLine(
            text="", confidence=price_line.confidence,
            left=round(price_line.cx - 40), top=price_line.bottom + 8,
            right=round(price_line.cx + 40), bottom=price_line.bottom + 48,
        )
        name, nearby, name_confidence = nearby_product_text(
            synthetic_anchor, None, lines, image_width,
            horizontal_radius=image_width / 8,
        )
        if (
            not name
            or not plausible_recovered_name(name, name_confidence)
        ):
            continue
        if (
            price_line.confidence < 20
            and unit_price_estimate(name, nearby) is None
            and not page_marker_kind
            and price_line.text.lstrip().startswith("(")
        ):
            continue
        displayed = displayed_special_price(synthetic_anchor, lines)
        special = choose_recovered_special(displayed, name, nearby)
        special = reconcile_special_price(
            special,
            displayed,
            name,
            nearby,
            multibuy_near(synthetic_anchor, lines),
        )
        if special is None or not 0 < special <= 1000:
            continue
        local_markers = [
            line for line in lines
            if promotion_marker_kind(line)
            and abs(line.cx - synthetic_anchor.cx) <= image_width / 8
            and price_line.bottom <= line.top <= synthetic_anchor.bottom + 20
        ]
        row_marker_kinds = {
            promotion_marker_kind(line)
            for line in lines
            if promotion_marker_kind(line)
            and price_line.bottom - 20 <= line.top <= synthetic_anchor.bottom + 20
        }
        marker_kind = promotion_marker_kind(local_markers[0]) if local_markers else ""
        if not marker_kind and len(row_marker_kinds) == 1:
            marker_kind = next(iter(row_marker_kinds))
        if not marker_kind:
            marker_kind = page_marker_kind
        was_candidates = [
            line for line in lines
            if money_after("WAS", line.text) is not None
            and abs(line.cx - synthetic_anchor.cx) <= image_width / 8
            and price_line.bottom <= line.top <= synthetic_anchor.bottom + 25
        ]
        regular = (
            money_after("WAS", min(was_candidates, key=lambda line: line.top).text)
            if was_candidates else None
        )
        # If OCR concatenated the large dollar digit with cents, use a nearby
        # WAS label to select a plausible scaled five-cent value. This repairs
        # readings such as 42.2 for a visible $4.20 price beside WAS $6.
        if regular is not None and special >= regular:
            estimate = unit_price_estimate(name, nearby)
            unit_supports_special = bool(
                estimate
                and estimate[1] <= 0.06
                and abs(special - estimate[0]) <= max(0.10, estimate[1] * 2)
            )
            if unit_supports_special:
                regular = None
            else:
                scaled_candidates = [
                    round(math.floor((special / divisor) * 20 + 0.5) / 20, 2)
                    for divisor in (10, 100)
                    if 0 < special / divisor < regular
                ]
                if scaled_candidates:
                    special = max(scaled_candidates)
        save = (
            round(regular - special, 2)
            if regular is not None and 0 < special < regular else None
        )
        if excluded_offer_reason(name, nearby):
            continue
        reasons: list[str] = []
        estimate = unit_price_estimate(name, nearby)
        price_backed_by_unit = bool(
            estimate
            and estimate[1] <= 0.06
            and abs(special - estimate[0]) <= max(0.10, estimate[1] * 2)
        )
        if not marker_kind and not price_backed_by_unit:
            reasons.append("offer_labels_missing")
        if (
            marker_kind != "everyday"
            and regular is None
            and not price_backed_by_unit
        ):
            reasons.extend(["save_amount_missing", "regular_price_missing"])
        products.append(
            Product(
                retailer=metadata.get("store_name", ""),
                region=infer_region(metadata.get("title", "")),
                catalogue_title=metadata.get("title", ""),
                catalogue_start_date=iso_date(metadata.get("start_date")),
                catalogue_end_date=iso_date(metadata.get("end_date")),
                page_number=page,
                product_name=name,
                special_price=special,
                regular_price=regular,
                save_amount=save,
                discount_percent=(
                    round(save / regular * 100, 2) if save is not None and regular else None
                ),
                promo_type=(
                    classify_promo(save, regular, nearby)
                    if regular is not None else marker_kind or "special"
                ),
                pack_size=extract_pack_size(name),
                unit_price_text=extract_unit_price(nearby),
                offer_text=nearby,
                ocr_confidence=round(name_confidence, 2),
                needs_review=bool(reasons),
                review_reason=";".join(reasons),
                source_page_image=page_url,
                source_catalogue_url=catalogue_url,
                price_evidence=(
                    "displayed_price"
                    if displayed is not None
                    and abs(displayed - special) <= 0.011
                    else ""
                ),
            )
        )
        existing_names.add(name.casefold())
    validated = deduplicate_products(
        [validate_product(product) for product in products]
    )
    return usable_products(validated)


def page_numbers(spec: str, total: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(1, total + 1))
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            result.update(range(start, end + 1))
        elif part:
            result.add(int(part))
    invalid = [page for page in result if page < 1 or page > total]
    if invalid:
        raise ValueError(f"Pages outside catalogue range 1-{total}: {invalid}")
    return sorted(result)


def write_outputs(
    products: Iterable[Product],
    output_base: Path,
    metadata: dict,
    run_info: dict | None = None,
) -> None:
    rows = [asdict(product) for product in usable_products(products)]
    output_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_base.with_suffix(".csv")
    json_path = output_base.with_suffix(".json")
    fields = list(asdict(Product("", "", "", "", "", 0, "", None, None,
                                 None, None, "", "", "", "", 0, False,
                                 "", "", "")).keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "metadata": metadata,
        "run": run_info or {},
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "product_count": len(rows),
        "products": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_products_output(
    json_path: Path, expected_pages_spec: str | None = None
) -> list[Product]:
    """Read products from one of this scraper's JSON outputs for batch resume."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("products")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid product output: {json_path}")
    if (
        expected_pages_spec is not None
        and payload.get("run", {}).get("pages_spec") != expected_pages_spec
    ):
        raise ValueError("saved output was created with a different --pages value")
    if payload.get("run", {}).get("validation_version") != VALIDATION_VERSION:
        raise ValueError("saved output uses an older validation version")
    try:
        return usable_products(
            [validate_product(Product(**row)) for row in rows]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Incompatible product output: {json_path}") from exc


def write_batch_manifest(manifest: list[dict], output_dir: Path) -> None:
    """Persist batch progress after each catalogue so interrupted runs resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "catalogue_number", "viewer_url", "status", "product_count",
        "review_count", "trusted_count", "output_csv", "error",
    ]
    csv_path = output_dir / "batch_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    (output_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def trusted_products(products: Iterable[Product]) -> list[Product]:
    """Return only rows supported strongly enough for precision-first use."""
    return [
        product for product in products
        if product.verification_status == "verified" and not product.needs_review
    ]


def write_quality_report(products: Iterable[Product], path: Path) -> None:
    """Summarise validation outcomes without hiding rejected rows."""
    rows = list(products)
    statuses: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for product in rows:
        statuses[product.verification_status] = (
            statuses.get(product.verification_status, 0) + 1
        )
        for reason in product.review_reason.split(";"):
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
    payload = {
        "validation_version": VALIDATION_VERSION,
        "total_rows": len(rows),
        "trusted_rows": len(trusted_products(rows)),
        "status_counts": dict(sorted(statuses.items())),
        "review_reason_counts": dict(
            sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def scrape_catalogue(
    catalogue_url: str,
    pages_spec: str,
    work_root: Path,
    delay: float,
    keep_images: bool,
    progress_prefix: str = "",
) -> tuple[list[Product], dict]:
    """Scrape one catalogue while keeping page cache isolation by slug."""
    store, slug = parse_catalogue_url(catalogue_url)
    metadata = get_metadata(store, slug)
    total = int(metadata["page_count"])
    selected_pages = page_numbers(pages_spec, total)
    work_dir = work_root / store / slug
    work_dir.mkdir(parents=True, exist_ok=True)
    products: list[Product] = []
    prefix = f"{progress_prefix} " if progress_prefix else ""
    print(f"{prefix}Catalogue: {metadata['title']} ({total} pages)")
    for index, page in enumerate(selected_pages, start=1):
        page_url = image_url(store, slug, page)
        image_path = work_dir / f"{page}.jpg"
        if not image_path.exists():
            image_path.write_bytes(request_bytes(page_url))
            time.sleep(max(0, delay))
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                width = image.width
        except Exception:
            width = 960
        words = run_tesseract(image_path)
        page_products = extract_products(
            words, metadata, page, page_url, catalogue_url, width
        )
        products.extend(page_products)
        print(
            f"{prefix}[{index}/{len(selected_pages)}] page {page}: "
            f"{len(page_products)} offers"
        )
    if not keep_images:
        shutil.rmtree(work_dir, ignore_errors=True)
    return products, metadata


def run_batch(args: argparse.Namespace, catalogue_urls: list[str]) -> int:
    """Process every link independently and also produce combined outputs."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_products: list[Product] = []
    manifest: list[dict] = []
    failures = 0
    for index, catalogue_url in enumerate(catalogue_urls, start=1):
        store, slug = parse_catalogue_url(catalogue_url)
        output_base = catalogue_output_base(output_dir, store, slug)
        output_json = output_base.with_suffix(".json")
        status = "completed"
        error = ""
        try:
            products: list[Product] | None = None
            if output_json.is_file() and not args.force:
                try:
                    products = read_products_output(output_json, args.pages)
                    status = "resumed"
                    print(
                        f"[{index}/{len(catalogue_urls)}] Reusing {len(products)} rows "
                        f"for {slug}"
                    )
                except ValueError as exc:
                    print(
                        f"[{index}/{len(catalogue_urls)}] Rebuilding {slug}: {exc}"
                    )
            if products is None:
                products, metadata = scrape_catalogue(
                    catalogue_url=catalogue_url,
                    pages_spec=args.pages,
                    work_root=Path(args.work_dir),
                    delay=args.delay,
                    keep_images=args.keep_images,
                    progress_prefix=f"[{index}/{len(catalogue_urls)}]",
                )
                write_outputs(
                    products,
                    output_base,
                    metadata,
                    {
                        "pages_spec": args.pages,
                        "validation_version": VALIDATION_VERSION,
                    },
                )
            all_products.extend(products)
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            products = []
            failures += 1
            status = "failed"
            error = str(exc)
            print(
                f"[{index}/{len(catalogue_urls)}] Error for {catalogue_url}: {exc}",
                file=sys.stderr,
            )
        manifest.append(
            {
                "catalogue_number": index,
                "viewer_url": catalogue_url,
                "status": status,
                "product_count": len(products),
                "review_count": sum(product.needs_review for product in products),
                "trusted_count": len(trusted_products(products)),
                "output_csv": str(output_base.with_suffix(".csv")),
                "error": error,
            }
        )
        write_batch_manifest(manifest, output_dir)

    batch_metadata = {
        "source_links_file": str(Path(args.links_file)),
        "catalogue_count": len(catalogue_urls),
        "successful_catalogue_count": len(catalogue_urls) - failures,
        "failed_catalogue_count": failures,
        "pages_spec": args.pages,
        "validation_version": VALIDATION_VERSION,
    }
    batch_run_info = {
        "pages_spec": args.pages,
        "validation_version": VALIDATION_VERSION,
    }
    write_outputs(
        all_products,
        output_dir / "all_catalogue_products",
        batch_metadata,
        batch_run_info,
    )
    verified_products = trusted_products(all_products)
    write_outputs(
        verified_products,
        output_dir / "trusted_catalogue_products",
        batch_metadata,
        batch_run_info,
    )
    write_quality_report(all_products, output_dir / "quality_report.json")
    review_count = sum(product.needs_review for product in all_products)
    print(
        f"Batch complete: {len(all_products)} rows from "
        f"{len(catalogue_urls) - failures}/{len(catalogue_urls)} catalogues; "
        f"{len(verified_products)} trusted; {review_count} flagged for review"
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "url", nargs="?", help="One Catalogue AU URL containing #catalogue=..."
    )
    parser.add_argument(
        "--links-file",
        help="CSV/text dataset containing catalogue URLs; processes every unique link",
    )
    parser.add_argument("--output", default="coles_catalogue_products",
                        help=(
                            "Single mode: output path without extension. Batch mode: "
                            "output directory (default: %(default)s)"
                        ))
    parser.add_argument("--pages", default="all",
                        help="all, a single page, or ranges such as 2-10,15")
    parser.add_argument("--work-dir", default=".catalogue_pages",
                        help="Downloaded-page cache directory")
    parser.add_argument("--delay", type=float, default=0.35,
                        help="Delay between new page downloads in seconds")
    parser.add_argument("--keep-images", action="store_true",
                        help="Keep downloaded images after extraction")
    parser.add_argument(
        "--force", action="store_true",
        help="Batch mode: reprocess catalogues that already have a JSON output",
    )
    args = parser.parse_args()

    if bool(args.url) == bool(args.links_file):
        parser.error("provide either one URL or --links-file, but not both")
    if not shutil.which("tesseract"):
        print("Error: tesseract is required but was not found on PATH.", file=sys.stderr)
        return 2
    try:
        if args.links_file:
            catalogue_urls = load_catalogue_urls(Path(args.links_file))
            print(f"Loaded {len(catalogue_urls)} unique catalogue links")
            return run_batch(args, catalogue_urls)
        products, metadata = scrape_catalogue(
            catalogue_url=args.url,
            pages_spec=args.pages,
            work_root=Path(args.work_dir),
            delay=args.delay,
            keep_images=args.keep_images,
        )
        single_run_info = {
            "pages_spec": args.pages,
            "validation_version": VALIDATION_VERSION,
        }
        output_base = Path(args.output)
        write_outputs(products, output_base, metadata, single_run_info)
        verified_products = trusted_products(products)
        write_outputs(
            verified_products,
            output_base.parent / f"{output_base.name}_trusted",
            metadata,
            single_run_info,
        )
        write_quality_report(
            products,
            output_base.parent / f"{output_base.name}_quality_report.json",
        )
        review_count = sum(product.needs_review for product in products)
        print(
            f"Wrote {len(products)} rows; {len(verified_products)} trusted; "
            f"{review_count} flagged for review"
        )
        return 0
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
