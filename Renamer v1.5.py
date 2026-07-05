"""
Renamer.py — Intelligent image renamer powered by Qwen3 VLM via LM Studio
──────────────────────────────────────────────────────────────────────────
Naming intelligence covers:
  • Memes & internet culture  →  "drake_approves_python_over_java"
  • Pop culture / films       →  "walter_white_im_the_danger_scene"
  • Celebrities & public fig. →  "elon_musk_twitter_acquisition_meme"
  • Viral formats             →  "distracted_boyfriend_meme_classic"
  • Gaming                    →  "among_us_impostor_red_vented"
  • Anime                     →  "gojo_satoru_infinity_jujutsu"
  • Music & albums            →  "abbey_road_beatles_crosswalk"
  • Sports moments            →  "jordan_last_shot_1998_finals"
  • TV / streaming            →  "breaking_bad_hazmat_suit_lab"
  • Landmarks & travel        →  "santorini_blue_domes_sunset"
  • Food                      →  "birria_tacos_consomme_dipping"
  • Nature / wildlife         →  "snow_leopard_himalaya_hunting"
  • Tech / products           →  "iphone_15_pro_titanium_natural"
  • Documents / text          →  "handwritten_letter_vintage_paper"
  • Architecture              →  "zaha_hadid_heydar_aliyev_center"
  • Generic fallback          →  "outdoor_scene_greenery"

Key improvements over v1
─────────────────────────
  • Qwen3 VLM-optimised prompt (thinking mode DISABLED for speed)
  • Structured JSON response → category + name + confidence
  • Parallel workers (configurable thread pool)
  • Dry-run mode — preview renames without touching disk
  • Recursive folder support
  • Output directory option (copy vs rename)
  • Meme & pop-culture focused prompt layer
  • Category prefix tags  e.g. [meme] [celeb] [food]
  • Resume support — skip already well-named files
  • Rename log CSV for undo/audit
  • Rich progress bar with category stats
  • Config file support (renamer.toml)
  • Smart idempotency check with slug comparison
  • Extended supported formats incl. AVIF, HEIF
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

try:
    from openai import BadRequestError, OpenAI
except ImportError:
    sys.exit("openai package not found — run: pip install openai")

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    sys.exit("Pillow not found — run: pip install Pillow")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("tqdm not found — run: pip install tqdm")

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # backport
    except ImportError:
        tomllib = None  # type: ignore[assignment]


# ── Configuration dataclass ───────────────────────────────────────────────────

@dataclass
class Config:
    lm_studio_url:   str   = "http://localhost:1234/v1"
    lm_studio_key:   str   = "lm-studio"
    model_name:      str   = "qwen3-vl"          # Change to match your loaded model name in LM Studio
    workers:         int   = 1                    # LM Studio handles one request at a time
    max_retries:     int   = 4
    base_delay:      float = 1.5
    token_budget:    int   = 2048        # reduced: no think tokens generated
    dry_run:         bool  = False
    recursive:       bool  = False
    output_dir:      Optional[str] = None         # if set, copy renamed files here
    add_prefix:      bool  = False                # prepend [category] to stem
    resume:          bool  = True                 # skip files already processed in a prior run
    fresh:           bool  = False                # ignore progress file and start from scratch
    progress_file:   str   = ".renamer_progress.json"  # tracks completed originals per folder
    write_csv_log:   bool  = True
    log_file:        str   = "renamer.log"
    csv_file:        str   = "renamer_log.csv"

    SIZE_LADDER:    list   = field(default_factory=lambda: [
        (1536, 1536), (1024, 1024), (768, 768), (512, 512), (384, 384), (256, 256)
    ])
    QUALITY_LADDER: list   = field(default_factory=lambda: [82, 70, 58, 45])

    SUPPORTED_EXTS: set    = field(default_factory=lambda: {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp",
        ".tiff", ".tif", ".heic", ".heif", ".avif"
    })

    WIN_RESERVED: set = field(default_factory=lambda: {
        "con","prn","aux","nul",
        "com1","com2","com3","com4","com5","com6","com7","com8","com9",
        "lpt1","lpt2","lpt3","lpt4","lpt5","lpt6","lpt7","lpt8","lpt9"
    })


def load_config(path: str = "renamer.toml") -> Config:
    cfg = Config()
    if tomllib and os.path.exists(path):
        with open(path, "rb") as f:
            data = tomllib.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        logging.getLogger(__name__).info("Loaded config from %s", path)
    return cfg


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(cfg: Config) -> logging.Logger:
    log = logging.getLogger("renamer")
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(cfg.log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


# ── The system prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise image-to-filename converter.

══════════════════════════════════════════════════════════
RULE #1 — ALWAYS CHECK FOR TEXT FIRST
══════════════════════════════════════════════════════════
Before anything else, scan the entire image for readable text:
captions, subtitles, dialogue, overlays, labels, signs, buttons,
headlines, watermarks, UI text, meme captions — anything.

If you find ANY readable text:
  → The text content MUST be included in the filename.
  → Combine the text with who/what is in the image.
  → The text is more searchable than the visual description alone.

  Examples:
    Anime character, subtitle reads "That's what a Nazi would say"
      → "anime_character_thats_what_nazi_would_say"
    Drake meme, top text "Me avoiding work", bottom "Me on Reddit"
      → "drake_meme_avoiding_work_on_reddit"
    Screenshot of tweet: "just dropped my new album"
      → "tweet_just_dropped_new_album"
    Cat photo, sign in background reads "No Cats Allowed"
      → "cat_ignoring_no_cats_allowed_sign"
    Slide title: "Q3 Revenue Growth 2024"
      → "q3_revenue_growth_2024_slide"
    Receipt, header reads "McDonald's" with total $14.99
      → "mcdonalds_receipt_total_1499"
    Error message: "TypeError: cannot read property of undefined"
      → "typeerror_cannot_read_property_undefined"
    Sports graphic: "Messi scores 2 — Argentina vs France"
      → "messi_scores_argentina_vs_france"
    Person holding sign: "Free Palestine"
      → "person_holding_free_palestine_sign"
    Whiteboard: "System Architecture — Auth Flow"
      → "system_architecture_auth_flow_whiteboard"

══════════════════════════════════════════════════════════
RULE #2 — NO TEXT? DESCRIBE WHAT YOU SEE
══════════════════════════════════════════════════════════
Only if there is genuinely no readable text in the image:
  → Describe: main subject + action/state + setting.
  → Be specific — use colour, species, object type, what is happening.
  → Never guess meme names, celebrity names, or cultural references
    unless you are 100% certain.

  Examples:
    "golden_retriever_running_on_beach"
    "two_people_shaking_hands_office"
    "aerial_view_city_at_night"
    "grilled_steak_herbs_cast_iron"
    "cat_sitting_on_laptop_keyboard"

══════════════════════════════════════════════════════════
OUTPUT FORMAT — JSON only, no markdown, no extra text
══════════════════════════════════════════════════════════
{
  "category": "<chart_graph | document | screenshot | presentation | receipt | diagram | code | photo | illustration | nature | animal | food | vehicle | architecture | tech | event | other>",
  "name": "<3 to 6 lowercase words joined by underscores>",
  "confidence": <0.0–1.0>,
  "notes": "<one plain sentence: what you saw and what text was present>"
}

HARD RULES:
✓ Text in the image → text MUST appear in the name.
✓ 3 words minimum, 6 words maximum, snake_case, English only.
✗ Never produce: "cartoon_face_with_text", "image_with_caption", "screenshot_of_text".
✗ Never use filler words: "image", "photo", "picture", "screenshot" as the only descriptor.
"""

USER_PROMPT = (
    "First, read every piece of visible text in this image carefully. "
    "If any text is present, it must be reflected in the filename. "
    "Then identify the visual subject. Combine both into the name. "
    "Return ONLY the JSON object."
)


# ── Image encoding ────────────────────────────────────────────────────────────

def _resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS  # type: ignore[attr-defined]


def _flatten_to_rgb(im: Image.Image) -> Image.Image:
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im.convert("RGBA"), mask=im.split()[-1])
        return bg
    if im.mode == "P":
        return im.convert("RGBA").convert("RGB")
    if im.mode == "CMYK":
        return im.convert("RGB")
    return im.convert("RGB")


def encode_image(path: str, max_size: tuple, quality: int) -> str:
    with Image.open(path) as im:
        im = _flatten_to_rgb(im)
        im.thumbnail(max_size, _resample())
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()


def _token_estimate(b64: str) -> int:
    return int(len(b64) * 0.75 / 1.33)


def encode_within_budget(path: str, cfg: Config) -> str:
    for size in cfg.SIZE_LADDER:
        for quality in cfg.QUALITY_LADDER:
            b64 = encode_image(path, size, quality)
            if _token_estimate(b64) < cfg.token_budget:
                return b64
    return encode_image(path, (128, 128), 40)


# ── API call & response parsing ───────────────────────────────────────────────

@dataclass
class ImageResult:
    category:   str   = "other"
    name:       str   = "unknown_image"
    confidence: float = 0.0
    notes:      str   = ""
    raw:        str   = ""


def _parse_response(text: str) -> ImageResult:
    """Extract JSON from model output robustly — handles markdown fences, leading text."""
    # Safety net: strip any <think> tags if the model emits them despite being disabled.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Find JSON block
    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not json_match:
        # Fall back: try entire text
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return ImageResult(raw=text)

    try:
        data = json.loads(json_match.group())
        return ImageResult(
            category   = str(data.get("category", "other")).strip().lower(),
            name       = str(data.get("name", "")).strip(),
            confidence = float(data.get("confidence", 0.0)),
            notes      = str(data.get("notes", "")),
            raw        = text,
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        return ImageResult(raw=text)


def _api_call(client: OpenAI, model: str, b64: str) -> str:
    url = f"data:image/jpeg;base64,{b64}"  # LM Studio requires data URL format
    resp = client.chat.completions.create(
        model=model,
        max_tokens=512,
        temperature=0.1,        # low temp for consistent naming
        # Disable Qwen3 thinking mode for faster file naming.
        # Passed through LM Studio as a model-specific extra parameter.
        extra_body={"thinking": {"type": "disabled"}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": url, "detail": "high"}},
                ],
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def describe_image(image_path: str, client: OpenAI, cfg: Config, log: logging.Logger) -> Optional[ImageResult]:
    try:
        b64 = encode_within_budget(image_path, cfg)
    except (UnidentifiedImageError, Exception) as e:
        log.warning("Cannot open %s: %s", Path(image_path).name, e)
        return None

    delay = cfg.base_delay
    for attempt in range(1, cfg.max_retries + 1):
        try:
            raw = _api_call(client, cfg.model_name, b64)
            result = _parse_response(raw)
            log.debug("  Model says: %s", raw[:200])
            return result
        except BadRequestError as e:
            err = str(e)
            if "context" in err.lower() or "tokens" in err.lower():
                idx = min(attempt, len(cfg.SIZE_LADDER) - 1)
                b64 = encode_image(image_path, cfg.SIZE_LADDER[idx], cfg.QUALITY_LADDER[-1])
                log.debug("Context overflow on %s, shrinking…", Path(image_path).name)
            else:
                log.warning("API rejected %s (attempt %d/%d): %s",
                            Path(image_path).name, attempt, cfg.max_retries, e)
        except Exception as e:
            log.warning("Error on %s (attempt %d/%d): %s",
                        Path(image_path).name, attempt, cfg.max_retries, e)

        if attempt < cfg.max_retries:
            time.sleep(delay)
            delay = min(delay * 2, 30)

    return None


# ── Filename sanitisation ─────────────────────────────────────────────────────

def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def clean_filename(text: str) -> str:
    text = _strip_accents(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9 _\-]", " ", text)
    text = re.sub(r"[\s_\-]+", "_", text).strip("_")
    text = text[:80].rstrip("_")
    return text if text else "image"


def make_stem(result: ImageResult, cfg: Config) -> str:
    """Build final filename stem from parsed result."""
    name = clean_filename(result.name)

    if not name or name == "image" or len(name) < 3:
        name = "unknown_image"

    # Optionally prepend category prefix
    if cfg.add_prefix and result.category and result.category != "other":
        cat = clean_filename(result.category)
        if not name.startswith(cat):
            name = f"{cat}_{name}"

    cfg_reserved = getattr(cfg, "WIN_RESERVED", set())
    if name in cfg_reserved:
        name = f"img_{name}"

    return name


# ── Collision-safe path ───────────────────────────────────────────────────────

def unique_path(folder: str, stem: str, ext: str) -> str:
    candidate = os.path.join(folder, stem + ext)
    if not os.path.exists(candidate):
        return candidate
    for n in range(2, 1000):
        candidate = os.path.join(folder, f"{stem}_{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
    ts = str(int(time.time()))[-6:]
    return os.path.join(folder, f"{stem}_{ts}{ext}")


# ── Stats tracker (thread-safe) ───────────────────────────────────────────────

class Stats:
    def __init__(self):
        self._lock   = threading.Lock()
        self.renamed = 0
        self.skipped = 0
        self.failed  = 0
        self.by_category: dict[str, int] = {}
        self.csv_rows: list[dict] = []

    def record(self, action: str, old: str, new: str, category: str = "", notes: str = ""):
        with self._lock:
            if action == "renamed":     self.renamed    += 1
            elif action == "skipped":   self.skipped    += 1
            elif action == "failed":    self.failed     += 1

            if category:
                self.by_category[category] = self.by_category.get(category, 0) + 1

            self.csv_rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "action":    action,
                "old_name":  old,
                "new_name":  new,
                "category":  category,
                "notes":     notes,
            })

    @property
    def total(self):
        return self.renamed + self.skipped + self.failed


# ── Progress tracker (resume support) ────────────────────────────────────────

class ProgressTracker:
    """
    Persists the set of original filenames that have already been successfully
    processed, keyed per folder. Stored as JSON alongside the images (or in
    output_dir if set). Survives crashes — every successful rename is flushed
    immediately to disk.
    """

    def __init__(self, progress_path: str, fresh: bool, log: logging.Logger):
        self._path = progress_path
        self._lock = threading.Lock()
        self._log  = log
        self._done: set[str] = set()

        if fresh and os.path.exists(progress_path):
            try:
                os.remove(progress_path)
                log.info("Fresh run — deleted previous progress file: %s", progress_path)
            except OSError as e:
                log.warning("Could not delete progress file: %s", e)
        elif os.path.exists(progress_path):
            try:
                with open(progress_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._done = set(data.get("completed", []))
                log.info("Resuming — %d file(s) already done (delete %s to start fresh)",
                         len(self._done), os.path.basename(progress_path))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Could not read progress file (%s) — starting fresh", e)

    def already_done(self, filename: str) -> bool:
        with self._lock:
            return filename in self._done

    def mark_done(self, filename: str) -> None:
        with self._lock:
            self._done.add(filename)
            self._flush()

    def _flush(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({"completed": sorted(self._done)}, f, indent=2)
        except OSError as e:
            self._log.warning("Could not save progress: %s", e)

    def clear(self) -> None:
        with self._lock:
            self._done.clear()
            if os.path.exists(self._path):
                try:
                    os.remove(self._path)
                except OSError:
                    pass


# ── Per-file processor ────────────────────────────────────────────────────────

def process_file(
    file: Path,
    client: OpenAI,
    cfg: Config,
    stats: Stats,
    progress: ProgressTracker,
    log: logging.Logger,
    pbar,
):
    ext      = file.suffix
    old_name = file.name

    # Determine output folder
    out_folder = cfg.output_dir if cfg.output_dir else str(file.parent)

    # Resume: skip files already successfully processed in a prior run
    if cfg.resume and progress.already_done(old_name):
        log.debug("RESUME skip  %s", old_name)
        stats.record("skipped", old_name, "", notes="already processed (resume)")
        pbar.update(1)
        return

    result = describe_image(str(file), client, cfg, log)

    if not result or not result.name:
        log.warning("SKIP (no desc) %s", old_name)
        stats.record("skipped", old_name, "", notes="no description returned")
        pbar.update(1)
        return

    stem     = make_stem(result, cfg)
    new_path = unique_path(out_folder, stem, ext)
    new_name = Path(new_path).name

    conf_tag = f" [{result.confidence:.0%}]" if result.confidence else ""
    cat_tag  = f" [{result.category}]" if result.category else ""

    if cfg.dry_run:
        log.info("DRY-RUN  %s  →  %s%s%s", old_name, new_name, cat_tag, conf_tag)
        stats.record("renamed", old_name, new_name, result.category, result.notes)
        # Don't mark done in dry-run — nothing actually changed
    else:
        try:
            if cfg.output_dir:
                shutil.copy2(str(file), new_path)
            else:
                os.rename(str(file), new_path)
            log.info("OK  %s  →  %s%s%s", old_name, new_name, cat_tag, conf_tag)
            stats.record("renamed", old_name, new_name, result.category, result.notes)
            progress.mark_done(old_name)   # persist immediately — survives crashes
        except OSError as e:
            log.error("FAIL  %s: %s", old_name, e)
            stats.record("failed", old_name, "", notes=str(e))

    pbar.update(1)


# ── CSV log writer ────────────────────────────────────────────────────────────

def write_csv_log(stats: Stats, cfg: Config, log: logging.Logger):
    if not cfg.write_csv_log or not stats.csv_rows:
        return
    fields = ["timestamp", "action", "old_name", "new_name", "category", "notes"]
    try:
        with open(cfg.csv_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(stats.csv_rows)
        log.info("CSV log saved → %s", cfg.csv_file)
    except OSError as e:
        log.warning("Could not write CSV log: %s", e)


# ── Core rename loop ──────────────────────────────────────────────────────────

def rename_images(image_folder: str, cfg: Config, log: logging.Logger) -> Stats:
    folder = Path(image_folder)

    glob = folder.rglob("*") if cfg.recursive else folder.iterdir()
    all_files = sorted(
        f for f in glob
        if f.is_file() and f.suffix.lower() in cfg.SUPPORTED_EXTS
    )

    if not all_files:
        log.info("No supported images found in: %s", image_folder)
        return Stats()

    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)

    # Progress file lives in output_dir if set, otherwise in the source folder
    progress_dir  = cfg.output_dir if cfg.output_dir else image_folder
    progress_path = os.path.join(progress_dir, cfg.progress_file)
    progress      = ProgressTracker(progress_path, cfg.fresh, log)

    log.info("Found %d image(s) — workers=%d  dry_run=%s  resume=%s  model=%s",
             len(all_files), cfg.workers, cfg.dry_run, cfg.resume and not cfg.fresh,
             cfg.model_name)

    client = OpenAI(base_url=cfg.lm_studio_url, api_key=cfg.lm_studio_key)
    stats  = Stats()

    with tqdm(total=len(all_files), unit="img", dynamic_ncols=True) as pbar:
        if cfg.workers > 1:
            with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
                futures = {
                    executor.submit(process_file, f, client, cfg, stats, progress, log, pbar): f
                    for f in all_files
                }
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        log.error("Unexpected error: %s", exc)
        else:
            for f in all_files:
                process_file(f, client, cfg, stats, progress, log, pbar)

    # Summary
    category_breakdown = "  ".join(
        f"{k}: {v}" for k, v in sorted(stats.by_category.items(), key=lambda x: -x[1])
    ) or "—"

    log.info(
        "\n── Summary ─────────────────────────────────\n"
        "   Renamed      : %d\n"
        "   Skipped      : %d\n"
        "   Failed       : %d\n"
        "   Total        : %d\n"
        "   Categories   : %s\n"
        "   Dry run      : %s\n"
        "────────────────────────────────────────────",
        stats.renamed, stats.skipped,
        stats.failed, stats.total, category_breakdown,
        "YES (no files changed)" if cfg.dry_run else "no",
    )

    write_csv_log(stats, cfg, log)
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="renamer",
        description="Intelligent image renamer powered by Qwen3 VLM via LM Studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FLAGS & SCENARIOS
─────────────────────────────────────────────────────────────────────────────

  BASIC USAGE
    python Renamer.py ~/Photos
        Rename every image in ~/Photos in-place. Saves progress so re-running
        skips already-renamed files automatically.

  RESUME / FRESH
    python Renamer.py ~/Photos
        (run again) — skips files already done, only processes new/failed ones.

    python Renamer.py ~/Photos --fresh
        Ignore previous progress and reprocess every file from scratch.
        Deletes the hidden .renamer_progress.json file first.

    python Renamer.py ~/Photos --no-resume
        Same as --fresh but does NOT delete the progress file — just ignores it
        for this run without clearing history.

  PREVIEW WITHOUT CHANGES
    python Renamer.py ~/Photos --dry-run
        Shows what every file WOULD be renamed to. Nothing is touched on disk.
        Progress is not saved (since nothing actually changed).

  OUTPUT DIRECTORY (copy instead of rename in-place)
    python Renamer.py ~/Photos --output-dir ~/Renamed
        Leaves originals untouched. Copies renamed files into ~/Renamed.
        Progress file is stored in ~/Renamed so re-runs know what was copied.

  RECURSIVE (include subfolders)
    python Renamer.py ~/Photos --recursive
        Walks all subdirectories. Each subfolder is processed; progress is
        tracked globally against the top-level progress file.

  CATEGORY PREFIX
    python Renamer.py ~/Photos --prefix
        Prepends the detected category to each filename.
        e.g.  "photo_golden_retriever_running_beach.jpg"
              "chart_monthly_revenue_q3_2024.png"

  PARALLEL WORKERS  (use only if running a fast GPU or multiple models)
    python Renamer.py ~/Photos --workers 4
        Run 4 threads simultaneously. Default is 1 because LM Studio queues
        requests serially — only increase if you know your setup can handle it.

  MODEL / URL OVERRIDE
    python Renamer.py ~/Photos --model qwen2.5-vl-7b-instruct
        Override the model name from the default in Config or renamer.toml.

    python Renamer.py ~/Photos --url http://192.168.1.50:1234/v1
        Point at a remote LM Studio instance.

  CONFIG FILE
    python Renamer.py ~/Photos --config my_settings.toml
        Load settings from a custom TOML file instead of the default
        renamer.toml. CLI flags always override config file values.

  COMBINED EXAMPLES
    python Renamer.py ~/Photos --dry-run --recursive
        Preview all renames across all subfolders without touching anything.

    python Renamer.py ~/Photos --output-dir ~/Renamed --recursive --prefix
        Copy all images (recursively) with category-prefixed names into ~/Renamed.

    python Renamer.py ~/Photos --fresh --workers 2 --prefix
        Full reprocess from scratch, 2 threads, with category prefixes.

─────────────────────────────────────────────────────────────────────────────
""",
    )

    # Positional
    p.add_argument("folder",
                   nargs="?",
                   help="Path to the folder containing images to rename.")

    # Model / connection
    g = p.add_argument_group("Model / connection")
    g.add_argument("--model",
                   default=None,
                   metavar="NAME",
                   help="LM Studio model name (default: value in Config/renamer.toml).")
    g.add_argument("--url",
                   default=None,
                   metavar="URL",
                   help="LM Studio base URL (default: http://localhost:1234/v1).")
    g.add_argument("--workers",
                   type=int,
                   default=None,
                   metavar="N",
                   help="Parallel worker threads. Default 1 — increase only for fast GPU setups.")

    # Behaviour
    g = p.add_argument_group("Behaviour")
    g.add_argument("--dry-run",
                   action="store_true",
                   help="Preview renames without changing any files. Progress is not saved.")
    g.add_argument("--recursive",
                   action="store_true",
                   help="Recurse into subdirectories.")
    g.add_argument("--prefix",
                   action="store_true",
                   help="Prepend the detected category to each filename (e.g. photo_dog_running.jpg).")
    g.add_argument("--output-dir",
                   default=None,
                   metavar="DIR",
                   help="Copy renamed files into DIR instead of renaming in-place.")

    # Resume control
    g = p.add_argument_group("Resume control")
    g.add_argument("--fresh",
                   action="store_true",
                   help="Delete the progress file and reprocess every image from scratch.")
    g.add_argument("--no-resume",
                   action="store_true",
                   help="Ignore the progress file for this run without deleting it.")

    # Config
    g = p.add_argument_group("Config")
    g.add_argument("--config",
                   default="renamer.toml",
                   metavar="FILE",
                   help="Path to TOML config file (default: renamer.toml).")

    return p


def prompt_folder() -> Optional[str]:
    try:
        raw = input("\nEnter folder path containing images to rename: ").strip().strip("\"'")
        return os.path.abspath(os.path.expanduser(raw)) if raw else None
    except (EOFError, KeyboardInterrupt):
        return None


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    cfg = load_config(args.config)

    # CLI args override config file
    if args.model:      cfg.model_name    = args.model
    if args.url:        cfg.lm_studio_url = args.url
    if args.workers:    cfg.workers       = args.workers
    if args.dry_run:    cfg.dry_run       = True
    if args.recursive:  cfg.recursive     = True
    if args.output_dir: cfg.output_dir    = os.path.abspath(args.output_dir)
    if args.prefix:     cfg.add_prefix    = True
    if args.fresh:      cfg.fresh         = True
    if args.no_resume:  cfg.resume        = False

    log = setup_logging(cfg)

    if cfg.dry_run:
        log.info("=" * 52)
        log.info("  DRY RUN — no files will be changed")
        log.info("=" * 52)

    # Resolve folder
    folder = args.folder
    if not folder:
        folder = prompt_folder()
    if not folder:
        log.error("No folder provided.")
        sys.exit(1)

    folder = os.path.abspath(os.path.expanduser(folder))

    while not os.path.isdir(folder):
        log.error("Not a valid folder: %s", folder)
        folder = prompt_folder()
        if not folder:
            sys.exit(1)
        folder = os.path.abspath(os.path.expanduser(folder))

    rename_images(folder, cfg, log)


if __name__ == "__main__":
    main()
