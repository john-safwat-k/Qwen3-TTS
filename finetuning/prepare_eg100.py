# Build train_raw.jsonl for Qwen3-TTS fine-tuning from
# ehabnegm/100-hour-Egyptian-dataset-single-speaker (CC-BY-NC-4.0 -> internal pilot only).
#
# Steps: find the metadata table -> detect columns -> filter (confidence, promo, duration)
#        -> resample every clip to 24 kHz mono wav -> pick one reference clip -> write jsonl.
#
# Usage:
#   hf download ehabnegm/100-hour-Egyptian-dataset-single-speaker --repo-type dataset --local-dir ~/data/eg100
#   python prepare_eg100.py --src ~/data/eg100 --out ~/data/eg100_24k --inspect      # look at columns first
#   python prepare_eg100.py --src ~/data/eg100 --out ~/data/eg100_24k                # build the jsonl
import argparse
import glob
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

TARGET_SR = 24000
ARABIC_RE = re.compile(r"[؀-ۿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"[0-9٠-٩]")
TATWEEL_RE = re.compile(r"ـ+")
BAD_CHARS_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")  # zero-width / bidi marks


def normalize_text(t):
    """Light normalisation only - do NOT convert Egyptian spelling to MSA."""
    t = BAD_CHARS_RE.sub("", str(t))
    t = TATWEEL_RE.sub("", t)
    t = t.replace("\u060c", "\u060c ").replace("\u061f", "\u061f ")  # space after Arabic comma / question mark
    t = re.sub(r"\s+", " ", t).strip()
    return t


def find_table(src):
    pats = ["**/*.parquet", "**/metadata*.csv", "**/*.csv", "**/metadata*.jsonl", "**/*.jsonl", "**/*.tsv"]
    for p in pats:
        files = sorted(glob.glob(os.path.join(src, p), recursive=True))
        if files:
            return files
    sys.exit(f"No metadata table (parquet/csv/jsonl/tsv) found under {src}")


def load_table(files):
    frames = []
    for f in files:
        if f.endswith(".parquet"):
            df = pd.read_parquet(f)
        elif f.endswith(".jsonl"):
            df = pd.read_json(f, lines=True)
        elif f.endswith(".tsv"):
            df = pd.read_csv(f, sep="\t")
        else:
            df = pd.read_csv(f)
        df["__table_dir"] = os.path.dirname(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pick(cols, *cands, required=True):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    for c in cands:  # substring match
        for lc, orig in low.items():
            if c in lc:
                return orig
    if required:
        sys.exit(f"Could not find a column like {cands} in {list(cols)} - pass it explicitly")
    return None


def load_audio(value, table_dir, src):
    """Returns (float32 mono waveform, sr). Handles HF parquet audio structs and file paths."""
    if isinstance(value, dict):
        if value.get("bytes"):
            wav, sr = sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=False)
            return (wav.mean(axis=1) if wav.ndim > 1 else wav), sr
        value = value.get("path")
    for base in (table_dir, src, ""):
        p = os.path.join(base, value) if base else value
        if os.path.isfile(p):
            wav, sr = librosa.load(p, sr=None, mono=True)
            return wav, sr
    hits = glob.glob(os.path.join(src, "**", os.path.basename(value)), recursive=True)
    if hits:
        return librosa.load(hits[0], sr=None, mono=True)
    raise FileNotFoundError(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder the dataset was downloaded to")
    ap.add_argument("--out", required=True, help="output folder for 24 kHz wavs + jsonl")
    ap.add_argument("--inspect", action="store_true", help="print columns / sample rows and exit")
    ap.add_argument("--audio_col"); ap.add_argument("--text_col"); ap.add_argument("--conf_col")
    ap.add_argument("--promo_col"); ap.add_argument("--dur_col")
    ap.add_argument("--min_conf", type=float, default=0.90)
    ap.add_argument("--min_dur", type=float, default=1.0)
    ap.add_argument("--max_dur", type=float, default=20.0)
    ap.add_argument("--language", default="arabic")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N kept rows (quick pilot)")
    ap.add_argument("--keep_latin", action="store_true", help="keep clips whose text has Latin letters (default: drop)")
    ap.add_argument("--keep_digits", action="store_true", help="keep clips whose text has digits (default: drop - digits are read inconsistently)")
    ap.add_argument("--trim_db", type=float, default=40.0, help="trim leading/trailing silence quieter than this (0 = off)")
    ap.add_argument("--val_ratio", type=float, default=0.02, help="fraction held out to val_raw.jsonl")
    ap.add_argument("--min_cps", type=float, default=6.0, help="drop clips with fewer chars/sec (likely missing words)")
    ap.add_argument("--max_cps", type=float, default=25.0, help="drop clips with more chars/sec (likely extra words)")
    args = ap.parse_args()
    src, out = os.path.expanduser(args.src), os.path.expanduser(args.out)

    df = load_table(find_table(src))
    cols = [c for c in df.columns if c != "__table_dir"]
    if args.inspect:
        print("rows:", len(df)); print("columns:", cols)
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(df[cols].head(5).astype(str).apply(lambda c: c.str[:60]))
        return

    audio_col = args.audio_col or pick(cols, "audio", "file", "path", "wav")
    text_col = args.text_col or pick(cols, "text", "transcription", "transcript", "sentence")
    conf_col = args.conf_col or pick(cols, "confidence", "conf", "score", required=False)
    promo_col = args.promo_col or pick(cols, "is_promo", "promo", "promotional", "sponsor", required=False)
    dur_col = args.dur_col or pick(cols, "duration", "dur", "length", required=False)
    print(f"columns -> audio={audio_col} text={text_col} conf={conf_col} promo={promo_col} dur={dur_col}")

    n0 = len(df)
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]
    df = df[df[text_col].astype(str).apply(lambda s: bool(ARABIC_RE.search(s)))]
    if conf_col:
        df = df[pd.to_numeric(df[conf_col], errors="coerce") >= args.min_conf]
    if promo_col:
        df = df[~df[promo_col].astype(str).str.lower().isin(["true", "1", "yes"])]
    if dur_col:
        d = pd.to_numeric(df[dur_col], errors="coerce")
        df = df[(d >= args.min_dur) & (d <= args.max_dur)]
    df[text_col] = df[text_col].astype(str).apply(normalize_text)
    if not args.keep_latin:
        df = df[~df[text_col].str.contains(LATIN_RE)]
    if not args.keep_digits:
        df = df[~df[text_col].str.contains(DIGIT_RE)]
    print(f"kept {len(df)}/{n0} rows after text/confidence/promo/duration/latin/digit filters")
    if args.limit:
        df = df.head(args.limit)

    os.makedirs(os.path.join(out, "wavs"), exist_ok=True)
    rows, src_srs, total_sec = [], {}, 0.0
    n_bad_cps = 0
    best_ref = None  # (score, path) - longest high-confidence clip between 5 and 12 s
    for n, (idx, r) in enumerate(df.iterrows()):
        try:
            wav, sr = load_audio(r[audio_col], r["__table_dir"], src)
        except Exception as e:  # noqa: BLE001
            print(f"skip row {idx}: {e}"); continue
        src_srs[sr] = src_srs.get(sr, 0) + 1
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        if args.trim_db > 0:
            _, (a, b) = librosa.effects.trim(wav, top_db=args.trim_db)
            pad = int(0.15 * TARGET_SR)  # keep ~150 ms of natural silence each side
            wav = wav[max(0, a - pad): min(len(wav), b + pad)]
        dur = len(wav) / TARGET_SR
        if not (args.min_dur <= dur <= args.max_dur):
            continue
        text = str(r[text_col])
        cps = len(text.replace(" ", "")) / dur
        if not (args.min_cps <= cps <= args.max_cps):
            n_bad_cps += 1
            continue
        peak = float(np.max(np.abs(wav))) or 1.0
        wav = (wav / peak * 0.95).astype(np.float32)  # peak-normalise
        path = os.path.abspath(os.path.join(out, "wavs", f"{n:06d}.wav"))
        sf.write(path, wav, TARGET_SR, subtype="PCM_16")
        total_sec += dur
        rows.append({"audio": path, "text": text, "language": args.language})
        conf = float(r[conf_col]) if conf_col else 1.0
        if 5.0 <= dur <= 12.0 and (best_ref is None or (conf, dur) > best_ref[0]):
            best_ref = ((conf, dur), path)
        if n % 500 == 0:
            print(f"  {n}/{len(df)} processed, {total_sec/3600:.1f} h so far")

    if not rows:
        sys.exit("No clips written - check the column mapping with --inspect")
    ref = best_ref[1] if best_ref else rows[0]["audio"]
    for row in rows:
        row["ref_audio"] = ref  # same reference for every sample (recommended by the Qwen README)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    n_val = int(len(rows) * args.val_ratio)
    val_rows = [rows[i] for i in idx[:n_val] if rows[i]["audio"] != ref]
    train_rows = [rows[i] for i in idx[n_val:]] + [rows[i] for i in idx[:n_val] if rows[i]["audio"] == ref]

    jsonl = os.path.join(out, "train_raw.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "val_raw.jsonl"), "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train {len(train_rows)} / val {len(val_rows)}")

    print(f"\nwrote {len(rows)} clips, {total_sec/3600:.1f} h -> {jsonl}")
    print(f"dropped {n_bad_cps} clips with implausible chars/sec (text/audio mismatch)")
    print(f"reference clip: {ref}")
    print(f"source sample rates: {src_srs}")
    if any(s < TARGET_SR for s in src_srs):
        print("WARNING: some source audio is below 24 kHz - upsampled clips will sound dull/band-limited.")


if __name__ == "__main__":
    main()