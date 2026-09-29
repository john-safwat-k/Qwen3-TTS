# Step 0 sanity check: can the 12Hz codec itself reproduce native Egyptian speech?
# Encode -> decode N training clips and LISTEN to them next to the originals.
# If these already sound accented/robotic, the problem is the codec, not the LM - no fine-tune will fix it.
#   python check_codec_roundtrip.py --jsonl ~/data/eg100_24k/val_raw.jsonl --n 10 --out roundtrip
import argparse, json, os, shutil
import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", required=True); ap.add_argument("--n", type=int, default=10)
ap.add_argument("--out", default="roundtrip"); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--tokenizer", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rows = [json.loads(l) for l in open(a.jsonl, encoding="utf-8")][: a.n]
tok = Qwen3TTSTokenizer.from_pretrained(a.tokenizer, device_map=a.device)
enc = tok.encode([r["audio"] for r in rows])
wavs, sr = tok.decode(enc)
for i, (r, w) in enumerate(zip(rows, wavs)):
    shutil.copy(r["audio"], os.path.join(a.out, f"{i:02d}_orig.wav"))
    sf.write(os.path.join(a.out, f"{i:02d}_codec.wav"), w, sr)
    print(i, r["text"])
print(f"wrote {len(rows)} pairs to {a.out}/ - compare *_orig.wav vs *_codec.wav")
