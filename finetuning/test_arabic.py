# Listen-test a checkpoint on fixed Egyptian sentences (NOT from the training set).
# IMPORTANT: always pass language="Arabic". Without it the model gets the "Auto" prefix
# (no language token) - a different input than it was trained on.
#   python test_arabic.py --ckpt output/checkpoint-epoch-2 --speaker kazyon_ar --out samples_ep2
import argparse, os
import soundfile as sf, torch
from qwen_tts import Qwen3TTSModel

SENTENCES = [
    "أهلاً بيك في كازيون، إزيك النهارده؟",
    "لو سمحت استنى معايا دقيقة واحدة بس، هشوفلك الطلب بتاعك.",
    "العرض ده ساري لحد آخر الأسبوع، فمتفوتوش.",
    "أنا مش فاهم إنت عايز إيه بالظبط، ممكن تقولها تاني؟",
    "إحنا فاتحين كل يوم من الساعة تسعة الصبح لحد الساعة اتناشر بالليل.",
    "والله العظيم الجو النهارده حر قوي، خلي بالك من نفسك.",
    "قعدنا على القهوة وشربنا شاي بالنعناع واتكلمنا كتير.",
    "معلش، الحاجة دي خلصت، بس هتنزل تاني بكرة إن شاء الله.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--speaker", default="kazyon_ar")
ap.add_argument("--language", default="Arabic"); ap.add_argument("--out", default="samples")
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--attn", default="sdpa")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tts = Qwen3TTSModel.from_pretrained(a.ckpt, device_map=a.device, dtype=torch.bfloat16, attn_implementation=a.attn)
wavs, sr = tts.generate_custom_voice(text=SENTENCES, speaker=a.speaker, language=[a.language] * len(SENTENCES))
for i, (t, w) in enumerate(zip(SENTENCES, wavs)):
    sf.write(os.path.join(a.out, f"{i:02d}.wav"), w, sr)
    print(f"{i:02d}.wav  {len(w)/sr:5.1f}s  {t}")
