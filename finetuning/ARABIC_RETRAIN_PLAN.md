# Egyptian Arabic retrain plan (v2)

## What went wrong in v1

The v1 scripts (`sft_12hz_ar.py`, `dataset_ar.py`) were built on the official `sft_12hz.py`, which has
three train/inference alignment bugs (upstream issue QwenLM/Qwen3-TTS#371, fix in PR #278, not merged):

| # | Bug | Effect |
|---|-----|--------|
| 1 | Text embeddings skip `talker.text_projection` in training; inference always applies it | Everything the model learned about Arabic text was learned on a different input than it gets at inference, so at inference it falls back to its pretrained (Chinese/English-heavy) behaviour |
| 2 | Talker loss shifted twice (`inputs[:, :-1]` + `labels[:, 1:]` + `ForCausalLMLoss` shifts again) | Talker trained to predict frame j+2 instead of j+1: it learns the wrong timing/phonetics |
| 3 | Sub-talker uses `hidden_states[codec_mask[:, :-1]]` (a hidden that already saw the frame) and its loss is shifted twice too | Codebooks 1-15 (the fine acoustic detail) are trained on a leaked input they never get at inference |

On top of that:

- **LR 2e-6, no warmup** (copied from the KSA fine-tune). That is very low for teaching a *new language*. The model barely moved away from the base model, and the new `arabic` token (id 2072, initialised as the mean of the 10 language embeddings) stayed close to "average language".
- **Inference must pass `language="Arabic"`.** The default is `Auto`, which builds a prefix *without* the language token, a different input from the one training used.
- **Data hygiene:** no silence trimming, digits and Latin words left in (read inconsistently), no check that the text length matches the audio length, no held-out set.

Together: the fine-tune taught the model very little that it could use at inference, so it spoke Arabic text with the base model's phonetics. That is the "Chinese speaker reading Arabic" sound.

## What changed in v2 (branch `arabic-finetune-v2`, uncommitted)

- `sft_12hz_ar.py`: fp32 master weights (bf16 weights silently dropped small updates), bugs 1-3 fixed (loss computed explicitly with a single shift), LR default 1e-5, warmup + cosine schedule, separate talker/sub-talker loss logs, `--eval_jsonl` held-out loss per epoch.
- `prepare_eg100.py`: silence trim, light text normalisation (no MSA conversion), drops clips with digits or Latin letters (flags to keep them), drops clips with an implausible chars-per-second rate (text/audio mismatch), writes `val_raw.jsonl` (2%).
- `check_codec_roundtrip.py`: encode/decode clips to check that the codec itself sounds native.
- `test_arabic.py`: fixed Egyptian test sentences, always with `language="Arabic"`.

## Plan

### Phase 0: sanity checks (≈1 h)
1. `python check_codec_roundtrip.py --jsonl <val_raw.jsonl> --n 10`. The `_codec.wav` files should sound like the originals. If they already sound accented, stop: the problem is the codec.
2. Listen to about 30 random clips while reading their transcripts. Check that the text matches the audio word for word, the speech is Egyptian (not MSA), and there is no music or second speaker.
3. Check the "source sample rates" line. If the audio is 16 kHz, expect a duller sound, but the accent is unaffected.

### Phase 1: rebuild the data
```bash
python prepare_eg100.py --src ~/data/eg100 --out ~/data/eg100_v2
python prepare_data.py --input_jsonl ~/data/eg100_v2/train_raw.jsonl --output_jsonl ~/data/eg100_v2/train.jsonl
python prepare_data.py --input_jsonl ~/data/eg100_v2/val_raw.jsonl   --output_jsonl ~/data/eg100_v2/val.jsonl
```
Aim for at least 50 h after filtering. If far less remains, loosen `--min_conf` or the cps limits.

### Phase 2: prove the fixed pipeline works (≈1 h GPU)
- **Overfit test:** take 50 clips (`head -50 train.jsonl`), train for 30 epochs at lr 2e-5, then synthesise those same sentences. They should come out close to the originals. If they don't, something is still misaligned; do not start the full run.
- Sub-talker loss must fall well below ~7.6 (ln 2048 = random guessing). If it sits above that, the alignment is still broken.

### Phase 3: full run
```bash
accelerate launch sft_12hz_ar.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl ~/data/eg100_v2/train.jsonl --eval_jsonl ~/data/eg100_v2/val.jsonl \
  --output_model_path output_v2 --batch_size 8 --grad_accum 4 \
  --lr 1e-5 --num_epochs 5 --speaker_name kazyon_ar
```
- After each epoch: `python test_arabic.py --ckpt output_v2/checkpoint-epoch-N --out samples_epN`.
- Choose the checkpoint by ear (native Egyptian listeners) plus eval loss. Stop when eval loss rises.
- If the accent is better but not native yet, run a second pass at lr 2e-5. If the output gets noisy or unstable, drop to 5e-6.

### Phase 4: only if it is still not native enough
- **More, and more varied, Egyptian data:** first fine-tune on multi-speaker Egyptian speech (language only), then fine-tune again on the single target voice. This is the approach that worked for Persian in upstream discussion #189.
- **Diacritisation or phonemes:** undiacritised Egyptian spelling is ambiguous. Feeding diacritised text or phonemes needs less data to learn the accent.
- Give the new language embedding row a higher LR than the rest of the model.

### How to judge "native"
- 3-5 native Egyptian listeners rate the `test_arabic.py` sentences from 1 to 5 for accent and naturalness, comparing v1 and v2 checkpoints blind.
- Optional: CER from an Egyptian-dialect ASR model on the generated audio (intelligibility).

## Note
The dataset license is CC-BY-NC-4.0: use it for the internal pilot only. A production voice for Kazyon needs licensed or recorded data.
