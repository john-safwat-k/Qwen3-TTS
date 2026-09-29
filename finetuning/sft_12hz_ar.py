# coding=utf-8
# Arabic-language fine-tuning for Qwen3-TTS-12Hz-{1.7B,0.6B}-Base.
# Based on QwenLM/Qwen3-TTS finetuning/sft_12hz.py (Apache-2.0).
#
# Changes vs. the official script:
#   1. Registers a new language ("arabic" by default) in talker_config.codec_language_id,
#      using an unused codec-embedding row (2072 by default, same choice as the public KSA fine-tune).
#   2. Warm-starts that row as the mean of the 10 existing language embeddings.
#   3. Uses TTSDatasetWithLanguage, which puts the language token in the codec prefix,
#      and injects the speaker embedding at the per-sample slot the dataset reports.
#   4. Saves the language into every checkpoint's config.json, so inference works with
#      generate_custom_voice(text=..., speaker=<name>, language="Arabic").
#   5. --attn_implementation flag (default sdpa; pass flash_attention_2 if flash-attn works on your GPU).
#   6. Configurable gradient accumulation (--grad_accum; official script hardcodes 4).
#   7. Accepts a hub name for --init_model_path (resolved to the local HF cache).
#
# v2 fixes (train/inference alignment - same as upstream PR QwenLM/Qwen3-TTS#278, issue #371):
#   8. Text embeddings now go through talker.text_projection, exactly like inference does.
#      (v1 fed raw text_embedding -> the model was trained on a different text input than it sees at inference.)
#   9. Talker loss computed here with ONE shift. v1 sliced inputs[:, :-1] / labels[:, 1:] AND let
#      ForCausalLMLoss shift again -> the talker learned to predict frame j+2 instead of j+1.
#  10. Sub-talker gets the hidden state that PREDICTED the frame (codec_mask[:, 1:]), matching
#      generation's past_hidden. v1 used codec_mask[:, :-1] = a hidden that had already seen the frame (leak).
#  11. Sub-talker loss computed here without the extra shift (v1 was shifted twice as well).
#  13. fp32 master weights (bf16 autocast). With bf16 weights, lr*grad updates smaller than
#      ~0.4% of a weight round to zero - at lr 2e-6 most of v1's updates were silently lost.
#  12. Warmup + cosine LR schedule, separate logging of talker / sub-talker loss, held-out eval loss.
import argparse
import json
import os
import shutil

import math

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dataset_ar import TTSDatasetWithLanguage
from huggingface_hub import snapshot_download
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup

SPEAKER_TOKEN_ID = 3000  # same slot the official script uses for the fine-tuned speaker
target_speaker_embedding = None


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--eval_jsonl", type=str, default=None, help="held-out set; eval loss printed each epoch")
    parser.add_argument("--sub_loss_weight", type=float, default=0.3)
    parser.add_argument("--bf16_weights", action="store_true", help="old behaviour: keep weights in bf16 (less memory, lossy updates)")
    parser.add_argument("--speaker_name", type=str, default="kazyon_ar")
    parser.add_argument("--language_name", type=str, default="arabic")
    parser.add_argument("--language_token_id", type=int, default=2072)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16", log_with="tensorboard"
    )

    # Resolve a hub name (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base) to its local cache folder,
    # because checkpoints are built by copying this folder.
    MODEL_PATH = args.init_model_path
    if not os.path.isdir(MODEL_PATH):
        MODEL_PATH = snapshot_download(MODEL_PATH)

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if args.bf16_weights else torch.float32,
        attn_implementation=args.attn_implementation,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # ---- 1) register the new language ----
    lang = args.language_name.lower()
    lang_map = dict(config.talker_config.codec_language_id)
    existing_ids = sorted(set(lang_map.values()))
    tc = config.talker_config
    reserved = {
        tc.codec_pad_id, tc.codec_bos_id, tc.codec_eos_token_id,
        tc.codec_think_id, tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id,
        SPEAKER_TOKEN_ID,
    }
    if lang in lang_map:
        accelerator.print(f"[lang] '{lang}' already registered as {lang_map[lang]} - reusing it")
        lang_id = lang_map[lang]
    else:
        lang_id = args.language_token_id
        if lang_id in existing_ids or lang_id in reserved or lang_id < 2048:
            raise ValueError(f"language_token_id {lang_id} collides with an existing/reserved codec id")
        lang_map[lang] = lang_id
        config.talker_config.codec_language_id = lang_map
        # also on the loaded model's config, so its inference helpers accept the new language
        qwen3tts.model.config.talker_config.codec_language_id = lang_map

        # ---- 2) warm-start: mean of the existing language embeddings ----
        emb = qwen3tts.model.talker.model.codec_embedding.weight
        with torch.no_grad():
            emb[lang_id] = emb[existing_ids].float().mean(dim=0).to(emb.dtype)
        accelerator.print(
            f"[lang] registered '{lang}' -> codec id {lang_id}, initialised from mean of {len(existing_ids)} languages"
        )

    train_data = [json.loads(line) for line in open(args.train_jsonl, encoding="utf-8")]
    n_lang = sum(1 for d in train_data if str(d.get("language", "auto")).lower() == lang)
    accelerator.print(f"[data] {len(train_data)} samples, {n_lang} tagged language='{lang}'")
    if n_lang == 0:
        accelerator.print(f"[data] WARNING: no sample has \"language\": \"{lang}\" - the new token will not be trained")

    dataset = TTSDatasetWithLanguage(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4
    )

    eval_dataloader = None
    if args.eval_jsonl:
        eval_data = [json.loads(line) for line in open(args.eval_jsonl, encoding="utf-8")]
        eval_ds = TTSDatasetWithLanguage(eval_data, qwen3tts.processor, config)
        eval_dataloader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=eval_ds.collate_fn, num_workers=2)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(qwen3tts.model, optimizer, train_dataloader)
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    # scheduler is stepped manually, once per real optimizer update
    updates_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum)
    total_updates = updates_per_epoch * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * total_updates), num_training_steps=total_updates
    )
    accelerator.print(f"[train] {total_updates} optimizer updates, lr={args.lr}, warmup={int(args.warmup_ratio * total_updates)}")

    def compute_loss(batch):
        global target_speaker_embedding
        input_ids = batch["input_ids"]
        codec_ids = batch["codec_ids"]
        ref_mels = batch["ref_mels"]
        text_embedding_mask = batch["text_embedding_mask"]
        codec_embedding_mask = batch["codec_embedding_mask"]
        attention_mask = batch["attention_mask"]
        codec_0_labels = batch["codec_0_labels"]
        codec_mask = batch["codec_mask"]
        speaker_pos = batch["speaker_pos"]

        speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
        if target_speaker_embedding is None:
            target_speaker_embedding = speaker_embedding

        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]

        # FIX 8: same text path as inference (text_embedding -> text_projection)
        input_text_embedding = model.talker.text_projection(
            model.talker.model.text_embedding(input_text_ids)
        ) * text_embedding_mask
        input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        rows = torch.arange(input_codec_embedding.shape[0], device=input_codec_embedding.device)
        input_codec_embedding[rows, speaker_pos.to(rows.device), :] = speaker_embedding.to(input_codec_embedding.dtype)

        input_embeddings = input_text_embedding + input_codec_embedding
        for i in range(1, 16):
            codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            input_embeddings = input_embeddings + codec_i_embedding * codec_mask.unsqueeze(-1)

        # no labels -> we compute the loss ourselves with exactly one shift
        outputs = model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            output_hidden_states=True,
        )
        # FIX 9: logits[t] (having seen positions <= t) predicts codec_0 at t+1
        logits = outputs.logits
        targets = codec_0_labels[:, 1:].to(logits.device)
        talker_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
        )

        # FIX 10: hidden at position p-1 (the one that predicted frame p) -> same as generation's past_hidden
        hidden_states = outputs.hidden_states[0][-1]  # [B, T-1, H]
        talker_hidden_states = hidden_states[codec_mask[:, 1:]]
        talker_codec_ids = codec_ids[codec_mask]
        sub_logits, _ = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
        # FIX 11: sub_logits[:, k] already predicts codebook k+1 -> no extra shift
        sub_loss = F.cross_entropy(
            sub_logits.float().reshape(-1, sub_logits.shape[-1]), talker_codec_ids[:, 1:].reshape(-1)
        )
        return talker_loss, sub_loss

    @torch.no_grad()
    def evaluate():
        if eval_dataloader is None:
            return None
        model.eval()
        tl, sl, n = 0.0, 0.0, 0
        for b in eval_dataloader:
            with accelerator.autocast():
                t, s_ = compute_loss(b)
            tl += t.item(); sl += s_.item(); n += 1
        model.train()
        return tl / max(n, 1), sl / max(n, 1)

    model.train()
    update = 0

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with accelerator.autocast():  # submodule calls bypass prepare()'s autocast
                    talker_loss, sub_loss = compute_loss(batch)
                loss = talker_loss + args.sub_loss_weight * sub_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()
                    update += 1

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step}/{len(train_dataloader)} | update {update}/{total_updates} "
                    f"| lr {scheduler.get_last_lr()[0]:.2e} | talker {talker_loss.item():.4f} | sub {sub_loss.item():.4f}"
                )

        ev = evaluate()
        if ev is not None:
            accelerator.print(f"[eval] epoch {epoch} | talker {ev[0]:.4f} | sub {ev[1]:.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            with open(os.path.join(MODEL_PATH, "config.json"), "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: SPEAKER_TOKEN_ID}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            talker_config["codec_language_id"] = lang_map  # includes the new language
            config_dict["talker_config"] = talker_config

            with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {
                k: (v.detach().to("cpu").to(torch.bfloat16) if v.is_floating_point() else v.detach().to("cpu"))
                for k, v in unwrapped_model.state_dict().items()
            }

            for k in [k for k in state_dict if k.startswith("speaker_encoder")]:
                del state_dict[k]

            weight = state_dict["talker.model.codec_embedding.weight"]
            weight[SPEAKER_TOKEN_ID] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            accelerator.print(f"[save] {output_dir}")


if __name__ == "__main__":
    train()