# coding=utf-8
# Arabic-language extension of the official Qwen3-TTS fine-tuning dataset.
# Based on QwenLM/Qwen3-TTS finetuning/dataset.py (Apache-2.0).
#
# What changes vs. the official dataset:
#   The official collate_fn always builds the "no language" codec prefix:
#       [nothink, think_bos, think_eos, <spk>, pad]                 (speaker slot at index 6)
#   When a sample has a language that exists in talker_config.codec_language_id,
#   this version builds the same prefix the model's own inference code builds:
#       [think, think_bos, <lang_id>, think_eos, <spk>, pad]        (speaker slot at index 7)
#   and shifts the rest of the sequence right by one position. That keeps training
#   and inference (Qwen3TTSModel.generate_custom_voice(..., language="Arabic")) aligned.
import torch

from dataset import TTSDataset


class TTSDatasetWithLanguage(TTSDataset):
    def _language_id(self, language):
        if language is None or str(language).lower() == "auto":
            return None
        lang_map = self.config.talker_config.codec_language_id
        key = str(language).lower()
        if key not in lang_map:
            raise ValueError(f"Language '{language}' is not in codec_language_id: {sorted(lang_map)}")
        return lang_map[key]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["language_id"] = self._language_id(self.data_list[idx].get("language", "Auto"))
        return sample

    def collate_fn(self, batch):
        assert self.lag_num == -1
        tc = self.config.talker_config

        # +1 position for every sample that carries a language token
        offsets = [0 if b["language_id"] is None else 1 for b in batch]
        item_length = [
            b["text_ids"].shape[1] + b["audio_codes"].shape[0] + p for b, p in zip(batch, offsets)
        ]
        max_length = max(item_length) + 8
        bsz, t = len(batch), max_length

        input_ids = torch.zeros((bsz, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((bsz, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((bsz, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((bsz, t), dtype=torch.bool)
        codec_mask = torch.zeros((bsz, t), dtype=torch.bool)
        attention_mask = torch.zeros((bsz, t), dtype=torch.long)
        codec_0_labels = torch.full((bsz, t), -100, dtype=torch.long)
        speaker_pos = torch.zeros((bsz,), dtype=torch.long)

        for i, (data, p) in enumerate(zip(batch, offsets)):
            text_ids = data["text_ids"]
            audio_codecs = data["audio_codes"]
            audio_codec_0 = audio_codecs[:, 0]

            L = text_ids.shape[1]
            C = audio_codec_0.shape[0]
            s = 8 + p  # first position after the prefix (8 in the official layout)

            # ---- text channel ----
            input_ids[i, :3, 0] = text_ids[0, :3]                      # <|im_start|>assistant\n
            input_ids[i, 3 : s - 1, 0] = self.config.tts_pad_token_id  # pads under the codec prefix
            input_ids[i, s - 1, 0] = self.config.tts_bos_token_id
            input_ids[i, s : s + L - 3, 0] = text_ids[0, 3:]
            input_ids[i, s + L - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, s + L - 2 : s + L + C, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, : s + L + C] = True

            # ---- codec channel: prefix ----
            if data["language_id"] is None:
                prefix = [tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]
            else:
                prefix = [tc.codec_think_id, tc.codec_think_bos_id, data["language_id"], tc.codec_think_eos_id]
            prefix += [0, tc.codec_pad_id]  # 0 = placeholder, replaced by the speaker embedding
            input_ids[i, 3:s, 1] = torch.tensor(prefix)
            spk = 3 + len(prefix) - 2  # 6 without language, 7 with
            speaker_pos[i] = spk

            # ---- codec channel: body ----
            input_ids[i, s : s + L - 2, 1] = tc.codec_pad_id
            input_ids[i, s + L - 2, 1] = tc.codec_bos_id
            input_ids[i, s + L - 1 : s + L - 1 + C, 1] = audio_codec_0
            input_ids[i, s + L - 1 + C, 1] = tc.codec_eos_token_id

            codec_0_labels[i, s + L - 1 : s + L - 1 + C] = audio_codec_0
            codec_0_labels[i, s + L - 1 + C] = tc.codec_eos_token_id

            codec_ids[i, s + L - 1 : s + L - 1 + C, :] = audio_codecs

            codec_embedding_mask[i, 3 : s + L + C] = True
            codec_embedding_mask[i, spk] = False

            codec_mask[i, s + L - 1 : s + L - 1 + C] = True
            attention_mask[i, : s + L + C] = True

        ref_mels = torch.cat([data["ref_mel"] for data in batch], dim=0)

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "speaker_pos": speaker_pos,
        }