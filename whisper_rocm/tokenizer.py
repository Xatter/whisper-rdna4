"""Tokenizer + suppress-token mask (DESIGN.md S3, S5).

Reuses openai-whisper's tiktoken-backed tokenizer module directly (the
"tiktoken (from openai-whisper package)" DESIGN.md asks for) rather than
re-deriving the BPE merge ranks and special-token table by hand -- that
table is exactly reproduced here only by importing it. No other
openai-whisper runtime code (model, decoding loop) is used.

Special tokens (verified against the large-v3-turbo checkpoint, n_vocab
51866, num_languages=100): sot=50258, en=50259, transcribe=50360,
notimestamps=50364, eot=50257.
"""

from __future__ import annotations

import torch
from whisper.tokenizer import get_tokenizer as _oa_get_tokenizer

SOT = 50258
EN = 50259
TRANSCRIBE = 50360
NO_TIMESTAMPS = 50364
EOT = 50257
TIMESTAMP_BEGIN = 50365  # verified against tokenizer.timestamp_begin for this checkpoint

PROMPT = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
# Timestamps mode (M5, orchestrator directive 2026-08-09): no notimestamps
# token in the prompt, so the model is free (in fact forced, by
# ops.apply_timestamp_rules on the torch path / K5's SS3c extension on the
# kernel path) to open every segment with a timestamp token.
PROMPT_TIMESTAMPS = [SOT, EN, TRANSCRIBE]


class Tokenizer:
    def __init__(self, n_vocab: int = 51866, num_languages: int = 100, language: str = "en"):
        self._oa = _oa_get_tokenizer(
            multilingual=True, num_languages=num_languages, language=language, task="transcribe"
        )
        assert self._oa.sot == SOT
        assert self._oa.eot == EOT
        assert self._oa.transcribe == TRANSCRIBE
        assert self._oa.no_timestamps == NO_TIMESTAMPS
        assert self._oa.timestamp_begin == TIMESTAMP_BEGIN
        self.n_vocab = n_vocab

    def decode(self, token_ids: list[int]) -> str:
        # Drop special tokens (>= eot's block start) before detokenizing text.
        text_ids = [t for t in token_ids if t < EOT]
        return self._oa.decode(text_ids)

    def suppress_token_ids(self) -> list[int]:
        """Mirror openai-whisper's DecodingTask._get_suppress_tokens with
        suppress_tokens="-1" (the default): non_speech_tokens plus the
        control tokens transcribe/translate/sot/sot_prev/sot_lm/no_speech.
        """
        ids = set(self._oa.non_speech_tokens)
        ids.add(self._oa.transcribe)
        ids.add(self._oa.translate)
        ids.add(self._oa.sot)
        ids.add(self._oa.sot_prev)
        ids.add(self._oa.sot_lm)
        if self._oa.no_speech is not None:
            ids.add(self._oa.no_speech)
        return sorted(ids)

    def suppress_mask(self, device: torch.device, timestamps: bool = False) -> torch.Tensor:
        """Additive (-inf at suppressed ids, 0 elsewhere) mask, shape (n_vocab,).
        Consumed by ops.logits_argmax (K5 in the kernel path).

        Two modes (M5, orchestrator directive 2026-08-09):
        - timestamps=False (default): also permanently suppresses the whole
          timestamp-token range (>= TIMESTAMP_BEGIN, review finding F10,
          belt-and-suspenders) -- the notimestamps prompt token already
          keeps the model from emitting these in practice, but the model
          was never given an architectural guarantee against it, so the
          mask closes the gap explicitly rather than relying on
          prompt-following behavior alone.
        - timestamps=True: the timestamp range is deliberately left open
          (that's the whole point of this mode); instead NO_TIMESTAMPS
          (50364) is explicitly suppressed here. ops.apply_timestamp_rules
          / K5's SS3c extension also suppress it unconditionally every
          step, so this is redundant defense-in-depth, not load-bearing --
          included because it was asked for explicitly and costs nothing.
        """
        mask = torch.zeros(self.n_vocab, dtype=torch.float32, device=device)
        mask[self.suppress_token_ids()] = float("-inf")
        if timestamps:
            mask[NO_TIMESTAMPS] = float("-inf")
        else:
            mask[TIMESTAMP_BEGIN:] = float("-inf")
        return mask

    def blank_suppress_ids(self) -> list[int]:
        """SuppressBlank: at the first sampled position only, suppress the
        space token and eot (openai-whisper decoding.py:SuppressBlank)."""
        return list(self._oa.encode(" ")) + [EOT]
