"""Custom logits processor using KenLM to bias scores."""

import math
from functools import lru_cache
from typing import Optional

import kenlm
import torch
from torch.nn.functional import log_softmax, softmax
from transformers import LogitsProcessor, WhisperProcessor

from .ngram import UNK

LOG10_TO_E = math.log(10)


class KenLMLogitsProcessor(LogitsProcessor):
    """Logits processor that biases scores towards a KenLM language model."""

    def __init__(
        self,
        lm: kenlm.Model,
        token_to_id: dict[str, int],
        processor: WhisperProcessor,
        bias_towards_lm: float = 0.5,
    ) -> None:
        """Initialize logits processor."""
        super().__init__()

        self.lm = lm
        self.bias = bias_towards_lm
        self.processor = processor

        # token string -> token id
        self.vocab_to_id = dict(token_to_id)
        self.vocab_to_id["</s>"] = processor.tokenizer.eos_token_id  # <|endoftext|>

        self.vocab_ids: set[int] = set(self.vocab_to_id.values())
        self.vocab_to_id_sorted = sorted(self.vocab_to_id.items(), key=lambda kv: kv[1])

        self.vocab_mask: Optional[torch.Tensor] = None
        self.special_token_ids = set(processor.tokenizer.all_special_ids)
        self.unknown_token_mask: Optional[torch.Tensor] = None
        self.id_to_token: dict[int, str] = {}

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        """Bias scores towards language model."""
        # Convert to probabilities
        scores = softmax(scores, dim=-1)

        if self.vocab_mask is None:
            # Initialize mask with only vocabulary token ids
            self.vocab_mask = torch.tensor(
                [(t_id in self.vocab_ids) for t_id in range(scores.shape[-1])],
                device=scores.device,
            )

        if self.unknown_token_mask is None:
            # Initialize mask with only unknown tokens (not vocab, not special)
            self.unknown_token_mask = torch.tensor(
                [
                    (t_id not in self.special_token_ids)
                    and (t_id not in self.vocab_ids)
                    for t_id in range(scores.shape[-1])
                ],
                device=scores.device,
            )

        for batch_idx in range(len(input_ids)):
            current_token_ids = tuple(
                t_id
                for t_id in input_ids[batch_idx].tolist()
                if t_id not in self.special_token_ids
            )

            state, prob = self._get_state_and_prob(current_token_ids)

            # Compute log probabilities for tokens in the vocabulary
            vocab_log_probs: list[float] = []
            for token_str, _token_id in self.vocab_to_id_sorted:
                next_state = kenlm.State()
                next_prob = prob + (
                    self.lm.BaseScore(state, token_str, next_state) * LOG10_TO_E
                )
                vocab_log_probs.append(next_prob)

            biased_vocab_probs = (
                torch.exp(
                    log_softmax(
                        torch.tensor(vocab_log_probs, device=scores.device), dim=-1
                    )
                )
                * self.bias
            )

            scores[batch_idx, self.vocab_mask] += biased_vocab_probs

        return scores

    @lru_cache
    def _get_state_and_prob(
        self, token_ids: tuple[int, ...]
    ) -> tuple[kenlm.State, float]:
        """Get state/probability for token ids (LRU cache)."""
        if not token_ids:
            state = kenlm.State()
            prob = 0.0
            self.lm.BeginSentenceWrite(state)
            return (state, prob)

        state, prob = self._get_state_and_prob(token_ids[:-1])

        for token_id in token_ids:
            token_str = self.id_to_token.get(token_id)
            if token_str is None:
                token_str = f"w_{token_id}"
                if token_str in self.vocab_to_id:
                    self.id_to_token[token_id] = token_str
                else:
                    token_str = UNK
                    self.id_to_token[token_id] = UNK

            next_state = kenlm.State()
            prob += self.lm.BaseScore(state, token_str, next_state) * LOG10_TO_E
            state = next_state

        return state, prob
