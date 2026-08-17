"""Contextual embedding extraction with explicit target-token alignment."""
from __future__ import annotations

import numpy as np
import torch


class EmbeddingExtractor:
    """Extract one target concept from every hidden layer of a Hugging Face model."""

    def _get_char_span_from_index(self, text: str, target_idx: int):
        parts = text.split()
        if target_idx < 0:
            target_idx = len(parts) + target_idx
        if target_idx < 0 or target_idx >= len(parts):
            return None
        search_pos = 0
        for i, part in enumerate(parts):
            start = text.find(part, search_pos)
            if start == -1:
                raise ValueError(f"Could not locate token {part!r} in {text!r}")
            end = start + len(part)
            if i == target_idx:
                return start, end
            search_pos = end
        return None

    def _refine_token_indices_by_concept(self, concept, tokenizer, input_ids, anchor_indices, search_radius=3):
        if not anchor_indices:
            return anchor_indices
        concept_lower = concept.lower()
        lo = max(0, min(anchor_indices) - search_radius)
        hi = min(len(input_ids), max(anchor_indices) + search_radius + 1)
        window_ids = input_ids[lo:hi]
        for span_len in range(1, 5):
            for start in range(len(window_ids) - span_len + 1):
                span_text = tokenizer.decode(window_ids[start : start + span_len]).strip().lower()
                if concept_lower in span_text:
                    abs_start = lo + start
                    return list(range(abs_start, abs_start + span_len))
        return anchor_indices

    def target_token_indices(self, text, tokenizer, target_word_index, concept=None):
        """Map a whitespace word index to tokenizer indices using character offsets."""
        char_span = self._get_char_span_from_index(text, target_word_index)
        if char_span is None:
            return [], None
        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded["offset_mapping"][0]
        indices = []
        start_char, end_char = char_span
        for i, (token_start, token_end) in enumerate(offsets.tolist()):
            if token_start == token_end:
                continue
            if max(start_char, token_start) < min(end_char, token_end):
                indices.append(i)
        if concept is not None and indices:
            indices = self._refine_token_indices_by_concept(
                concept, tokenizer, encoded["input_ids"][0].tolist(), indices
            )
        return indices, char_span

    def extract_contextual_single(
        self,
        text: str,
        model,
        tokenizer,
        target_word_index: list[int] | None = None,
        concept: str | None = None,
        pooling: str = "mean",
    ) -> list[np.ndarray]:
        """Return ``[hidden_layers, hidden_dim]`` for one target word.

        The input embedding layer is intentionally skipped, matching the original
        activation files used in the paper.
        """
        word_position = target_word_index[0] if target_word_index else None
        inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        if word_position is None:
            target_indices = [i for i, (s, e) in enumerate(offsets.tolist()) if s != e]
        else:
            char_span = self._get_char_span_from_index(text, word_position)
            if char_span is None:
                return [np.zeros(model.config.hidden_size, dtype=np.float32) for _ in range(model.config.num_hidden_layers)]
            start_char, end_char = char_span
            target_indices = []
            for i, (token_start, token_end) in enumerate(offsets.tolist()):
                if token_start == token_end:
                    continue
                if max(start_char, token_start) < min(end_char, token_end):
                    target_indices.append(i)
            if concept is not None and target_indices:
                target_indices = self._refine_token_indices_by_concept(
                    concept, tokenizer, inputs["input_ids"][0].tolist(), target_indices
                )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:]

        if not target_indices:
            return [np.zeros(model.config.hidden_size, dtype=np.float32) for _ in hidden_states]

        out = []
        for layer in hidden_states:
            target = layer[0, target_indices]
            if pooling == "last":
                pooled = target[-1]
            elif pooling == "mean":
                pooled = target.mean(dim=0)
            else:
                raise ValueError("pooling must be 'mean' or 'last'")
            out.append(pooled.detach().cpu().numpy().astype(np.float32))
        return out
