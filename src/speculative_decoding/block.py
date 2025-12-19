"""Exact draft/verify sampling over a target block-verification contract."""

from __future__ import annotations

import math
import random
from typing import Protocol, Sequence

from .sampling import (
    ProbabilityModel,
    SamplingResult,
    _distribution,
    _validate_request,
    sample_distribution,
    validate_distribution,
)


class BlockProbabilityModel(Protocol):
    vocab_size: int

    def block_distributions(self, prefix: Sequence[int], proposals: Sequence[int]) -> Sequence[Sequence[float]]:
        """Return k+1 rows: p(next | prefix + proposals[:i]), i=0..k."""
        ...

    def commit(self, prefix: Sequence[int]) -> None:
        """Discard speculative state after the actually committed prefix."""
        ...


class ReferenceBlockModel:
    """Serial probability-table adapter for contract tests, not a speedup backend."""

    def __init__(self, model: ProbabilityModel):
        self.model = model
        self.vocab_size = model.vocab_size

    def block_distributions(self, prefix: Sequence[int], proposals: Sequence[int]) -> list[tuple[float, ...]]:
        return [
            _distribution(self.model, list(prefix) + list(proposals[:index])) for index in range(len(proposals) + 1)
        ]

    def commit(self, prefix: Sequence[int]) -> None:
        """The probability-table adapter has no cache to mutate or roll back."""
        return None


def speculative_sample_block(
    target: BlockProbabilityModel,
    draft: ProbabilityModel,
    prefix: Sequence[int] = (),
    *,
    max_new_tokens: int = 16,
    draft_tokens: int = 4,
    seed: int = 0,
    max_context: int = 4096,
    stop_token: int | None = None,
) -> SamplingResult:
    """Exact acceptance/residual sampling with one target API call per block.

    CachedHFModel implements the API using one physical target forward. The
    table adapter implements it serially and makes no physical-forward claim.
    RNG draw order matches the serial sampler for identical probabilities.
    """
    _validate_request(target, prefix, max_new_tokens, max_context)
    _validate_request(draft, prefix, max_new_tokens, max_context)
    if target.vocab_size != draft.vocab_size:
        raise ValueError("target and draft vocabularies must match")
    if type(seed) is not int or type(draft_tokens) is not int or not 1 <= draft_tokens <= 64:
        raise ValueError("integer seed and draft_tokens in [1, 64] required")
    if stop_token is not None and (type(stop_token) is not int or not 0 <= stop_token < target.vocab_size):
        raise ValueError("stop_token outside vocabulary")
    rng = random.Random(seed)
    result = SamplingResult(tokens=[], events=[])
    initial = list(prefix)

    def commit() -> None:
        committed = initial + result.tokens
        target.commit(committed)
        draft_commit = getattr(draft, "commit", None)
        if draft_commit is not None:
            draft_commit(committed)

    def append(token: int, kind: str) -> bool:
        result.tokens.append(token)
        result.events.append({"kind": kind, "token": token, "position": len(result.tokens) - 1})
        if token == stop_token:
            result.stop_reason = "stop_token"
            return True
        return len(result.tokens) >= max_new_tokens

    while len(result.tokens) < max_new_tokens:
        base = initial + result.tokens
        proposals = []
        draft_rows = []
        for _ in range(min(draft_tokens, max_new_tokens - len(result.tokens))):
            q = _distribution(draft, base + proposals)
            proposals.append(sample_distribution(q, rng))
            draft_rows.append(q)
            result.draft_calls += 1
            result.proposed_tokens += 1
            if proposals[-1] == stop_token:
                break
        rows = target.block_distributions(base, proposals)
        if len(rows) != len(proposals) + 1:
            raise ValueError("target block must return one row per proposal plus the bonus row")
        target_rows = [validate_distribution(row) for row in rows]
        if any(len(row) != target.vocab_size for row in target_rows):
            raise ValueError("target block distribution vocabulary mismatch")
        result.target_calls += 1
        rejected = False
        for index, (token, q) in enumerate(zip(proposals, draft_rows)):
            p = target_rows[index]
            result.verified_tokens += 1
            if rng.random() < min(1.0, p[token] / q[token]):
                result.accepted_tokens += 1
                if append(token, "accepted_draft"):
                    commit()
                    return result
            else:
                result.rejected_tokens += 1
                residual = [max(pi - qi, 0.0) for pi, qi in zip(p, q)]
                mass = math.fsum(residual)
                if mass <= 0:
                    raise ArithmeticError("rejection without positive residual mass")
                done = append(sample_distribution([value / mass for value in residual], rng), "target_residual")
                rejected = True
                if done:
                    commit()
                    return result
                break
        if not rejected and len(result.tokens) < max_new_tokens:
            if append(sample_distribution(target_rows[-1], rng), "target_bonus"):
                commit()
                return result
        commit()
    return result
