"""Correctness-first draft/verify decoding; no throughput claims are made here.

The probability backend is deliberately separated from the decoding algorithm.
This module retains the serial reference path. The block contract and cached HF
implementation live in block.py and hf.py respectively.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Protocol, Sequence


class ProbabilityModel(Protocol):
    vocab_size: int

    def distribution(self, prefix: Sequence[int]) -> Sequence[float]: ...


def validate_distribution(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("distribution must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("probabilities must be real numbers, not booleans or strings")
    probabilities = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("probabilities must be finite and nonnegative")
    try:
        total = math.fsum(probabilities)
    except OverflowError as error:
        raise ValueError("probability sum overflowed") from error
    if not math.isclose(total, 1.0, abs_tol=1e-6, rel_tol=1e-6):
        raise ValueError("probabilities must sum to one")
    return tuple(value / total for value in probabilities)


def sample_distribution(values: Sequence[float], rng: random.Random) -> int:
    probabilities = validate_distribution(values)
    draw = rng.random()
    cumulative = 0.0
    last_positive = 0
    for token, probability in enumerate(probabilities):
        if probability > 0:
            last_positive = token
        cumulative += probability
        if draw < cumulative:
            return token
    return last_positive  # floating-point accumulation only


@dataclass(frozen=True)
class MarkovModel:
    """Synthetic finite-state next-token model with a known exact distribution."""

    initial: tuple[float, ...]
    transitions: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial", validate_distribution(self.initial))
        if self.transitions is not None:
            if len(self.transitions) != self.vocab_size:
                raise ValueError("one transition row is required per token")
            for row in self.transitions:
                if len(row) != self.vocab_size:
                    raise ValueError("transition vocabulary mismatch")
                validate_distribution(row)
            object.__setattr__(self, "transitions", tuple(validate_distribution(row) for row in self.transitions))

    @property
    def vocab_size(self) -> int:
        return len(self.initial)

    def distribution(self, prefix: Sequence[int]) -> Sequence[float]:
        if any(type(token) is not int or not 0 <= token < self.vocab_size for token in prefix):
            raise ValueError("prefix token out of vocabulary")
        if prefix and self.transitions is not None:
            return self.transitions[prefix[-1]]
        return self.initial


@dataclass
class SamplingResult:
    tokens: list[int]
    events: list[dict]
    draft_calls: int = 0
    target_calls: int = 0
    proposed_tokens: int = 0
    verified_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    stop_reason: str = "max_new_tokens"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["accepted_per_proposed"] = self.accepted_tokens / self.proposed_tokens if self.proposed_tokens else None
        result["accepted_per_verified"] = self.accepted_tokens / self.verified_tokens if self.verified_tokens else None
        return result


def _distribution(model: ProbabilityModel, prefix: Sequence[int]) -> tuple[float, ...]:
    probabilities = validate_distribution(model.distribution(prefix))
    if len(probabilities) != model.vocab_size:
        raise ValueError("backend returned a different vocabulary size")
    return probabilities


def _validate_request(target: ProbabilityModel, prefix: Sequence[int], max_new_tokens: int, max_context: int) -> None:
    if type(target.vocab_size) is not int or not 1 <= target.vocab_size <= 1_000_000:
        raise ValueError("backend vocabulary must be an integer in [1, 1000000]")
    if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer in [0, 4096]")
    if type(max_context) is not int or not 1 <= max_context <= 131072:
        raise ValueError("max_context must be in [1, 131072]")
    if len(prefix) + max_new_tokens > max_context:
        raise ValueError("request exceeds the explicit context budget")
    if any(type(token) is not int or not 0 <= token < target.vocab_size for token in prefix):
        raise ValueError("prefix token out of vocabulary")


def speculative_sample(
    target: ProbabilityModel,
    draft: ProbabilityModel,
    prefix: Sequence[int] = (),
    *,
    max_new_tokens: int = 16,
    draft_tokens: int = 4,
    seed: int = 0,
    max_context: int = 4096,
    stop_token: int | None = None,
) -> SamplingResult:
    """Sample exactly from target p using draft q and residual max(p-q, 0).

    Correctness requires the same token vocabulary for p and q. Backends return
    their complete normalized distribution: truncation/top-k must be applied by
    the backend consistently if desired. Trace records contain token IDs only.
    """
    _validate_request(target, prefix, max_new_tokens, max_context)
    _validate_request(draft, prefix, max_new_tokens, max_context)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if target.vocab_size != draft.vocab_size:
        raise ValueError("target and draft must share a token vocabulary")
    if type(draft_tokens) is not int or not 1 <= draft_tokens <= 64:
        raise ValueError("draft_tokens must be in [1, 64]")
    if stop_token is not None and (type(stop_token) is not int or not 0 <= stop_token < target.vocab_size):
        raise ValueError("stop_token is outside vocabulary")
    rng = random.Random(seed)
    result = SamplingResult(tokens=[], events=[])
    initial = list(prefix)

    def append(token: int, kind: str) -> bool:
        result.tokens.append(token)
        result.events.append({"kind": kind, "token": token, "position": len(result.tokens) - 1})
        if token == stop_token:
            result.stop_reason = "stop_token"
            return True
        return len(result.tokens) >= max_new_tokens

    while len(result.tokens) < max_new_tokens:
        base = initial + result.tokens
        proposals: list[int] = []
        proposal_probabilities: list[tuple[float, ...]] = []
        count = min(draft_tokens, max_new_tokens - len(result.tokens))
        for _ in range(count):
            q = _distribution(draft, base + proposals)
            proposal_probabilities.append(q)
            proposals.append(sample_distribution(q, rng))
            result.draft_calls += 1
            result.proposed_tokens += 1
            if proposals[-1] == stop_token:
                break

        rejected = False
        for token, q in zip(proposals, proposal_probabilities):
            p = _distribution(target, initial + result.tokens)
            result.target_calls += 1
            result.verified_tokens += 1
            # A token sampled from q necessarily has q[token] > 0.
            acceptance = min(1.0, p[token] / q[token])
            if rng.random() < acceptance:
                result.accepted_tokens += 1
                if append(token, "accepted_draft"):
                    return result
            else:
                result.rejected_tokens += 1
                residual = [max(pi - qi, 0.0) for pi, qi in zip(p, q)]
                total = math.fsum(residual)
                if total <= 0:
                    raise ArithmeticError("rejection without positive residual mass")
                replacement = sample_distribution([value / total for value in residual], rng)
                if append(replacement, "target_residual"):
                    return result
                rejected = True
                break
        if not rejected and len(result.tokens) < max_new_tokens:
            p = _distribution(target, initial + result.tokens)
            result.target_calls += 1
            if append(sample_distribution(p, rng), "target_bonus"):
                return result
    return result


def greedy(
    model: ProbabilityModel, prefix: Sequence[int] = (), *, max_new_tokens: int = 16, max_context: int = 4096
) -> list[int]:
    """Separate deterministic baseline; not stochastic speculative sampling."""
    _validate_request(model, prefix, max_new_tokens, max_context)
    output: list[int] = []
    for _ in range(max_new_tokens):
        p = _distribution(model, list(prefix) + output)
        output.append(max(range(len(p)), key=lambda token: p[token]))
    return output


def exact_joint_distribution(
    model: ProbabilityModel,
    prefix: Sequence[int] = (),
    *,
    horizon: int = 3,
    stop_token: int | None = None,
    max_sequences: int = 4096,
) -> dict[tuple[int, ...], float]:
    """Enumerate the finite target law independently of speculative sampling.

    This oracle multiplies target conditionals along the full output tree. It
    contains no draft model, acceptance ratio, rejection correction, or RNG.
    Stop-token leaves have variable lengths and their probability is retained.
    Exact refers to complete enumeration; arithmetic is still float64.
    """
    _validate_request(model, prefix, horizon, 4096)
    if type(horizon) is not int or not 0 <= horizon <= 8:
        raise ValueError("exact enumeration horizon must be an integer in [0, 8]")
    if type(max_sequences) is not int or not 1 <= max_sequences <= 100_000:
        raise ValueError("exact enumeration leaf budget must be in [1, 100000]")
    if stop_token is not None and (type(stop_token) is not int or not 0 <= stop_token < model.vocab_size):
        raise ValueError("stop token outside vocabulary")
    frontier: list[tuple[tuple[int, ...], float]] = [((), 1.0)]
    leaves: dict[tuple[int, ...], float] = {}
    while frontier:
        output, probability = frontier.pop()
        if len(output) == horizon or (output and output[-1] == stop_token):
            leaves[output] = probability
            continue
        distribution = _distribution(model, list(prefix) + list(output))
        children = [(output + (token,), probability * value) for token, value in enumerate(distribution) if value > 0]
        if len(frontier) + len(leaves) + len(children) > max_sequences:
            raise ValueError("exact enumeration exceeds its leaf budget")
        frontier.extend(children)
    return dict(sorted(leaves.items()))
