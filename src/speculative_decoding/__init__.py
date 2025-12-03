"""Reference speculative sampling with explicit, inspectable probability kernels."""

from .sampling import MarkovModel, SamplingResult, greedy, sample_distribution, speculative_sample

__all__ = ["MarkovModel", "SamplingResult", "greedy", "sample_distribution", "speculative_sample"]
