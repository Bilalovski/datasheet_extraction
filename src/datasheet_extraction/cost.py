"""Token accounting and cost estimation for the DeepSeek API.

Quality numbers without cost numbers don't decide anything: the interesting
question is never "is v4-pro better than v4-flash" (it should be) but "is it
better by enough to justify roughly 3x the input price on this task". So every
extraction carries its usage, and the ablation reports accuracy next to what the
accuracy cost.

DeepSeek bills prompt tokens at two different rates depending on whether they hit
its context cache, and reports the split per request — so the numbers here are
measured, not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

#: USD per million tokens, as published at https://api-docs.deepseek.com/quick_start/pricing
#: and read on 2026-07-15: (cache-hit input, cache-miss input, output).
#:
#: A cache hit costs ~2% of a miss, which is a far steeper discount than most
#: providers give — worth knowing before optimising anything else about cost.
PRICING: dict[str, tuple[float, float, float]] = {
    "deepseek-v4-flash": (0.0028, 0.14, 0.28),
    "deepseek-v4-pro": (0.003625, 0.435, 0.87),
}

#: Aliases DeepSeek retires on 2026-07-24. Mapped so an old id still prices
#: correctly rather than raising, but resolve() will tell you to stop using them.
DEPRECATED_ALIASES: dict[str, str] = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash",
}


def resolve(model: str) -> str:
    """Map a model id to the one that pricing is published under."""
    return DEPRECATED_ALIASES.get(model, model)


@dataclass(frozen=True)
class Usage:
    """Token counts for one request, as reported by the API.

    DeepSeek splits prompt tokens into cache hits and misses itself, so this
    mirrors that split rather than inventing one.
    """

    cache_miss_tokens: int = 0
    cache_hit_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def from_response(cls, usage: object) -> Usage:
        """Read an OpenAI-shaped ``response.usage``, tolerating absent cache fields.

        ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens`` are DeepSeek
        extensions, not part of the OpenAI schema. When they are missing, treat
        the whole prompt as a cache miss — the conservative reading, which
        over-states cost rather than quietly under-stating it.
        """
        hit = getattr(usage, "prompt_cache_hit_tokens", None)
        miss = getattr(usage, "prompt_cache_miss_tokens", None)
        if hit is None or miss is None:
            hit = 0
            miss = getattr(usage, "prompt_tokens", 0) or 0

        return cls(
            cache_miss_tokens=miss or 0,
            cache_hit_tokens=hit or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            cache_miss_tokens=self.cache_miss_tokens + other.cache_miss_tokens,
            cache_hit_tokens=self.cache_hit_tokens + other.cache_hit_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def prompt_tokens(self) -> int:
        """Everything the model read, cached or not."""
        return self.cache_miss_tokens + self.cache_hit_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of prompt tokens served from cache."""
        return self.cache_hit_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


def estimate_cost(model: str, usage: Usage) -> float:
    """Estimate what one request cost, in USD.

    Raises ``KeyError`` for an unknown model rather than guessing a rate — a
    silently wrong cost figure is worse than no figure.
    """
    resolved = resolve(model)
    try:
        hit_rate, miss_rate, out_rate = PRICING[resolved]
    except KeyError:
        raise KeyError(
            f"no pricing for {model!r}; add it to PRICING (have: {', '.join(sorted(PRICING))})"
        ) from None

    return (
        usage.cache_hit_tokens * hit_rate / 1_000_000
        + usage.cache_miss_tokens * miss_rate / 1_000_000
        + usage.output_tokens * out_rate / 1_000_000
    )
