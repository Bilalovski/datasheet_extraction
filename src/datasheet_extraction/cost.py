"""Token accounting and cost estimation.

Quality numbers without cost numbers don't decide anything: the interesting
question is never "is Opus better than Haiku" (it is) but "is it better by
enough to justify 5x the input price on this task". So every extraction carries
its usage, and the ablation reports accuracy next to what the accuracy cost.
"""

from __future__ import annotations

from dataclasses import dataclass

#: USD per million tokens, as published on 2026-06-24. Rates move; re-check
#: https://platform.claude.com/docs/en/about-claude/models/overview before
#: quoting any figure this module produces.
#:
#: Sonnet 5 additionally has introductory pricing ($2.00/$10.00 per MTok)
#: through 2026-08-31. These are the standard rates, so a Sonnet 5 run costs
#: less than reported until then rather than more.
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

#: Cache reads bill at roughly a tenth of the base input rate.
CACHE_READ_MULTIPLIER = 0.1

#: Cache writes bill at 1.25x the base input rate for the default 5-minute TTL.
CACHE_WRITE_MULTIPLIER = 1.25

#: The Batch API halves both input and output rates.
BATCH_MULTIPLIER = 0.5


@dataclass(frozen=True)
class Usage:
    """Token counts for one request, as reported by the API."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_response(cls, usage: object) -> Usage:
        """Read an SDK ``response.usage`` object, tolerating absent cache fields."""
        return cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0)
            or 0,
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
        )

    @property
    def total_prompt_tokens(self) -> int:
        """Everything the model read, cached or not.

        ``input_tokens`` alone is the uncached remainder — reading it as the
        prompt size understates a well-cached run by whatever the cache served.
        """
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


def estimate_cost(model: str, usage: Usage, batch: bool = False) -> float:
    """Estimate what one request cost, in USD.

    Raises ``KeyError`` for an unknown model rather than guessing a rate — a
    silently wrong cost figure is worse than no figure.
    """
    try:
        input_rate, output_rate = PRICING[model]
    except KeyError:
        raise KeyError(
            f"no pricing for {model!r}; add it to PRICING (rates: {', '.join(sorted(PRICING))})"
        ) from None

    discount = BATCH_MULTIPLIER if batch else 1.0
    per_token_in = input_rate / 1_000_000 * discount
    per_token_out = output_rate / 1_000_000 * discount

    return (
        usage.input_tokens * per_token_in
        + usage.cache_read_input_tokens * per_token_in * CACHE_READ_MULTIPLIER
        + usage.cache_creation_input_tokens * per_token_in * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * per_token_out
    )
