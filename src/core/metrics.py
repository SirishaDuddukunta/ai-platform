# WHAT THIS FILE DOES:
# Defines a small data class that describes "how much did one LLM call cost?"
# Every time we call the LLM, the API tells us how many tokens it used.
# Tokens = the chunks of text the model reads/writes (roughly 1 token = 3/4 of a word).
# Providers charge money PER TOKEN, so tokens → dollars is simple multiplication.
# BaseModel is Pydantic's "smart class": it validates types automatically
# (e.g. if something tries to put a string where an int belongs, it errors loudly).
from pydantic import BaseModel

# --- PRICING CONSTANTS ---
# Groq charges $0.59 for every 1 MILLION tokens the model READS (input/prompt).
# Dividing by 1,000,000 gives us the price of ONE single token.
INPUT_COST_PER_TOKEN = 0.59 / 1_000_000   # = $0.00000059 per input token

# Groq charges $0.79 for every 1 MILLION tokens the model WRITES (output/completion).
OUTPUT_COST_PER_TOKEN = 0.79 / 1_000_000  # = $0.00000079 per output token
# NOTE: if you ever switch models, these two numbers are the ONLY thing to update.


class TokenMetrics(BaseModel):
    # How many tokens were in everything we SENT to the model (our prompt).
    prompt_tokens: int = 0        # "= 0" means: if nobody provides a value, default to 0

    # How many tokens the model WROTE back to us (its answer).
    completion_tokens: int = 0

    # prompt_tokens + completion_tokens (the API usually gives us this directly).
    total_tokens: int = 0

    # Which company's API we used (useful later if you add OpenAI/Anthropic).
    provider: str = "groq"

    # THE NEW FIELD: the actual dollar cost of this one call.
    # This is the field recruiters look for — "no cost tracking = red flag".
    cost_usd: float = 0.0

    # @classmethod means: this function belongs to the CLASS, not to one object.
    # You call it like TokenMetrics.from_usage(...) — it BUILDS a TokenMetrics for you.
    # Think of it as a smart constructor/factory.
    @classmethod
    def from_usage(cls, usage, provider: str = "groq") -> "TokenMetrics":
        """Build a TokenMetrics object from the `usage` object the Groq API returns."""

        # getattr(obj, "name", default) safely reads obj.name.
        # If the attribute doesn't exist, it returns the default (0) instead of crashing.
        prompt = getattr(usage, "prompt_tokens", 0)          # tokens we sent
        completion = getattr(usage, "completion_tokens", 0)  # tokens the model wrote

        # THE COST MATH: (input tokens × input price) + (output tokens × output price).
        # round(..., 8) keeps 8 decimal places — costs per call are tiny fractions of a cent.
        cost = round(
            prompt * INPUT_COST_PER_TOKEN + completion * OUTPUT_COST_PER_TOKEN, 8
        )

        # cls(...) creates the TokenMetrics object (cls = "this class").
        # We fill in every field explicitly.
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            # If the API didn't give total_tokens, compute it ourselves as a fallback.
            total_tokens=getattr(usage, "total_tokens", prompt + completion),
            provider=provider,
            cost_usd=cost,
        )