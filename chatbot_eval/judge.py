"""
chatbot_eval.judge
==================

Assertion helpers for testing chatbot responses that are correct in meaning
but never byte-identical to a fixed expected string.

Two functions are exposed:

- semantic_similarity(a, b) -> float
    Embeds both strings with OpenAI's embeddings API and returns the cosine
    similarity between the two vectors, in the range [0.0, 1.0]. A high score
    means the two strings are semantically close even if worded differently.

- llm_judge(question, expected, actual) -> Verdict
    Sends the question, the expected answer, and the actual chatbot response
    to a judge model with a scoring rubric, and returns a structured Verdict
    (label + reason). Use this when semantic similarity alone isn't a strong
    enough signal, e.g. when a wrong response is fluent and on-topic but
    states the wrong fact.

Both functions require the OPENAI_API_KEY environment variable to be set
before they are called. Importing this module does not require the key -
only calling semantic_similarity() or llm_judge() does.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Literal, Optional

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
JUDGE_MODEL = "gpt-4o-mini"

VerdictLabel = Literal["correct", "partially_correct", "wrong"]

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazily construct the OpenAI client so importing this module never
    requires OPENAI_API_KEY to already be set - only calling into it does."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running any test "
                "that calls semantic_similarity() or llm_judge()."
            )
        _client = OpenAI(api_key=api_key)
    return _client


@dataclass(frozen=True)
class Verdict:
    """Structured result of an LLM-judge call."""

    label: VerdictLabel
    reason: str

    @property
    def passed(self) -> bool:
        """True for a fully correct verdict. `partially_correct` is treated
        as a failure by default - loosen this in the caller if a suite wants
        partial credit to count as a pass."""
        return self.label == "correct"


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_similarity(a: str, b: str) -> float:
    """Return a 0.0-1.0 cosine similarity score between two strings' embeddings.

    Raises RuntimeError if OPENAI_API_KEY is unset. Raises openai.OpenAIError
    subclasses on API failures (rate limit, network, auth).
    """
    if not a.strip() or not b.strip():
        return 0.0

    client = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[a, b])
    vec_a = response.data[0].embedding
    vec_b = response.data[1].embedding

    similarity = _cosine_similarity(vec_a, vec_b)
    # Guard against floating point drift pushing a near-perfect match
    # slightly outside the [0, 1] range.
    return max(0.0, min(1.0, similarity))


_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a chatbot testing framework.
You will be given a user question, the expected (golden) answer, and the chatbot's
actual response. Decide whether the actual response is an acceptable answer to the
question given the expected answer as the source of truth.

Score using exactly one of these labels:
- "correct": the actual response conveys the same facts and intent as the expected
  answer, even if phrased completely differently.
- "partially_correct": the actual response is on-topic and not factually wrong, but
  omits something the expected answer requires, or hedges where the expected answer
  is definitive.
- "wrong": the actual response contradicts the expected answer, states a different
  fact (wrong date, wrong price, wrong action), answers a different question, or is
  off-topic.

Respond with ONLY a JSON object of the shape:
{"label": "correct" | "partially_correct" | "wrong", "reason": "<one sentence>"}

Do not include any other text, markdown, or code fences in your response."""


def llm_judge(question: str, expected: str, actual: str) -> Verdict:
    """Ask a judge model to score `actual` against `expected` for `question`.

    Returns a Verdict with a label of "correct", "partially_correct", or
    "wrong", plus a one-sentence reason. Raises RuntimeError if OPENAI_API_KEY
    is unset, or ValueError if the judge model's response can't be parsed as
    the expected JSON shape.
    """
    client = _get_client()

    user_prompt = (
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"Actual response: {actual}"
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        label = parsed["label"]
        reason = parsed["reason"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            "llm_judge: could not parse judge model output as the expected "
            f"JSON verdict. Raw output: {raw!r}"
        ) from exc

    if label not in ("correct", "partially_correct", "wrong"):
        raise ValueError(f"llm_judge: unexpected label {label!r} in judge output")

    return Verdict(label=label, reason=reason)
