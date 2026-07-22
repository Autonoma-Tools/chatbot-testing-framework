"""
tests/test_chatbot_responses.py
================================

Golden-dataset regression tests for a chatbot, using two complementary
assertion layers from chatbot_eval.judge:

1. semantic_similarity(actual, expected) - a fast embedding-based check that
   catches responses that drift far from the expected meaning.
2. llm_judge(question, expected, actual) - a slower model-graded check that
   catches responses that are semantically close but factually or tonally
   wrong (e.g. right shape of answer, wrong day of the week).

Replace `get_chatbot_response()` with a call into your real chatbot (an API
client, a driver for your chat UI, etc). Everything else - the dataset, the
parametrization, and the two-layer assertion - works unchanged.

Run with:
    OPENAI_API_KEY=sk-... pytest tests/test_chatbot_responses.py -v
"""

from __future__ import annotations

import pytest

from chatbot_eval.judge import llm_judge, semantic_similarity

# Similarity below this threshold fails a test outright, before the LLM judge
# is even consulted. Tune this against real production traffic - start
# conservative, log actual scores for the first few weeks, and adjust based
# on where real failures cluster rather than guessing at a round number.
SIMILARITY_THRESHOLD = 0.75

# The golden dataset: each entry is one (question, expected_answer) pair the
# chatbot must handle correctly. Treat this list as a versioned artifact -
# add a new entry every time a real production failure mode is discovered,
# so the same regression can never sneak back in silently.
GOLDEN_DATASET = [
    {
        "id": "reschedule-appointment",
        "question": "can I move my appointment to Thursday",
        "expected": "Sure, I've moved your appointment to Thursday.",
    },
    {
        "id": "cancel-appointment",
        "question": "I need to cancel my appointment",
        "expected": "Your appointment has been cancelled. Let me know if you'd like to rebook.",
    },
    {
        "id": "business-hours",
        "question": "what time do you open",
        "expected": "We're open from 9am to 6pm, Monday through Friday.",
    },
    {
        "id": "refund-policy",
        "question": "can I get a refund if I cancel within 24 hours",
        "expected": "Yes, cancellations made at least 24 hours in advance are fully refundable.",
    },
    {
        "id": "out-of-scope-request",
        "question": "can you write my resignation letter for me",
        "expected": "I can only help with scheduling and account questions for this business, so I can't help with that.",
    },
]


def get_chatbot_response(question: str) -> str:
    """Stub chatbot client. Replace this with a call to your real chatbot,
    e.g. `requests.post(CHATBOT_API_URL, json={"message": question}).json()["reply"]`
    or a driver call against your chat UI.

    This stub echoes a canned response per question so the test file is
    runnable end-to-end (against the real OpenAI-backed assertion layer)
    before any real chatbot integration exists.
    """
    canned_responses = {
        "can I move my appointment to Thursday": "Done, your new slot is Thursday afternoon.",
        "I need to cancel my appointment": "Your appointment has been cancelled. Want me to help you rebook?",
        "what time do you open": "We open at 9am and close at 6pm on weekdays.",
        "can I get a refund if I cancel within 24 hours": (
            "Yes, as long as you cancel 24 hours ahead of time, you'll get a full refund."
        ),
        "can you write my resignation letter for me": (
            "That's outside what I can help with here, I'm only set up for "
            "scheduling and account questions."
        ),
    }
    return canned_responses.get(question, "I'm not sure how to help with that.")


@pytest.mark.parametrize(
    "case",
    GOLDEN_DATASET,
    ids=[case["id"] for case in GOLDEN_DATASET],
)
def test_chatbot_response_matches_golden_answer(case: dict) -> None:
    question = case["question"]
    expected = case["expected"]

    actual = get_chatbot_response(question)

    similarity = semantic_similarity(actual, expected)
    assert similarity >= SIMILARITY_THRESHOLD, (
        f"[{case['id']}] semantic similarity {similarity:.2f} is below "
        f"threshold {SIMILARITY_THRESHOLD}.\n"
        f"  question: {question}\n  expected: {expected}\n  actual:   {actual}"
    )

    verdict = llm_judge(question=question, expected=expected, actual=actual)
    assert verdict.passed, (
        f"[{case['id']}] LLM judge returned '{verdict.label}': {verdict.reason}\n"
        f"  question: {question}\n  expected: {expected}\n  actual:   {actual}"
    )
