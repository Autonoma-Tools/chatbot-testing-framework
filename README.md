# chatbot-testing-framework

A minimal chatbot testing framework in Python: a pytest wrapper with a semantic-similarity and LLM-judge assertion helper for evaluating non-deterministic chatbot responses against a golden dataset.

> Companion code for the Autonoma blog post: **[Build a Chatbot Testing Framework in 5 Steps](https://getautonoma.com/blog/chatbot-testing-framework)**

## Requirements

Python 3.10+, and an `OPENAI_API_KEY` in your environment (used for both the embeddings call in `semantic_similarity()` and the chat-completions call in `llm_judge()`).

## Quickstart

```bash
git clone https://github.com/Autonoma-Tools/chatbot-testing-framework.git
cd chatbot-testing-framework
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
pytest tests/test_chatbot_responses.py -v
```

## Project structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── chatbot_eval/
│   ├── __init__.py
│   └── judge.py              # semantic_similarity() + llm_judge() assertion helpers
├── tests/
│   └── test_chatbot_responses.py   # golden-dataset pytest suite
└── examples/
    └── run_golden_tests.sh   # one-shot script: install deps + run the suite
```

- `chatbot_eval/` — the library module. `semantic_similarity(a, b)` embeds two strings with OpenAI's embeddings API and returns a 0-1 cosine similarity score. `llm_judge(question, expected, actual)` sends a scoring rubric to a judge model and returns a structured verdict (`correct` / `partially_correct` / `wrong`) plus a one-sentence reason.
- `tests/` — a small inline golden dataset of `(question, expected_answer)` pairs, parametrized into individual pytest cases. Each case calls a stub `get_chatbot_response()` you replace with your real chatbot client, then asserts the response clears both the similarity threshold and the LLM-judge verdict.
- `examples/` — `run_golden_tests.sh` installs dependencies and runs the suite end-to-end in one command.

## How the two assertion layers work together

An assertion like `assert response == expected` only works on deterministic code. Chatbot responses are never byte-identical between runs, so this framework runs two checks in sequence instead:

1. **Semantic similarity** (fast, cheap) — embeds the actual and expected response and measures cosine distance. Catches responses that drift far from the expected meaning.
2. **LLM judge** (slower, costs a model call) — sends the question, expected answer, and actual response to a judge model with a rubric. Catches responses that are semantically close but factually or tonally wrong (right shape of answer, wrong fact).

Tune `SIMILARITY_THRESHOLD` in `tests/test_chatbot_responses.py` against real production traffic: start conservative, log actual scores for the first few weeks, and adjust based on where real failures cluster rather than guessing at a round number upfront.

## About

This repository is maintained by [Autonoma](https://getautonoma.com) as reference material for the linked blog post. Autonoma builds autonomous AI agents that plan, execute, and maintain end-to-end tests directly from your codebase.

If something here is wrong, out of date, or unclear, please [open an issue](https://github.com/Autonoma-Tools/chatbot-testing-framework/issues/new).

## License

Released under the [MIT License](./LICENSE) © 2026 Autonoma Labs.
