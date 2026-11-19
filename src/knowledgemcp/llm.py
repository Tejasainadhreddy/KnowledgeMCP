"""Language models. BedrockLLM calls Claude through the Bedrock Converse API.

ExtractiveLLM is an offline stand-in for tests and local demos: it copies the best-matching sentence
from the sources and cites it. It does not "understand" anything; it just lets the pipeline run
without AWS. Never report eval numbers from it as model quality.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Protocol

from .embeddings import rough_tokens
from .rerank import terms

NOT_FOUND = "NOT_FOUND"


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    def generate(self, system: str, prompt: str, max_tokens: int = 600) -> LLMResult: ...


class BedrockLLM:
    def __init__(self, client, model_id: str):
        self.client = client
        self.model_id = model_id

    def generate(self, system: str, prompt: str, max_tokens: int = 600) -> LLMResult:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        text = "".join(part.get("text", "") for part in resp["output"]["message"]["content"])
        usage = resp.get("usage", {})
        return LLMResult(text.strip(), int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0)))


_SOURCE = re.compile(r'<source id="S(\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class ExtractiveLLM:
    def __init__(self, min_coverage: float = 0.5):
        self.min_coverage = min_coverage

    def generate(self, system: str, prompt: str, max_tokens: int = 600) -> LLMResult:
        question = prompt.split("Question:", 1)[1].split("\n", 1)[0] if "Question:" in prompt else prompt
        q = terms(question)
        best, best_score = None, 0.0
        for sid, body in _SOURCE.findall(prompt):
            for sentence in _SENTENCE.split(html.unescape(body)):
                sentence = sentence.strip()
                if not sentence or sentence.startswith("[") or len(sentence) < 15:
                    continue
                score = len(q & terms(sentence)) / max(1, len(q))
                if score > best_score:
                    best, best_score = (sid, sentence), score
        if best is None or best_score < self.min_coverage:
            text = NOT_FOUND
        else:
            sentence = best[1].rstrip(".")
            text = f"{sentence} [S{best[0]}]."
        return LLMResult(text, rough_tokens([system, prompt]), rough_tokens([text]))
