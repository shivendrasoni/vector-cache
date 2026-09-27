import logging
import os
from typing import Any, List, Optional

import requests

from vector_cache.judges.base import BaseJudge, Candidate, JudgeResult

logger = logging.getLogger(__name__)

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

QUERY_TASK = ("Would a correct answer to the cached query also fully and correctly answer the user's query, "
              "given both contexts?")
QUERY_CRITERIA = {
    "true": "Both ask for the same information with the same constraints; only the wording differs",
    "false": "They differ in subject, entity, constraint, scope, time frame or requested detail",
}

ANSWER_TASK = "Does the cached response fully and correctly answer the user's query as asked, given the context?"
ANSWER_CRITERIA = {
    "true": "The response addresses exactly what the query asks, with nothing missing or about something else",
    "false": "The response answers a different question, misses part of the query, or would mislead the user",
}

NEGATION_TASK = ("Does the user's query differ from the cached query by a negation or exclusion "
                 "(for example with vs without, include vs exclude, can vs cannot)?")

CACHEABLE_QUESTIONS = {
    "time_sensitive": {
        "type": "noul",
        "instructions": "Would the correct answer to this query change over hours or days "
                        "(news, prices, weather, 'today', 'latest', 'current')?",
    },
    "personal": {
        "type": "noul",
        "instructions": "Is the query about the user's own private data (their account, order, orders, "
                        "messages, schedule)?",
    },
    "failed_response": {
        "type": "noul",
        "instructions": "Is the response a refusal, an error message, or a failure to answer the query?",
    },
}


class JevJudge(BaseJudge):
    """Cache-hit verifier backed by TypeSafe's Jev System One model.

    mode="query"  compares the new query with each cached query (short state, cheapest).
    mode="answer" asks whether each cached response answers the new query (more robust, more tokens).

    Any API failure is treated as "no match", so an outage degrades to cache misses, never to wrong hits.

    api_key, model and url fall back to TYPESAFE_API_KEY, TYPESAFE_MODEL and TYPESAFE_API_URL. Overriding
    the url and model lets this judge target any System One model served behind the same /v1/systemone
    request shape; models with a different API should implement BaseJudge instead.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, mode: str = "query",
                 accept: float = 0.9, reject: float = 0.5, negation_reject: float = 0.5,
                 timeout: float = 1.0, url: Optional[str] = None, session: Optional[requests.Session] = None):
        if mode not in ("query", "answer"):
            raise ValueError("mode must be 'query' or 'answer'")
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("TypeSafe API key missing: pass api_key or set TYPESAFE_API_KEY")
        self.model = model or os.environ.get("TYPESAFE_MODEL") or DEFAULT_MODEL
        self.mode = mode
        self.accept = accept
        self.reject = reject
        self.negation_reject = negation_reject
        self.timeout = timeout
        self.url = url or os.environ.get("TYPESAFE_API_URL") or TYPESAFE_URL
        # A shared session keeps the TLS connection alive, which matters more than model time on far regions.
        self.session = session or requests.Session()

    def _call(self, state: Any, questions: dict) -> dict:
        response = self.session.post(
            self.url,
            json={"state": state, "model": self.model, "questions": questions},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["answers"]

    def _build_questions(self, candidates: List[Candidate]) -> dict:
        questions = {}
        for i, candidate in enumerate(candidates):
            if self.mode == "query":
                instructions = {"task": QUERY_TASK, "cached_query": candidate.query,
                                "cached_context": candidate.context}
                criteria = QUERY_CRITERIA
            else:
                instructions = {"task": ANSWER_TASK, "cached_response": str(candidate.response)}
                criteria = ANSWER_CRITERIA
            questions[f"match_{i}"] = {"type": "noul", "instructions": instructions, "criteria": criteria}
            questions[f"negation_{i}"] = {
                "type": "noul",
                "instructions": {"task": NEGATION_TASK, "cached_query": candidate.query},
            }
        return questions

    def judge(self, query: str, context: str, candidates: List[Candidate]) -> JudgeResult:
        if not candidates:
            return JudgeResult(index=None)
        try:
            answers = self._call({"user_query": query, "context": context}, self._build_questions(candidates))
            probabilities = []
            for i in range(len(candidates)):
                match = float(answers[f"match_{i}"]["noul"])
                # A negation flip ("with" vs "without") vetoes the match regardless of its score.
                if float(answers[f"negation_{i}"]["noul"]) >= self.negation_reject:
                    match = 0.0
                probabilities.append(match)
        except Exception as e:
            logger.warning("Jev judge failed, treating as cache miss: %s", e)
            return JudgeResult(index=None, error=str(e))

        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        best_p = probabilities[best]
        if best_p >= self.accept:
            return JudgeResult(index=best, probabilities=probabilities)
        return JudgeResult(index=None, probabilities=probabilities, uncertain=best_p >= self.reject)

    def is_cacheable(self, query: str, context: str, response: Any) -> bool:
        try:
            answers = self._call({"query": query, "context": context, "response": str(response)},
                                 CACHEABLE_QUESTIONS)
            return all(float(answers[name]["noul"]) < self.reject for name in CACHEABLE_QUESTIONS)
        except Exception as e:
            # Fail closed: an unchecked response is not stored.
            logger.warning("Jev cacheability check failed, skipping cache insert: %s", e)
            return False
