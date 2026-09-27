from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Candidate:
    query: str
    context: str
    response: Any


@dataclass
class JudgeResult:
    # Index into the candidates list of the accepted candidate, None when every candidate is rejected.
    index: Optional[int]
    probabilities: List[float] = field(default_factory=list)
    # True when the best candidate landed between "clear no" and the accept threshold.
    uncertain: bool = False
    error: Optional[str] = None


class BaseJudge(ABC):
    @abstractmethod
    def judge(self, query: str, context: str, candidates: List[Candidate]) -> JudgeResult:
        """Decide which cached candidate (if any) can be served for the new query."""
        pass

    def is_cacheable(self, query: str, context: str, response: Any) -> bool:
        """Decide whether a fresh response is safe to store. Default: always cache."""
        return True
