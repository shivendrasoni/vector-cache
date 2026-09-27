from .base import BaseJudge, Candidate, JudgeResult

try:
    from .typesafe_jev import JevJudge
except ImportError:
    JevJudge = None
