import requests

from vector_cache.cache_storage.lru import LRUCache
from vector_cache.embedding.base_embedding import BaseEmbedding
from vector_cache.judges import JevJudge
from vector_cache.judges.base import BaseJudge, Candidate, JudgeResult
from vector_cache.main import VectorCache, decode_entry, encode_entry
from vector_cache.utils.guards import negations_match, numbers_match
from vector_cache.vector_stores.base import VectorStoreInterface


class FakeEmbedding(BaseEmbedding):
    def get_embeddings(self, text):
        return [0.0]

    @property
    def dimension(self):
        return 1


class FakeVectorStore(VectorStoreInterface):
    """Returns the scripted similarities for whatever ids have been added, in insertion order."""

    def __init__(self, similarities):
        self.similarities = similarities
        self.ids = []

    def add(self, embedding, **kwargs):
        self.ids.append(f"id{len(self.ids)}")
        return self.ids[-1]

    def search(self, embedding, top_n=1, include_distances=True, **kwargs):
        n = min(top_n, len(self.ids))
        return self.ids[:n], self.similarities[:n]


class ScriptedJudge(BaseJudge):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def judge(self, query, context, candidates):
        self.calls.append((query, candidates))
        return self.result


def make_cache(similarities, judge, **kwargs):
    return VectorCache(FakeEmbedding(), LRUCache(), FakeVectorStore(similarities), judge=judge, **kwargs)


def test_entry_roundtrip_and_legacy():
    assert decode_entry(encode_entry("q", "c", "r")) == ("q", "c", "r")
    assert decode_entry("plain legacy response") == (None, None, "plain legacy response")
    assert decode_entry('{"some": "json answer"}') == (None, None, '{"some": "json answer"}')
    obj = object()
    assert decode_entry(encode_entry("q", "c", obj)) == (None, None, obj)


def test_without_judge_behaviour_unchanged():
    cache = VectorCache(FakeEmbedding(), LRUCache(), FakeVectorStore([0.95]), initial_similarity_threshold=0.9)
    cache.add_query_to_index("vegan cake recipe", "answer")
    assert cache.find_similar_queries("vegan cake recipe")[0] == "answer"


def test_grey_zone_hit_uses_judge():
    judge = ScriptedJudge(JudgeResult(index=0, probabilities=[0.97]))
    cache = make_cache([0.9], judge)
    cache.add_query_to_index("vegan chocolate cake recipe", "answer")
    result, similarity = cache.find_similar_queries("chocolate cake without animal products")
    assert result == "answer" and similarity == 0.9
    assert cache.judge_calls == 1 and cache.cache_hits == 1
    assert judge.calls[0][1][0].query == "vegan chocolate cake recipe"


def test_grey_zone_reject():
    judge = ScriptedJudge(JudgeResult(index=None, probabilities=[0.2]))
    cache = make_cache([0.9], judge)
    cache.add_query_to_index("vegan chocolate cake recipe", "answer")
    assert cache.find_similar_queries("chocolate cake with eggs")[0] is None
    assert cache.judge_rejects == 1 and cache.cache_hits == 0


def test_band_edges_skip_judge():
    judge = ScriptedJudge(JudgeResult(index=0))
    high = make_cache([0.96], judge)  # just above the default 0.95 ceiling
    high.add_query_to_index("q", "answer")
    assert high.find_similar_queries("q")[0] == "answer"
    low = make_cache([0.5], judge)
    low.add_query_to_index("q", "answer")
    assert low.find_similar_queries("other")[0] is None
    assert judge.calls == []


def test_numeric_guard_blocks_before_judge():
    judge = ScriptedJudge(JudgeResult(index=0))
    cache = make_cache([0.995], judge)
    cache.add_query_to_index("best laptops under $500", "answer")
    assert cache.find_similar_queries("best laptops under $300")[0] is None
    assert judge.calls == []
    assert numbers_match("sold 1,000 units", "sold 1000 units")


def test_negation_guard():
    assert not negations_match("chocolate cake with gluten", "chocolate cake without gluten")
    assert not negations_match("can I take ibuprofen", "why can't I take ibuprofen")
    assert negations_match("which animals are not mammals", "what animals aren't mammals")
    assert negations_match("vegan cake recipe", "plant based cake recipe")


def test_negation_mismatch_above_ceiling_goes_to_judge():
    judge = ScriptedJudge(JudgeResult(index=None, probabilities=[0.1]))
    cache = make_cache([0.97], judge)
    cache.add_query_to_index("chocolate cake with gluten", "answer")
    assert cache.find_similar_queries("chocolate cake without gluten")[0] is None
    assert len(judge.calls) == 1 and cache.judge_rejects == 1


def test_legacy_entries_are_not_judged():
    judge = ScriptedJudge(JudgeResult(index=0))
    cache = make_cache([0.9], judge)
    key = cache.vector_store.add([0.0])
    cache.db.set_response(key, "legacy answer")
    assert cache.find_similar_queries("q")[0] is None
    assert judge.calls == []


def test_check_cacheable_skips_insert():
    class NoCache(ScriptedJudge):
        def is_cacheable(self, query, context, response):
            return False

    cache = make_cache([0.9], NoCache(JudgeResult(index=0)), check_cacheable=True)
    cache.add_query_to_index("weather today", "sunny")
    assert cache.vector_store.ids == []


class FakeResponse:
    def __init__(self, answers):
        self.answers = answers

    def raise_for_status(self):
        pass

    def json(self):
        return {"answers": {k: {"type": "noul", "noul": v} for k, v in self.answers.items()}}


class FakeSession:
    def __init__(self, answers=None, exc=None):
        self.answers, self.exc, self.payloads = answers, exc, []

    def post(self, url, json, headers, timeout):
        self.payloads.append(json)
        if self.exc:
            raise self.exc
        return FakeResponse(self.answers)


CANDIDATES = [Candidate("cached a", "", "ra"), Candidate("cached b", "", "rb")]


def test_jev_picks_best_candidate_and_builds_request():
    session = FakeSession({"match_0": 0.3, "negation_0": 0.1, "match_1": 0.95, "negation_1": 0.1})
    result = JevJudge(api_key="k", session=session).judge("new", "ctx", CANDIDATES)
    assert result.index == 1
    payload = session.payloads[0]
    assert payload["model"] == "jev-latest"
    assert payload["state"] == {"user_query": "new", "context": "ctx"}
    assert payload["questions"]["match_1"]["instructions"]["cached_query"] == "cached b"


def test_jev_negation_vetoes_match():
    session = FakeSession({"match_0": 0.97, "negation_0": 0.8, "match_1": 0.1, "negation_1": 0.0})
    result = JevJudge(api_key="k", session=session).judge("cake with gluten", "", CANDIDATES)
    assert result.index is None


def test_jev_uncertain_band():
    session = FakeSession({"match_0": 0.7, "negation_0": 0.0, "match_1": 0.1, "negation_1": 0.0})
    result = JevJudge(api_key="k", session=session).judge("q", "", CANDIDATES)
    assert result.index is None and result.uncertain


def test_jev_failure_is_a_miss_and_not_cacheable():
    judge = JevJudge(api_key="k", session=FakeSession(exc=requests.Timeout("slow")))
    result = judge.judge("q", "", CANDIDATES)
    assert result.index is None and "slow" in result.error
    assert judge.is_cacheable("q", "", "r") is False


def test_jev_answer_mode_sends_response():
    session = FakeSession({"match_0": 0.95, "negation_0": 0.0})
    JevJudge(api_key="k", mode="answer", session=session).judge("q", "", CANDIDATES[:1])
    assert session.payloads[0]["questions"]["match_0"]["instructions"]["cached_response"] == "ra"


def test_jev_config_from_env(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    monkeypatch.setenv("TYPESAFE_MODEL", "other-system-one")
    monkeypatch.setenv("TYPESAFE_API_URL", "https://example.com/v1/systemone")
    judge = JevJudge(session=FakeSession({}))
    assert (judge.api_key, judge.model, judge.url) == ("env-key", "other-system-one", "https://example.com/v1/systemone")
    assert JevJudge(model="jev-1.13.0", session=FakeSession({})).model == "jev-1.13.0"


def test_jev_defaults_without_env(monkeypatch):
    for name in ("TYPESAFE_MODEL", "TYPESAFE_API_URL"):
        monkeypatch.delenv(name, raising=False)
    judge = JevJudge(api_key="k", session=FakeSession({}))
    assert judge.model == "jev-latest" and judge.url == "https://api.typesafe.ai/v1/systemone"
