"""Reproducible diagnostic sweep for the semantic cache; no secret is stored here.

From the repo root: pip install -e '.[sentence-transformers,typesafe]'
                    pip install -r eval/judge/requirements.txt
                    python eval/judge/run.py --mode cosine --output cosine.json
                    TYPESAFE_API_KEY=<set in your shell securely> python eval/judge/run.py --mode jev --output jev.json
The Jev command incurs API charges. Never commit its output if it contains private queries.
"""
import argparse
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FLOORS = (0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 1.00)


def percentile(values, p):
    values = sorted(values)
    return values[int(p * (len(values) - 1))] if values else None


def summarize(details):
    counts = {name: 0 for name in ("TP", "TN", "FP", "FN")}
    for row in details:
        hit = row["observed_hit"]
        expected = row["expected_hit"]
        counts[("TP" if hit else "FN") if expected else ("FP" if hit else "TN")] += 1
    hits = counts["TP"] + counts["FP"]
    positives = counts["TP"] + counts["FN"]
    times = [row["latency_ms"] for row in details]
    judged = [row["latency_ms"] for row in details if row["judge_called"]]
    return {
        **counts, "correct": counts["TP"] + counts["TN"], "total": len(details),
        "precision": counts["TP"] / hits if hits else None,
        "recall": counts["TP"] / positives if positives else None,
        "judge_calls": sum(row["judge_called"] for row in details),
        "latency_ms_p50": statistics.median(times),
        "latency_ms_p95": percentile(times, .95),
        "judged_latency_ms_p50": statistics.median(judged) if judged else None,
        "judged_latency_ms_p95": percentile(judged, .95),
    }


class ChromaCosineStore:
    """Small evaluation-only adapter, avoiding the Chroma 1.5.9 import in chroma_db.py.

    This does not fix the library's production compatibility issue.
    """
    def __init__(self, client):
        self.collection = client.create_collection(
            name="judge_eval_" + uuid.uuid4().hex, metadata={"hnsw:space": "cosine"}
        )
        self.inserted_id = None

    def add(self, embedding, **kwargs):
        self.inserted_id = uuid.uuid4().hex
        self.collection.add(ids=[self.inserted_id], embeddings=[embedding.tolist() if hasattr(embedding, "tolist") else embedding])
        return self.inserted_id

    def search(self, embedding, top_n=1, include_distances=True, **kwargs):
        result = self.collection.query(
            query_embeddings=[embedding.tolist() if hasattr(embedding, "tolist") else embedding],
            n_results=top_n, include=["distances"] if include_distances else [],
        )
        ids = result["ids"][0]
        return ids, ([1 - float(d) for d in result["distances"][0]] if include_distances else [])


def run(mode, model_name, floors, ceilings, timeout):
    import chromadb
    from vector_cache.cache_storage.lru import LRUCache
    from vector_cache.embedding.sentence_bert import SentenceBertEmbeddings
    from vector_cache.judges.typesafe_jev import JevJudge
    from vector_cache.main import VectorCache, encode_entry

    if mode == "jev" and not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("Set TYPESAFE_API_KEY in the process environment to run live Jev; do not add it to a file")
    cases = json.loads((Path(__file__).parent / "cases.json").read_text())
    embedder = SentenceBertEmbeddings(model_name)
    client = chromadb.EphemeralClient()
    prepared = []
    # One cached query per independent collection; no previous query can contaminate the next.
    for case in cases:
        store = ChromaCosineStore(client)
        seed = embedder.get_embeddings(": " + case["seed_query"])
        key = store.add(seed)
        prepared.append((case, store, key))

    results = []
    for floor in floors:
        for ceiling in ([None] if mode == "cosine" else ceilings):
            details = []
            for case, store, key in prepared:
                db = LRUCache()
                db.set_response(key, encode_entry(case["seed_query"], "", "placeholder response"))
                judge = JevJudge(model="jev-latest", timeout=timeout) if mode == "jev" else None
                cache = VectorCache(
                    embedder, db, store,
                    initial_similarity_threshold=floor, judge=judge,
                    judge_band=(floor, ceiling if ceiling is not None else 0.95), judge_top_k=1,
                )
                start = time.monotonic()
                response, similarity = cache.find_similar_queries(case["new_query"])
                details.append({
                    "id": case["id"], "expected_hit": case["expected_hit"],
                    "observed_hit": response is not None, "similarity": similarity,
                    "judge_called": cache.judge_calls,
                    "latency_ms": round((time.monotonic() - start) * 1000, 2),
                })
            results.append({"floor": floor, "ceiling": ceiling, "metrics": summarize(details), "cases": details})
            print(f"floor={floor:.2f} ceiling={ceiling} {results[-1]['metrics']}", file=sys.stderr)
    return {"mode": mode, "model": model_name, "judge_model": "jev-latest" if mode == "jev" else None,
            "timeout_s": timeout if mode == "jev" else None, "results": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("cosine", "jev"), required=True)
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--floors", nargs="+", type=float, default=FLOORS)
    p.add_argument("--ceilings", nargs="+", type=float, default=(.95, 1.0))
    p.add_argument("--timeout", type=float, default=8)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "jev" and any(floor > ceiling for floor in a.floors for ceiling in a.ceilings):
        print("Warning: floor > ceiling rows have no judge calls; they only show the direct-hit path", file=sys.stderr)
    a.output.write_text(json.dumps(run(a.mode, a.model, a.floors, a.ceilings, a.timeout), indent=2) + "\n")
