# Semantic-cache judge diagnostic

This is a **small illustrative test**, not a production benchmark. The 23 hand-labelled pairs in `cases.json` contain 12 valid reuses and 11 unsafe near misses. Three rubrics guide interpretation: safe hit reuse (TP/FN), unsafe-hit avoidance (TN/FP), and combined tagging plus judge-call/latency cost. The cached response is a placeholder; answer correctness is not measured.

## Run

```sh
python -m pip install -e '.[sentence-transformers,typesafe]'
python -m pip install -r eval/judge/requirements.txt
python eval/judge/run.py --mode cosine --output /tmp/cosine.json
# Set TYPESAFE_API_KEY securely in the process environment first. Do not commit it.
python eval/judge/run.py --mode jev --output /tmp/jev.json
```

The live command makes billable API requests. It uses `jev-latest`, query mode, an 8-second timeout, floors 0.80 through 1.00, and ceilings 0.95 and 1.0. Two different thresholds are involved: the **floor** is the minimum cosine score eligible for a hit (the 1.0 floor is a degenerate sweep endpoint and allows no meaningful hits); the **bypass ceiling** is the score above which the code directly serves the cached response without Jev. Ceiling 1.0 does NOT mean "identical strings only" - with a .85 floor, scores from .85 up to 1.0 go to the judge. The floor must not exceed the ceiling: rows above 0.95 with ceiling 0.95 make no Jev calls and do not test judge quality. No credentials are stored or printed. To compare only meaningful bands, use `--floors .80 .85 .90 .95 --ceilings .95 1.0`.

Each pair gets its own ephemeral cosine Chroma collection, real MiniLM embeddings, one cached entry, and one lookup through the repository's `VectorCache` and `JevJudge`. The adapter in `run.py` sidesteps a current repository import error (`UniqueConstraintError` is absent from Chroma 1.5.9) without changing production code. This script does **not** repair that compatibility issue. Adaptive thresholds and cacheability checks are off. Timing excludes model loading and seed indexing; it includes lookup-time embedding inference, Chroma retrieval and any judge call. Results vary with unpinned embedding and `jev-latest` revisions.

## Observed live run, 27 September 2026

At floor 0.85 the actual earlier run (including embedding inference in lookup timing) gave:

| Policy | TP | TN | FP (wrong hits) | FN | Correct | Hit precision | Hit recall | Jev calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cosine only | 9 | 2 | 9 | 3 | 11/23 | 50% | 75% | 0 |
| Jev, ceiling .95 | 9 | 10 | 1 | 3 | 19/23 | 90% | 75% | 13 |
| Jev, ceiling 1.0 | 9 | 11 | 0 | 3 | 20/23 | 100% | 75% | 16 |

Thus the 1.0 ceiling prevented all nine wrong hits **on this set** while retaining all nine true hits. Correct tagging rose from 48% to 87%. The remaining three false misses scored below the floor. With the default .95 ceiling, case 19 (`gluten-free` versus `contain gluten`) scored .957 and bypassed Jev, serving the wrong answer; at ceiling 1.0 Jev rejected it. This is a design issue to fix before treating the judge as a safety gate.

Historical p50/p95 full lookup was 167/210 ms at floor .85, ceiling .95; judged lookups 187/219 ms. At ceiling 1.0 full lookup p50/p95 was 164/229 ms. These sequential one-off measurements are not production SLOs. No API errors were observed in the run, but the library's default one-second timeout was not tested. Price/token spend was not measured. See the full report for the threshold sweep and case-level cosine scores.

## Production POC decision

**Suitable for an instrumented, non-customer-impacting POC after the .95 bypass is disabled or fixed, not yet for serving live customer answers as a correctness gate.** First repair Chroma-version compatibility and pin dependencies/models, add explicit tenant/context boundaries and real stored answers, test multi-candidate retrieval and actual workload labels, and test failure/latency at the intended timeout. The single-entry, 23-pair synthetic run cannot establish safe production thresholds, throughput, spend, or multi-tenant isolation.
