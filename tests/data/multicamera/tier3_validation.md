# Full benchmark parity

The full 30-scene sample matched NVIDIA's evaluator at revision `1eebcf0f74a510994fe4c886f4fa77fbc6724ea8`:

```bash
uv run --with pandas python scripts/verify_multicamera_eval.py \
    --tier1 --tier2 --tier3 \
    --tier3-sample-dir <sample-dir> --num-cores 1 --write-goldens
```

The Tier 3 comparison completed after 716.133 seconds. Every scene from `scene_061` through `scene_090` passed, and the final percentages matched the published sample result within the recorded tolerance:

- HOTA: 49.2825
- DetA: 49.1998
- AssA: 49.3655
- LocA: 77.0547

Exact per-scene values from both evaluators are stored as 30 deterministic JSONL records in `tier3_comparison.jsonl`. `provenance.json` records its SHA256, the accepted ours-vs-NVIDIA tolerance, full scene range and count, input hashes, evaluator identity, environment versions, and the receipt digest.
