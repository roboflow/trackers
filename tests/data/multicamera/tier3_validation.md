# Tier 3 NVIDIA validation

The independent review validator executed:

```text
python scripts/verify_multicamera_eval.py --tier3 --tier3-sample-dir /tmp/aic24eval_full/MTMC_Tracking_2024/eval/sample_file --num-cores 1
```

The command exited 0 after 1,081.750 seconds against pinned evaluator revision `1eebcf0f74a510994fe4c886f4fa77fbc6724ea8`. The runner compared our result with NVIDIA for every mapped scene from `scene_061` through `scene_090`, then compared both final means with the published headline.

Final percentages:

- HOTA: 49.2826
- DetA: 49.1998
- AssA: 49.3655
- LocA: 77.0546

Per-scene numeric values were not retained and are not reconstructed here. The source validator report was `.reports/review/2026-08-11T17-59-13Z/validate-qa-tier3.md`.
