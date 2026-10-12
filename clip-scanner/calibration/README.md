# Calibration

`clip-scanner calibrate` checks whether the judges agree with reality. It runs on clips whose
performance you already know.

1. Put clips in `clips/hits/` and `clips/flops/` (or list URLs) and describe them in `manifest.toml`
   with their view counts.
2. Run:

   ```bash
   uv run clip-scanner calibrate --fetch          # --fetch downloads entries that only have a URL
   ```

   Each clip is judged as posted, as one window covering the whole file, by the **full** panel plus the devil's
   advocate. The run prints a per-clip line (✓ when the judges match the label, ✗ when they don't),
   precision, recall, the share of known flops that would have passed, and the score gap between
   your weakest hit and your strongest flop.
3. It then sweeps `overall_min`, `rubric_mean_min` and `min_survival_confidence` over the cached
   verdicts (no extra API calls) and recommends the **loosest bar that still rejects every known
   flop**. Among equally good options it picks the strictest. Apply it with:

   ```bash
   uv run clip-scanner calibrate --write-config
   ```

Results (per-clip verdicts, the full threshold grid and cost) are written to
`calibration/results/<timestamp>/calibration.json`. Media and results stay out of git.

Rules of thumb:

- Flops are the valuable examples. The tool exists to reject them.
- If no threshold rejects every flop, the judges can't tell those flops from your hits. Read the
  per-clip reasons, then edit `docs/joe-style-guide.md` (for example, add the pattern those flops share to
  "What flops") and re-run. The style guide is part of every judge's prompt.
- Re-calibrate after any change to the style guide, the model or the effort level.
