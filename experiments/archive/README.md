# Superseded runs (kept for transparency)

`20261025-120027_E_agentmamba` and `20261025-120325_H_learned_mamba` were produced on the TEST split with a
query-parsing bug: `wo_opened_on` accepted only `asset` anchors, so work orders on ladles (IDs parse as kind
`ladle`) were answered with a clarification request. The bug was found while analysing these test errors.
It was fixed (regression test `test_work_order_on_a_ladle_is_answerable`), the controller was retrained,
development runs were moved to the VALIDATION split, and all reported test runs were regenerated afterwards.
Pre-fix E: accuracy 0.677, EM 0.625 (n=192).
