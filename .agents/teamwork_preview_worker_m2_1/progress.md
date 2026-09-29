# Progress — Milestone M2: 3-Tier Baseline Hierarchy

Last visited: 2026-09-22T23:11:30Z

## Status
- [x] Initialized DISPATCH.md and BRIEFING.md
- [x] Read ORIGINAL_REQUEST.md, PROJECT.md, Explorer reports, and inspect existing codebase
- [x] Fix M1 issues in `fed_lunar/models/negative_gen.py` (line 281 fallback noise 1-to-1 correspondence)
- [x] Fix M1 issues in `fed_lunar/federated/strategy.py` (unit-norm gradient scaling in `dr_cagrad` and `dr_pcgrad`)
- [x] Implement F5: `fed_lunar/baselines/naive_lunar.py` (`NaiveFedLunar`)
- [x] Implement F6: `fed_lunar/baselines/fed_ae.py` (`FedAutoEncoder`)
- [x] Implement F7: `fed_lunar/baselines/fedprox_lunar.py` (`FedProxLunar`, `PCGradFedLunar`)
- [x] Implement F8: `fed_lunar/baselines/loc_nfst_bound.py` (`LOC_NFST_Bound`)
- [x] Create `fed_lunar/baselines/__init__.py`
- [x] Implement `tests/test_baselines.py`
- [x] Run full test suite (`pytest tests/test_baselines.py tests/test_lunar_model.py tests/test_cmnp_purging.py tests/test_droga_alignment.py -v`) — 24 passed in 14.95s (100% success)
- [ ] Write `handoff.md` and notify parent
