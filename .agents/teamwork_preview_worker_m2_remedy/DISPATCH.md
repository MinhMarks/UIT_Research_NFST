# Task Dispatch: Milestone M2 Remediation Worker

## Objective
Remediate interface contract regressions and input type compatibility in `fed_lunar/baselines/` so that all baseline unit tests and the entire 126-test E2E suite pass completely.

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Detailed Requirements
1. **Target File: `fed_lunar/baselines/loc_nfst_bound.py`**:
   - Add properties:
     ```python
     @property
     def null_basis(self) -> Optional[np.ndarray]:
         return self.W

     @property
     def threshold(self) -> float:
         return self.max_train_score
     ```
   - Add method alias:
     ```python
     score = decision_function
     ```
   - Update `fit` signature to handle arbitrary kwargs and tolerance:
     ```python
     def fit(self, client_train_data, y_clusters=None, verbose=False, tol=None, rounds=None, **kwargs):
         if tol is not None:
             self.epsilon_svd = float(tol)
             self.epsilon_near_null = float(tol)
         ...
     ```
   - Low-rank orthogonal complement SVD handling:
     When $rank(P_t) < D$, compute full SVD (`full_matrices=True`) so that normal data projecting onto the unpopulated orthogonal complement $U_{:, rank\_Pt:}$ achieves exact zero residual norm ($< 10^{-29}$).

2. **Target Files: `naive_lunar.py`, `fed_ae.py`, `fedprox_lunar.py`**:
   - Update `_format_client_data` in all baseline files to accept `isinstance(client_train_data, (list, tuple))`.
   - Ensure all baseline classes define `score = decision_function` for API uniformity.

3. **Verification**:
   - Run: `pytest tests/test_baselines.py tests/test_m2_math_verification.py -v` (must pass 34/34).
   - Run: `python tests/run_e2e_tests.py` (must pass 126/126 with exit code 0).
   - Document all test outcomes and commands in `handoff.md`.
