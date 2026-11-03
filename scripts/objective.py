"""Python reference implementation of the Step 2 RL objective (blueprint 5.1).

Lets us score any SMILES set offline, without launching REINVENT. The transforms
and the aggregator are ported from REINVENT4 v4.5.11:
  reinvent/scoring/transforms/sigmoid_functions.py
  reinvent/scoring/transforms/sigmoids.py, steps.py
  reinvent/scoring/aggregators/means.py
  reinvent/scoring/scorer.py:159-178

    total = geometric_mean([(COOH, 1.0), (TPSA, 1.0), (SAScore, 0.5)]) * alert_filter

COOH is a GATE, not a penalty: REINVENT's MatchingSubstructure hardcodes
0.5 * (1.0 + match), and with that a molecule WITHOUT the carboxylate scored a flat
0.500 while molecules WITH it had a median of 0.060 - the agent's best move was to
drop the MIDAS anchor. GroupCount + right_step removes that inversion.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

__all__ = [
    "ALERT_SMARTS",
    "CARBOXYLATE_SMARTS",
    "GATE_FLOOR",
    "double_sigmoid",
    "geometric_mean",
    "reverse_sigmoid",
    "right_step",
    "score",
    "score_smiles",
    "sigmoid",
]

CARBOXYLATE_SMARTS = "[CX3](=O)[OX2H1,OX1-]"
ALERT_SMARTS = [
    "[NX3;H2][c]",  # primary aniline only - anilides/sulfonanilides/THN excluded
    "[*;r8]",
    "[*;r9]",
    "[*;r10]",
    "N=[N+]=[N-]",
    "C(=O)Cl",
    "[SH]",
    "[Nr0][Nr0]",
]

W_COOH, W_TPSA, W_SASCORE = 1.0, 1.0, 0.5
TPSA_WINDOW = (40.0, 115.0, 120.0, 20.0, 20.0)
SASCORE_WINDOW = (6.0, 8.0, 0.5)

# geometric_mean clamps a 0 component to 1e-8, so a molecule missing the carboxylate
# can reach at most this. Under DAP with sigma=128 that is ~128 nat below a perfect
# score - a gate in practice, though not literally zero. Triage re-applies a hard cut.
GATE_FLOOR = float(1e-8 ** (W_COOH / (W_COOH + W_TPSA + W_SASCORE)))

_CARBOXYLATE = Chem.MolFromSmarts(CARBOXYLATE_SMARTS)
_ALERTS = [Chem.MolFromSmarts(s) for s in ALERT_SMARTS]

_params = FilterCatalog.FilterCatalogParams()
_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS = FilterCatalog.FilterCatalog(_params)


def _stable_sigmoid(x: np.ndarray, k: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    h = k * x * np.log(10)
    positive = h >= 0
    y = np.zeros_like(x)
    y[positive] = 1.0 / (1.0 + np.exp(-h[positive]))
    y[~positive] = np.exp(h[~positive]) / (1.0 + np.exp(h[~positive]))
    return y.astype(np.float32)


def sigmoid(values, low: float, high: float, k: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    x = values - (high + low) / 2
    if high - low == 0:
        return (10.0 * k * x > 0).astype(np.float32)
    return _stable_sigmoid(x, 10.0 * k / (high - low))


def reverse_sigmoid(values, low: float, high: float, k: float) -> np.ndarray:
    return 1.0 - sigmoid(values, low, high, k)


def double_sigmoid(values, low, high, coef_div=100.0, coef_si=150.0,
                   coef_se=150.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    center = (high - low) / 2 + low
    out = np.zeros_like(x)
    left, right = x < center, x >= center
    if coef_div == 0:
        out[left] = (coef_si * (x[left] - low) > 0).astype(np.float32)
        out[right] = 1 - (coef_se * (x[right] - high) > 0).astype(np.float32)
    else:
        out[left] = _stable_sigmoid(x[left] - low, coef_si / coef_div)
        out[right] = 1 - _stable_sigmoid(x[right] - high, coef_se / coef_div)
    return out


def right_step(values, high: float) -> np.ndarray:
    return np.array([1.0 if v >= high else 0.0 for v in values], dtype=float)


def geometric_mean(pairs: list[tuple[np.ndarray, float]]) -> np.ndarray:
    scores = np.array([s for s, _ in pairs], dtype=float)
    weights = np.array([w for _, w in pairs], dtype=float)
    weights = np.broadcast_to(weights.reshape(-1, 1), scores.shape).astype(float)
    scores = np.maximum(scores, 1e-8)
    return np.prod(scores ** (weights / np.maximum(weights.sum(axis=0), 1e-8)), axis=0)


def score(mols: list[Chem.Mol]) -> dict[str, np.ndarray]:
    counts = np.array([len(m.GetSubstructMatches(_CARBOXYLATE)) for m in mols])
    cooh = right_step(counts, 1.0)
    tpsa = double_sigmoid(np.array([Descriptors.TPSA(m) for m in mols]), *TPSA_WINDOW)
    sa = reverse_sigmoid(
        np.array([sascorer.calculateScore(m) for m in mols]), *SASCORE_WINDOW
    )
    alerts = np.array([
        0.0 if (any(m.HasSubstructMatch(p) for p in _ALERTS) or _PAINS.HasMatch(m))
        else 1.0
        for m in mols
    ])
    total = geometric_mean(
        [(cooh, W_COOH), (tpsa, W_TPSA), (sa, W_SASCORE)]
    ) * alerts
    return {"total": total, "cooh": cooh, "tpsa": tpsa, "sascore": sa,
            "alerts": alerts}


def score_smiles(smiles_list: list[str]) -> dict[str, np.ndarray]:
    """Score the parseable molecules, and report WHICH ones were scored.

    Unparseable SMILES are dropped, so the returned arrays can be shorter than
    `smiles_list`. The result therefore carries a "smiles" entry listing exactly
    the SMILES that were scored, in order: a caller pairing its own labels with
    these scores must zip against that, never against its original list, or a
    single parse failure silently shifts every label after it onto the wrong score.
    """
    kept, mols = [], []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            kept.append(smiles)
            mols.append(mol)
    result = score(mols)
    result["smiles"] = kept
    return result
