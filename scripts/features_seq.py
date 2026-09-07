"""features_seq.py: sequence-derived features, as pure vectorised functions.

Every feature here is computed from the amino-acid sequence alone, with no network call
and no model. They are the cheap half of the feature table; the ESM-2 disorder and
embedding features (spec section 5.4) run separately on ZeroGPU and cache by seq_md5.

Nothing here goes into a prompt as a raw sequence. Llama's tokeniser shreds amino-acid
strings at roughly one token per two or three residues with no semantic structure, so a
400-residue protein would burn about 150 meaningless tokens. These numbers go in instead.

Scales and constants are named and sourced rather than inlined as magic numbers, because
a silent sign error in a hydropathy scale produces plausible-looking output for months.
"""
from __future__ import annotations

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# Kyte & Doolittle (1982), J Mol Biol 157:105. Positive is hydrophobic.
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
KD_MIN, KD_MAX = -4.5, 4.5

# EMBOSS pKa set, as used by iep. Documented because pI shifts by ~0.3 units between
# the common sets and a comparison against another tool will otherwise look broken.
PKA_SIDE = {"C": 8.5, "D": 3.9, "E": 4.1, "H": 6.5, "K": 10.8, "R": 12.5, "Y": 10.1}
PKA_NTERM, PKA_CTERM = 8.6, 3.6
POSITIVE = ("H", "K", "R")
NEGATIVE = ("C", "D", "E", "Y")

AROMATIC = ("F", "W", "Y")
# FoldIndex, Prilusky et al. (2005) Bioinformatics 21:3435.
FOLDINDEX_H, FOLDINDEX_C = 2.785, 1.151

TM_WINDOW = 19          # a membrane-spanning helix is about 20 residues
TM_KD_THRESHOLD = 1.6   # mean KD over the window
SIGNAL_WINDOW = 10
SIGNAL_SEARCH_END = 30
LOWCOMP_WINDOW = 20
LOWCOMP_ENTROPY_BITS = 2.9   # SEG-like: a 20-residue window of uniform composition is 4.3


def encode(seq: str) -> np.ndarray:
    """Sequence to integer codes; anything outside the standard 20 becomes -1."""
    return np.fromiter((AA_INDEX.get(c, -1) for c in seq), dtype=np.int8, count=len(seq))


def counts_matrix(seqs: list[str]) -> np.ndarray:
    """(n_seqs, 20) counts of the standard amino acids."""
    out = np.zeros((len(seqs), 20), dtype=np.int32)
    for i, s in enumerate(seqs):
        codes = encode(s)
        codes = codes[codes >= 0]
        if codes.size:
            np.add.at(out[i], codes, 1)
    return out


def _kd_vector() -> np.ndarray:
    return np.array([KD[a] for a in AA], dtype=np.float64)


def gravy(counts: np.ndarray) -> np.ndarray:
    """Grand average of hydropathy: the mean KD value over the sequence."""
    n = counts.sum(axis=1)
    return np.where(n > 0, counts @ _kd_vector() / np.maximum(n, 1), np.nan)


def aromatic_fraction(counts: np.ndarray) -> np.ndarray:
    idx = [AA_INDEX[a] for a in AROMATIC]
    n = counts.sum(axis=1)
    return np.where(n > 0, counts[:, idx].sum(axis=1) / np.maximum(n, 1), np.nan)


def residue_fraction(counts: np.ndarray, aa: str) -> np.ndarray:
    n = counts.sum(axis=1)
    return np.where(n > 0, counts[:, AA_INDEX[aa]] / np.maximum(n, 1), np.nan)


def net_charge(counts: np.ndarray, ph: float = 7.0) -> np.ndarray:
    """Net charge at a given pH from the side chains plus both termini."""
    q = np.zeros(counts.shape[0], dtype=np.float64)
    has = counts.sum(axis=1) > 0
    for aa in POSITIVE:
        q += counts[:, AA_INDEX[aa]] / (1.0 + 10.0 ** (ph - PKA_SIDE[aa]))
    for aa in NEGATIVE:
        q -= counts[:, AA_INDEX[aa]] / (1.0 + 10.0 ** (PKA_SIDE[aa] - ph))
    q += has / (1.0 + 10.0 ** (ph - PKA_NTERM))
    q -= has / (1.0 + 10.0 ** (PKA_CTERM - ph))
    return np.where(has, q, np.nan)


def isoelectric_point(counts: np.ndarray, lo: float = 0.0, hi: float = 14.0,
                      iterations: int = 60) -> np.ndarray:
    """pI by vectorised bisection over the whole set at once.

    Charge is monotonically decreasing in pH, so bisection is safe and 60 halvings of a
    14-unit interval converge far below the precision the value deserves.
    """
    n = counts.shape[0]
    lo_a = np.full(n, lo)
    hi_a = np.full(n, hi)
    for _ in range(iterations):
        mid = (lo_a + hi_a) / 2.0
        q = net_charge_at(counts, mid)
        # where charge is still positive the pI lies above mid
        hi_a = np.where(q > 0, hi_a, mid)
        lo_a = np.where(q > 0, mid, lo_a)
    pi = (lo_a + hi_a) / 2.0
    return np.where(counts.sum(axis=1) > 0, pi, np.nan)


def net_charge_at(counts: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """Vector version of net_charge with a per-sequence pH array."""
    q = np.zeros(counts.shape[0], dtype=np.float64)
    has = counts.sum(axis=1) > 0
    for aa in POSITIVE:
        q += counts[:, AA_INDEX[aa]] / (1.0 + 10.0 ** (ph - PKA_SIDE[aa]))
    for aa in NEGATIVE:
        q -= counts[:, AA_INDEX[aa]] / (1.0 + 10.0 ** (PKA_SIDE[aa] - ph))
    q += has / (1.0 + 10.0 ** (ph - PKA_NTERM))
    q -= has / (1.0 + 10.0 ** (PKA_CTERM - ph))
    return q


# --------------------------------------------------------------------------- per-sequence

def kd_profile(seq: str) -> np.ndarray:
    """Per-residue KD value; unknown residues score 0."""
    return np.array([KD.get(c, 0.0) for c in seq], dtype=np.float64)


def _windowed_mean(x: np.ndarray, w: int) -> np.ndarray:
    if x.size < w or w <= 0:
        return np.empty(0)
    c = np.concatenate(([0.0], np.cumsum(x)))
    return (c[w:] - c[:-w]) / w


def tm_helices(seq: str, window: int = TM_WINDOW,
               threshold: float = TM_KD_THRESHOLD) -> int:
    """Count of predicted membrane-spanning segments.

    A hydrophobic window heuristic, not TMHMM: it counts maximal runs of positions whose
    KD mean over `window` residues exceeds `threshold`. Good enough to separate soluble
    from polytopic, which is the distinction the model needs, and honest about being an
    estimate.
    """
    m = _windowed_mean(kd_profile(seq), window)
    if m.size == 0:
        return 0
    hot = m > threshold
    # count runs of True
    return int(np.sum(hot[1:] & ~hot[:-1]) + (1 if hot[0] else 0))


def has_signal_peptide(seq: str, window: int = SIGNAL_WINDOW,
                       search_end: int = SIGNAL_SEARCH_END,
                       threshold: float = TM_KD_THRESHOLD) -> bool:
    """A hydrophobic h-region inside the N-terminal 30 residues.

    Deliberately crude, and it cannot tell a signal peptide from an N-terminal
    transmembrane helix, which is a known confusion in every method that does this.
    """
    head = seq[:search_end]
    m = _windowed_mean(kd_profile(head), window)
    return bool(m.size and m.max() > threshold)


def foldindex_profile(seq: str, window: int = 51) -> np.ndarray:
    """FoldIndex per position: negative means predicted disordered.

    I = 2.785 * <H> - |<R>| - 1.151, with <H> the KD hydropathy rescaled to [0, 1].
    """
    kd = kd_profile(seq)
    h = (kd - KD_MIN) / (KD_MAX - KD_MIN)
    charge = np.array([1.0 if c in ("K", "R") else -1.0 if c in ("D", "E") else 0.0
                       for c in seq], dtype=np.float64)
    w = min(window, len(seq)) if len(seq) else 0
    if w == 0:
        return np.empty(0)
    return FOLDINDEX_H * _windowed_mean(h, w) - np.abs(_windowed_mean(charge, w)) - FOLDINDEX_C


def disorder_fractions(seq: str, window: int = 51) -> tuple[float, float, float, float]:
    """(overall, N-terminal, C-terminal, internal) predicted-disordered fractions.

    N- and C-terminal are the first and last 30 residues of the profile, matching the
    spec's split, because terminal disorder is what construct trimming actually targets.
    """
    p = foldindex_profile(seq, window)
    if p.size == 0:
        return (np.nan,) * 4
    dis = p < 0
    edge = min(30, p.size)
    nterm = float(dis[:edge].mean())
    cterm = float(dis[-edge:].mean())
    inner = dis[edge:-edge] if p.size > 2 * edge else np.empty(0)
    internal = float(inner.mean()) if inner.size else 0.0
    return float(dis.mean()), nterm, cterm, internal


def low_complexity_fraction(seq: str, window: int = LOWCOMP_WINDOW,
                            bits: float = LOWCOMP_ENTROPY_BITS) -> float:
    """Fraction of positions inside a low-entropy window (SEG-like).

    A uniform 20-residue window scores about 4.3 bits; poly-Q or poly-A collapses towards
    0. The threshold marks the windows a crystallographer would call disqualifying.
    """
    n = len(seq)
    if n < window:
        return 0.0
    codes = encode(seq)
    low = np.zeros(n - window + 1, dtype=bool)
    for i in range(n - window + 1):
        w = codes[i:i + window]
        w = w[w >= 0]
        if w.size == 0:
            continue
        _, c = np.unique(w, return_counts=True)
        p = c / w.size
        low[i] = float(-(p * np.log2(p)).sum()) < bits
    return float(low.mean())
