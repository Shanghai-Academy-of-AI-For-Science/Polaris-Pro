"""SMILES whole-atom tokenizer for the mol modality AR decoder.

Each chemical "atomic unit" (organic-subset atom, multi-char element,
bracketed atom, bond, branch, ring closure, stereo/charge marker) is one
token. Bracketed atoms (e.g. ``[C@@H]``, ``[Fe+3]``) are expanded into a
``[`` / element / modifiers / ``]`` token sequence mirroring SMILES grammar.
Token IDs 0..2 are ``<pad>``/``<cls>``/``<sep>``; 3..N are chemical tokens.
Order is fixed for stable IDs — checkpoints must use exactly ``MOL_VOCAB_SIZE``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# Vocab construction — order matters for stable token IDs.
_SPECIAL_TOKENS = ["<pad>", "<cls>", "<sep>"]

# Single-letter "organic subset" atoms (used outside brackets), listed first.
_ORGANIC_UPPER = ["B", "C", "N", "O", "P", "S", "F", "I"]

# Aromatic subset (lowercase) — only meaningful in aromatic rings.
# Single-letter aromatic atoms used outside brackets, plus the multi-char
# aromatic forms that SMILES requires inside brackets (``[se]`` aromatic
# selenium, ``[te]`` tellurium, ``[as]`` arsenic, ``[si]`` silicon — these
# show up in published medicinal-chemistry datasets like ChEMBL/PubChem).
_AROMATIC_LOWER = ["b", "c", "n", "o", "p", "s"]
_AROMATIC_LOWER_2 = ["se", "te", "as", "si"]

# All chemical-element symbols for Z ∈ [1, 99].  TWO-LETTER symbols MUST
# come before SINGLE-LETTER ones in the regex alternation for greedy
# matching to work; here we put two-letter first, single-letter second.
# (Single-letter symbols already in _ORGANIC_UPPER will share their
# existing token ID via the dedup step below.)
_ELEMENTS_TWO_LETTER = [
    # Z=2..18
    "He",
    "Li", "Be",                       "Ne",
    "Na", "Mg", "Al", "Si",           "Cl", "Ar",
    # Z=19..36
    "Ca", "Sc", "Ti",       "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    # Z=37..54
    "Rb", "Sr",       "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te",       "Xe",
    # Z=55..86 (Cs..Rn)
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta",       "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    # Z=87..99 (Fr..Es)
    "Fr", "Ra",
    "Ac", "Th", "Pa",       "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
]

# Single-letter element symbols (Z ∈ {1, 5, 6, 7, 8, 9, 14, 15, 16, 17,
# 19, 23, 39, 53, 74, 92, 53...}).  We list them AFTER the two-letter
# alternation so the regex prefers ``Cl`` over ``C`` when both could
# match.  The dedup pass below collapses single-letter element symbols
# that already exist in the organic-subset list (B/C/N/O/P/S/F/I) onto
# the same token IDs, so generated SMILES keep a single canonical ID
# per element symbol.
_ELEMENTS_ONE_LETTER = [
    "H",                                                  # Z=1
    "B", "C", "N", "O", "F",                              # Z=5,6,7,8,9 (organic dups)
    "P", "S",                                             # Z=15,16 (organic dups)
    "K",                                                  # Z=19
    "V",                                                  # Z=23
    "Y",                                                  # Z=39
    "I",                                                  # Z=53 (organic dup)
    "W",                                                  # Z=74
    "U",                                                  # Z=92
]

# Structural / bond / charge / stereo tokens (single character or fixed
# multi-char like ``@@``).  Order doesn't matter much, but we keep
# similar tokens grouped for readability.
_STRUCTURAL = [
    "[", "]",
    "(", ")",
    "-", "=", "#", ":", "/", "\\", ".",     # bonds + disconnect
    "+",                                     # charge sign
    "@", "@@",                               # chirality
]

# Single-digit ring closures — these can also serve as charge magnitudes
# inside brackets (e.g. ``[Fe+3]``).  Same token IDs are used in both
# contexts; the regex distinguishes by the surrounding context (inside
# brackets vs. inline).
_DIGITS = [str(d) for d in range(10)]

# %NN ring closures.  Two-digit form is required by SMILES when a ring
# closure index ≥ 10 is needed.
_PERCENT_RING = [f"%{n:02d}" for n in range(10, 100)]


def _build_vocab() -> Tuple[Dict[str, int], List[str]]:
    """Build ``token → id`` map and the inverse ``id → token`` list.

    Performs first-occurrence dedup: if the same string appears in
    multiple lists (e.g. ``"C"`` is in both ``_ORGANIC_UPPER`` and
    ``_ELEMENTS_ONE_LETTER``), only the FIRST occurrence reserves a new
    ID; subsequent occurrences are dropped from the table.
    """
    seen: Dict[str, int] = {}
    inverse: List[str] = []

    def _add_all(tokens):
        for tok in tokens:
            if tok not in seen:
                seen[tok] = len(inverse)
                inverse.append(tok)

    _add_all(_SPECIAL_TOKENS)
    _add_all(_ORGANIC_UPPER)
    _add_all(_AROMATIC_LOWER)
    _add_all(_AROMATIC_LOWER_2)
    _add_all(_ELEMENTS_TWO_LETTER)
    _add_all(_ELEMENTS_ONE_LETTER)
    _add_all(_STRUCTURAL)
    _add_all(_DIGITS)
    _add_all(_PERCENT_RING)
    return seen, inverse


MOL_TOKEN_TO_ID, MOL_ID_TO_TOKEN = _build_vocab()
MOL_VOCAB_SIZE = len(MOL_ID_TO_TOKEN)

# Reserved special tokens (well-known IDs that callers can use as
# sentinels without rebuilding the dict).
MOL_PAD_ID = MOL_TOKEN_TO_ID["<pad>"]   # 0
MOL_CLS_ID = MOL_TOKEN_TO_ID["<cls>"]   # 1
MOL_SEP_ID = MOL_TOKEN_TO_ID["<sep>"]   # 2


# ---------------------------------------------------------------------------
# Regex tokenizer
# ---------------------------------------------------------------------------

# Build the SMILES regex.  Order of alternation = priority order:
#   1. Bracketed atoms — opaque whole; we descend into them programmatically.
#   2. Two-letter element symbols (greedy first so ``Cl`` beats ``C``).
#   3. ``%NN`` ring closures.
#   4. Single-letter element symbols (organic + rare metals).
#   5. Aromatic single-letter symbols.
#   6. Structural / bond / digit characters.
_TWO_CHAR_ALT = "|".join(re.escape(e) for e in _ELEMENTS_TWO_LETTER)
_ONE_CHAR_ATOM = "[A-Z]"      # any uppercase ASCII letter
_AROMATIC_AT = "[" + "".join(_AROMATIC_LOWER) + "]"
_BRACKET = r"\[[^\]]+\]"

SMILES_REGEX = re.compile(
    _BRACKET                              # 1) [...] opaque
    + r"|" + _TWO_CHAR_ALT                # 2) two-letter elements
    + r"|" + r"%\d{2}"                    # 3) %NN ring closures
    + r"|" + _ONE_CHAR_ATOM               # 4) one-letter atom (covers organic + rare)
    + r"|" + _AROMATIC_AT                 # 5) aromatic lowercase
    + r"|" + r"[\d\(\)\-=#:/\\.+@]"       # 6) structural single-char
)

# Bracketed-atom inner regex.  Matches the element first (one or two
# letters), then optional explicit-H count, optional stereo, optional
# charge, in the order SMILES grammar requires.  Used by
# ``_tokenize_bracket`` to break a [...] expression into our token
# stream.
_AROMATIC_LOWER_2_ALT = "|".join(re.escape(e) for e in _AROMATIC_LOWER_2)
_BRACKET_INNER_RE = re.compile(
    r"^"
    r"(?P<element>"
    + _AROMATIC_LOWER_2_ALT + r"|"   # aromatic two-letter (se, te, as, si)
    + _TWO_CHAR_ALT + r"|"            # uppercase two-letter (Cl, Br, Mn, Fe...)
    + r"[A-Z][a-z]?|"                 # generic two-letter (defensive; rare)
    + _AROMATIC_AT                    # aromatic single-letter (b/c/n/o/p/s)
    + r")"
    r"(?P<chiral>@@?)?"
    r"(?P<hcount>H\d*)?"
    r"(?P<charge>[+\-]\d*)?"
    r"$"
)


class MolSmilesTokenizer:
    """Whole-atom SMILES tokenizer.

    Public API:
        encode(smiles)            → List[int]    (mol-vocab ids)
        decode(ids)               → str           (canonical SMILES form;
                                                    may differ from input
                                                    in trivial whitespace)

    Raises ``ValueError`` if the regex consumes < len(smiles) characters
    (i.e. an unrecognised char was hit).  This is intentionally strict —
    silently dropping unrecognised chars would corrupt training data.
    """

    def __init__(self):
        self.token_to_id = MOL_TOKEN_TO_ID
        self.id_to_token = MOL_ID_TO_TOKEN

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, smiles: str) -> List[int]:
        """Tokenize a SMILES string into a list of integer IDs."""
        if not smiles:
            return []
        ids: List[int] = []
        cursor = 0
        for m in SMILES_REGEX.finditer(smiles):
            if m.start() != cursor:
                bad = smiles[cursor:m.start()]
                raise ValueError(
                    f"Invalid SMILES: unrecognized substring {bad!r} at "
                    f"position {cursor} in {smiles!r}"
                )
            tok = m.group()
            if tok.startswith("[") and tok.endswith("]"):
                ids.extend(self._encode_bracket(tok, smiles))
            else:
                tid = self.token_to_id.get(tok)
                if tid is None:
                    raise ValueError(
                        f"Invalid SMILES: unknown token {tok!r} in {smiles!r}"
                    )
                ids.append(tid)
            cursor = m.end()
        if cursor != len(smiles):
            bad = smiles[cursor:]
            raise ValueError(
                f"Invalid SMILES: trailing unrecognized substring {bad!r} "
                f"in {smiles!r}"
            )
        return ids

    def _encode_bracket(self, bracketed: str, full_smiles: str) -> List[int]:
        """Tokenize a bracketed atom expression into our token stream.

        Format: ``[`` element [chirality] [hcount] [charge] ``]``.
        Each piece becomes a separate token.
        """
        inner = bracketed[1:-1]
        m = _BRACKET_INNER_RE.match(inner)
        if not m:
            raise ValueError(
                f"Invalid bracketed atom {bracketed!r} in {full_smiles!r}"
            )
        out: List[int] = [self.token_to_id["["]]

        element = m.group("element")
        # Element symbol may not be in our vocab if the SMILES contains a
        # transactinide / Z>=100; encoder should already have rejected
        # those samples, but we still need a graceful tokenizer error.
        elem_id = self.token_to_id.get(element)
        if elem_id is None:
            raise ValueError(
                f"Element {element!r} not in mol vocab (bracketed atom "
                f"{bracketed!r} in {full_smiles!r})"
            )
        out.append(elem_id)

        chiral = m.group("chiral")
        if chiral:
            out.append(self.token_to_id[chiral])

        hcount = m.group("hcount")
        if hcount:
            out.append(self.token_to_id["H"])
            digits = hcount[1:]  # strip the 'H'
            for d in digits:
                out.append(self.token_to_id[d])

        charge = m.group("charge")
        if charge:
            sign = charge[0]
            mag = charge[1:]
            out.append(self.token_to_id[sign])
            for d in mag:
                out.append(self.token_to_id[d])

        out.append(self.token_to_id["]"])
        return out

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int]) -> str:
        """Convert an ID list back to a SMILES string by simple concat.

        Preserves the order produced by ``encode`` so a round-trip
        ``decode(encode(s)) == s`` for all SMILES strings the tokenizer
        accepts.
        """
        return "".join(self.id_to_token[i] for i in ids if 0 <= i < len(self.id_to_token))


__all__ = [
    "MOL_TOKEN_TO_ID",
    "MOL_ID_TO_TOKEN",
    "MOL_VOCAB_SIZE",
    "MOL_PAD_ID",
    "MOL_CLS_ID",
    "MOL_SEP_ID",
    "MolSmilesTokenizer",
    "SMILES_REGEX",
]
