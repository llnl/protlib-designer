from types import SimpleNamespace
import sys

import pytest
import torch

sys.modules.setdefault(
    "transformers",
    SimpleNamespace(AutoModelForMaskedLM=object, AutoTokenizer=object),
)

from protlib_designer.scorer.plm_scorer import PLMScorer
from protlib_designer.utils import validate_canonical_sequence_and_wildtypes


def test_validate_canonical_sequence_and_wildtypes_rejects_noncanonical_sequence_residue():
    with pytest.raises(ValueError, match="Sequence contains non-canonical residue") as excinfo:
        validate_canonical_sequence_and_wildtypes("XVQL", {1: "X"})

    message = str(excinfo.value)
    assert "X" in message
    assert "position 1" in message


def test_validate_canonical_sequence_and_wildtypes_rejects_noncanonical_position_wildtype():
    with pytest.raises(ValueError, match="Positions reference non-canonical wildtype residue") as excinfo:
        validate_canonical_sequence_and_wildtypes("EVQL", {1: "X"})

    message = str(excinfo.value)
    assert "X" in message
    assert "position 1" in message


def test_get_scores_surfaces_clear_error_if_noncanonical_wildtype_bypasses_preflight():
    scorer = PLMScorer.__new__(PLMScorer)
    scorer.mask = False
    scorer.score_type = "minus_llr"
    scorer.mapping = None
    scorer.aa_token_indices = [1, 2, 3]
    scorer.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 24 if token == "X" else 1)
    scorer.prepare_input = lambda sequence, positions, chain_type: (
        ["A A"],
        {1: "X"},
        [1],
        "A",
        0,
    )
    scorer.forward_pass = lambda batch, chain_token: torch.zeros((1, 2, 30))

    with pytest.raises(ValueError, match="Non-canonical residues .* cannot be scored") as excinfo:
        scorer.get_scores("EVQL", ["EA1"], "heavy")

    message = str(excinfo.value)
    assert "X" in message
    assert "position 1" in message
