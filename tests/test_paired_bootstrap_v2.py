import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_residual_v2 import hierarchical_paired_bootstrap, paired_bootstrap


def test_bootstrap_preserves_pairs_and_recovers_known_extremes():
    same = paired_bootstrap([0, 1, 0, 1], [0, 1, 0, 1])
    assert same['ci95_pp'] == [0., 0.] and same['improvement_pp'] == 0
    improved = paired_bootstrap([0] * 32, [1] * 32)
    assert improved['ci95_pp'] == [100., 100.] and improved['recovered'] == 32
    with pytest.raises(ValueError): paired_bootstrap([0], [0, 1])


def test_hierarchical_bootstrap_resamples_training_seeds_and_conditions():
    improved = hierarchical_paired_bootstrap([0] * 4, [[1] * 4] * 5)
    assert improved['improvement_pp'] == 100
    assert improved['ci95_pp'] == [100., 100.]
    assert improved['per_training_seed_improvement_pp'] == [100.] * 5
    assert hierarchical_paired_bootstrap([0, 1], [[0, 1]] * 5) == hierarchical_paired_bootstrap([0, 1], [[0, 1]] * 5)
    with pytest.raises(ValueError):
        hierarchical_paired_bootstrap([0, 1], [[0]])
