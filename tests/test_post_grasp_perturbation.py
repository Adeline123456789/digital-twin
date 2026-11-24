import numpy as np

from scripts.audit_post_grasp_observability import area_under_roc, cardinal_direction, evaluate_xy


def test_auc_counts_ties_and_separates_scores():
    assert area_under_roc([2., 3.], [0., 1.]) == 1.
    assert area_under_roc([1.], [1.]) == .5


def test_episode_level_xy_audit_recovers_heldout_cardinal_directions():
    train_targets = np.array([[.1, 0.], [-.2, 0.], [0., .1], [0., -.1]])
    heldout_targets = train_targets * .8
    train_features = np.c_[train_targets, train_targets ** 2]
    heldout_features = np.c_[heldout_targets, heldout_targets ** 2]
    nominal_train = np.zeros_like(train_features)
    nominal_test = np.zeros_like(heldout_features)
    directions = ['positive_x', 'negative_x', 'positive_y', 'negative_y']
    metrics = evaluate_xy(train_features, nominal_train, train_targets,
                          heldout_features, nominal_test, heldout_targets,
                          directions, directions)

    assert metrics['detection_auroc'] == 1.
    assert metrics['direction_accuracy'] == 1.
    assert metrics['current_direction_macro_recall'] == 1.


def test_direction_label_describes_current_error_not_treatment():
    assert cardinal_direction([-.01, .2]) == 'positive_y'
