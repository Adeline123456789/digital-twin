from robot_agent.interaction_events import Truth as T


def test_all_strong_kleene_truth_tables():
    values = [T.FALSE, T.UNKNOWN, T.TRUE]
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            assert a & b is values[min(i, j)]
            assert a | b is values[max(i, j)]
        assert ~a is values[2-i]
