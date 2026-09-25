from my_predictor.embedding import window_starts


def test_window_starts_match_current_handoff_example() -> None:
    assert window_starts(3423, 1022, 128) == [0, 894, 1788, 2401]


def test_window_starts_cover_short_and_exact_sequences() -> None:
    assert window_starts(100, 1022, 128) == [0]
    assert window_starts(1022, 1022, 128) == [0]


def test_window_starts_reject_invalid_overlap() -> None:
    try:
        window_starts(100, 20, 20)
    except ValueError:
        pass
    else:
        raise AssertionError("overlap equal to window_size must be rejected")
