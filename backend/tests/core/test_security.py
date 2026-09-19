from app.core.security import API_KEY_PREFIX, generate_api_key, hash_api_key, verify_api_key


def test_generate_api_key_has_expected_prefix() -> None:
    raw_key = generate_api_key()
    assert raw_key.startswith(API_KEY_PREFIX)
    assert raw_key.startswith("rv_live_")


def test_generate_api_key_is_unique_each_call() -> None:
    keys = {generate_api_key() for _ in range(50)}
    assert len(keys) == 50


def test_hash_api_key_is_deterministic() -> None:
    raw_key = generate_api_key()
    assert hash_api_key(raw_key) == hash_api_key(raw_key)


def test_hash_api_key_differs_for_different_keys() -> None:
    key_a = generate_api_key()
    key_b = generate_api_key()
    assert hash_api_key(key_a) != hash_api_key(key_b)


def test_hash_api_key_never_equals_raw_key() -> None:
    raw_key = generate_api_key()
    assert hash_api_key(raw_key) != raw_key


def test_verify_api_key_true_for_matching_pair() -> None:
    raw_key = generate_api_key()
    stored_hash = hash_api_key(raw_key)
    assert verify_api_key(raw_key, stored_hash) is True


def test_verify_api_key_false_for_wrong_key() -> None:
    stored_hash = hash_api_key(generate_api_key())
    assert verify_api_key(generate_api_key(), stored_hash) is False
