import numpy as np

from app import (
    ecc_majority,
    ecc_repeat,
    pack_message_bits,
    prng_order,
    psnr,
    unpack_message_bits,
)


def test_message_round_trip():
    header, payload = pack_message_bits("portfolio test", "demo-key")
    total = np.concatenate([header, payload])
    ok, message = unpack_message_bits(total, "demo-key")
    assert ok is True
    assert message == "portfolio test"


def test_wrong_key_is_rejected():
    header, payload = pack_message_bits("secret", "correct-key")
    total = np.concatenate([header, payload])
    ok, _ = unpack_message_bits(total, "wrong-key")
    assert ok is False


def test_ecc_majority_recovers_single_error_per_triplet():
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    encoded = ecc_repeat(bits, 3)
    corrupted = encoded.copy()
    corrupted[0] ^= 1
    corrupted[3] ^= 1
    decoded = ecc_majority(corrupted, 3)
    assert np.array_equal(decoded, bits)


def test_prng_order_is_key_deterministic():
    a = prng_order(128, "same-key")
    b = prng_order(128, "same-key")
    c = prng_order(128, "different-key")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_identical_images_have_high_psnr():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert psnr(image, image) == 99.0
