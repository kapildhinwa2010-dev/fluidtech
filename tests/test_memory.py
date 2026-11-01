"""The frozen register: exactness, append-only, fail-loud, constant size."""
import numpy as np

from superfluid import FrozenRegister, SLOT_BYTES, budget_split, ctx_key


def test_the_slot_is_20_bytes():
    assert SLOT_BYTES == 20


def test_a_recalled_value_is_bit_exact():
    reg = FrozenRegister(capacity=16)
    key = ctx_key([1, 2, 3])
    reg.promote(123456, key, sig=0, anchor=0)
    slot, value = reg.recall(key)
    assert value == 123456
    assert isinstance(value, int)
    assert reg.val[slot] == 123456          # stored as an int32, not a float


def test_a_signature_mismatch_fails_loud():
    reg = FrozenRegister(capacity=16)
    key = ctx_key([1, 2, 3])
    reg.promote(99, key, sig=0xDEADBEEF, anchor=0)
    assert reg.recall(key, sig=0xDEADBEEF)[1] == 99
    assert reg.recall(key, sig=0x12345678) is None    # fails, never lies


def test_the_register_is_append_only():
    reg = FrozenRegister(capacity=16)
    key = ctx_key([4, 5])
    reg.promote(1, key, sig=0, anchor=0)
    reg.promote(2, key, sig=0, anchor=0)
    assert reg.n == 2                          # two slots, not one overwritten
    assert reg.recall(key)[1] == 2             # the latest wins


def test_overflow_is_refused():
    reg = FrozenRegister(capacity=2)
    reg.promote(1, ctx_key([1]), sig=0, anchor=0)
    reg.promote(2, ctx_key([2]), sig=0, anchor=0)
    try:
        reg.promote(3, ctx_key([3]), sig=0, anchor=0)
    except MemoryError:
        pass
    else:
        raise AssertionError("the register must REFUSE to overflow")


def test_the_budget_law_never_over_commits():
    b = budget_split(16 * 1024, M=1024, r=8, slot_bytes=SLOT_BYTES)
    assert b["fluid_bytes"] + b["register_bytes"] <= 16 * 1024
    b2 = budget_split(16 * 1024, M=100000, r=8, slot_bytes=SLOT_BYTES)
    assert b2["M_reduced"]
    assert b2["fluid_bytes"] + b2["register_bytes"] <= 16 * 1024


def test_a_round_trip_through_bytes():
    reg = FrozenRegister(capacity=8)
    key = ctx_key([9, 9, 9])
    reg.promote(4242, key, sig=7, anchor=3)
    blob = reg.to_bytes()
    reg2 = FrozenRegister.from_bytes(blob, capacity=8)
    assert reg2.recall(key, sig=7)[1] == 4242
