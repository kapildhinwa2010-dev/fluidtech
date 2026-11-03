import pytest

torch = pytest.importorskip("torch")

from fluidlayer.qwen2_model import Qwen2Model, _rms_norm  # noqa: E402


def test_rms_norm_uses_last_dimension():
    x = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
    weight = torch.ones(2)

    actual = _rms_norm(x, weight, eps=0.0)
    expected = torch.tensor([[3.0 / (12.5 ** 0.5), 4.0 / (12.5 ** 0.5)],
                             [1.0, 1.0]])

    torch.testing.assert_close(actual, expected)


def test_rope_rotates_first_and_second_halves():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos = torch.zeros(1, 1, 1, 4)
    sin = torch.ones(1, 1, 1, 4)

    actual = Qwen2Model._apply_rope(x, cos, sin)

    torch.testing.assert_close(actual, torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))
