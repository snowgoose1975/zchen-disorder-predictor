import torch

from my_predictor.model import ResidueBiGRU, ResidueConvBiGRU, ResidueTCN


def test_bigru_packed_sequence_is_padding_invariant() -> None:
    torch.manual_seed(20260925)
    model = ResidueBiGRU(input_dim=3, hidden_dim=4, dropout=0.0)
    model.eval()

    short = torch.randn(1, 4, 3)
    padded_short = torch.nn.functional.pad(short, (0, 0, 0, 3))
    second = torch.randn(1, 7, 3)

    with torch.inference_mode():
        single = model(short, torch.tensor([4]))
        batched = model(
            torch.cat([padded_short, second], dim=0),
            torch.tensor([4, 7]),
        )

    assert torch.allclose(single[0, :4], batched[0, :4], atol=1e-6)

def test_conv_bigru_packed_sequence_is_padding_invariant() -> None:
    torch.manual_seed(20260925)
    model = ResidueConvBiGRU(input_dim=3, hidden_dim=4, dropout=0.0)
    model.eval()

    short = torch.randn(1, 4, 3)
    padded_short = torch.nn.functional.pad(short, (0, 0, 0, 3))
    second = torch.randn(1, 7, 3)

    with torch.inference_mode():
        single = model(short, torch.tensor([4]))
        batched = model(
            torch.cat([padded_short, second], dim=0),
            torch.tensor([4, 7]),
        )

    assert single.shape == (1, 4)
    assert torch.allclose(single[0, :4], batched[0, :4], atol=1e-6)
def test_tcn_is_padding_invariant() -> None:
    torch.manual_seed(20260925)
    model = ResidueTCN(input_dim=3, hidden_dim=4, dropout=0.0)
    model.eval()

    short = torch.randn(1, 4, 3)
    padded_short = torch.nn.functional.pad(short, (0, 0, 0, 3))
    second = torch.randn(1, 7, 3)

    with torch.inference_mode():
        single = model(short, torch.tensor([4]))
        batched = model(
            torch.cat([padded_short, second], dim=0),
            torch.tensor([4, 7]),
        )

    assert single.shape == (1, 4)
    assert torch.allclose(single[0, :4], batched[0, :4], atol=1e-6)
