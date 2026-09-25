import torch

from scripts.train import reduce_masked_loss


def test_residue_and_protein_loss_weighting_differ_as_expected() -> None:
    losses = torch.tensor([[1.0, 3.0, 99.0], [10.0, 88.0, 77.0]])
    valid = torch.tensor([[True, True, False], [True, False, False]])

    residue_loss, residue_units = reduce_masked_loss(losses, valid, "residue")
    protein_loss, protein_units = reduce_masked_loss(losses, valid, "protein")

    assert residue_units == 3
    assert protein_units == 2
    assert torch.isclose(residue_loss, torch.tensor(14.0 / 3.0))
    assert torch.isclose(protein_loss, torch.tensor(6.0))


def test_protein_loss_ignores_all_unknown_sample() -> None:
    losses = torch.tensor([[2.0, 4.0], [8.0, 9.0]])
    valid = torch.tensor([[True, False], [False, False]])
    result, units = reduce_masked_loss(losses, valid, "protein")
    assert units == 1
    assert torch.isclose(result, torch.tensor(2.0))
