import torch

from functools import partial
from torch.nn.utils.rnn import pad_sequence

nlp_pad_sequence = partial(pad_sequence, batch_first=True, padding_value=0)


def shoot_nans(x: torch.Tensor):
    nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
    n = torch.logical_not(nan_mask).long().cpu().sum().item()
    return torch.masked_fill(x, mask=nan_mask, value=0.0), n


def make_mask_2d(lengths: torch.Tensor):
    """Create binary mask from lengths indicating which indices are padding"""
    # Make sure `lengths` is a 1d array
    assert len(lengths.shape) == 1

    max_length = torch.amax(lengths, dim=0).item()
    mask = (
        torch.arange(max_length)
        .to(lengths.device)
        .expand((lengths.shape[0], max_length))
    )
    # Shape batch x timesteps
    mask = torch.ge(mask, lengths.unsqueeze(1))

    return mask


def make_mask_3d(source_lengths: torch.Tensor, target_lengths: torch.Tensor):
    """
    Make binary mask indicating which combinations of indices involve at least 1 padding element.
    Can be used to mask, for example, a batch attention matrix between 2 sequences
    """
    # Calculate binary masks for source and target
    # Then invert boolean values and convert to float (necessary for bmm later)
    source_mask = (~make_mask_2d(source_lengths)).float()
    target_mask = (~make_mask_2d(target_lengths)).float()

    # Add dummy dimensions for bmm
    source_mask = source_mask.unsqueeze(2)
    target_mask = target_mask.unsqueeze(1)

    # Calculate combinations by batch matrix multiplication
    mask = torch.bmm(source_mask, target_mask).bool()
    # Invert boolean values
    mask = torch.logical_not(mask)
    return mask


def max_pool_2d(x: torch.Tensor, lengths: torch.Tensor):
    # x: shape [batch x timesteps x features]
    mask = make_mask_2d(lengths).to(x.device).unsqueeze(-1)
    x = torch.masked_fill(x, mask=mask, value=-1e9)
    x = torch.max(x, dim=1).values
    return x


def sum_pool_2d(x: torch.Tensor, lengths: torch.Tensor):
    # x: shape [batch x timesteps x features]
    mask = make_mask_2d(lengths).to(x.device).unsqueeze(-1)
    x = torch.masked_fill(x, mask=mask, value=0.0)
    x = torch.sum(x, dim=1)
    return x
