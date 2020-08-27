import torch


def pairwise_distance(a, squared=False, eps=1e-16):
    """Computes the pairwise distance matrix with numerical stability."""
    operand = a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1)
    operandT = torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    pairwise_distances_squared = torch.add(operand,operandT) - 2 * (torch.mm(a, torch.t(a)))
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * eps)

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)
    return pairwise_distances