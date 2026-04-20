import torch
from model import PrunableLinear

def sparsity_loss(model):
    loss = 0
    # Iterate through all modules in the model
    for m in model.modules():
        if isinstance(m, PrunableLinear):  # Only consider prunable layers
            loss += torch.sigmoid(m.gate_scores).sum()  # L1 penalty on gate values
    return loss  # Total sparsity loss across all layers


def calculate_sparsity(model, threshold=1e-2):
    total, pruned = 0, 0

    # Iterate through all modules to compute sparsity
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            g = torch.sigmoid(m.gate_scores)  # Convert gate scores to [0,1]
            total += g.numel()  # Total number of gates
            pruned += (g < threshold).sum().item()  # Count gates close to zero

    return 100 * pruned / total  # Return sparsity percentage