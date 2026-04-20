import torch
from model import PrunableNet
from data import get_data_loaders
from train import train
from eval import test
from visualize import plot_tradeoff, plot_gate_distribution
from loss import calculate_sparsity  # needed for final sparsity calculation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # select GPU if available

trainloader, testloader = get_data_loaders()

lambda_values = [1e-5, 1e-4, 1e-3]  # different sparsity strengths
results = []
epochs = 50

for lam in lambda_values:

    print(f"\n--- Training with Lambda = {lam} ---")

    model = PrunableNet().to(device)

    # separate gate parameters from normal weights for different optimization behavior
    gate_params = [m.gate_scores for m in model.modules() if hasattr(m, "gate_scores")]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]

    # use different learning rates for gates vs weights
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": 1e-3, "weight_decay": 1e-4},
        {"params": gate_params, "lr": 5e-3, "weight_decay": 0.0}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # smooth LR decay

    best_acc = 0
    best_model = None

    for epoch in range(epochs):
        acc = train(model, trainloader, optimizer, scheduler, epoch, epochs, lam, device)

        # track best-performing model during training
        if acc > best_acc:
            best_acc = acc
            best_model = model

    test_acc = test(best_model, testloader, device)  # evaluate best model

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    sparsity = calculate_sparsity(best_model)  # compute % of pruned weights
    results.append((lam, test_acc, sparsity))  # store results for comparison

# visualize trade-off and pruning behavior
plot_tradeoff(results)
plot_gate_distribution(best_model)