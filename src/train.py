import torch
import torch.nn.functional as F
from loss import sparsity_loss, calculate_sparsity

def train(model, trainloader, optimizer, scheduler, epoch, total_epochs, lambda_max, device):

    lam = lambda_max * (epoch / total_epochs)  # gradually increase sparsity pressure

    model.train()
    total_ce, correct, total = 0, 0, 0

    for x, y in trainloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        ce = F.cross_entropy(out, y)
        sp = sparsity_loss(model)  # L1 penalty on all gate values

        loss = ce + lam * sp  # combined objective: accuracy + sparsity

        loss.backward()
        optimizer.step()

        total_ce += ce.item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    scheduler.step()  # adjust learning rate over epochs

    acc = 100 * correct / total
    sparsity = calculate_sparsity(model)  # % of gates effectively pruned

    print(f"Epoch [{epoch+1}/{total_epochs}] | "
          f"CE Loss: {total_ce/len(trainloader):.4f} | "
          f"Acc: {acc:.2f}% | "
          f"Sparsity: {sparsity:.2f}% | "
          f"λ: {lam:.6f}")

    return acc