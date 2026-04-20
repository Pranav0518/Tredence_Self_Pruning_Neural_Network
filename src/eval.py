def test(model, testloader, device):

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            _, pred = out.max(1)

            total += y.size(0)
            correct += pred.eq(y).sum().item()

    return 100 * correct / total