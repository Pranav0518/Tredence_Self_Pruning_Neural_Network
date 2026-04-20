import torch
import random
import numpy as np
import os
import pandas as pd

# ==============================
# SEED (REPRODUCIBILITY)
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================
# SAVE RESULTS TO CSV
# ==============================
def save_results(results, path="experiments/results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results, columns=["lambda", "accuracy", "sparsity"])
    df.to_csv(path, index=False)


# ==============================
# SAVE MODEL
# ==============================
def save_model(model, path="experiments/best_model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


# ==============================
# LOAD MODEL
# ==============================
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


# ==============================
# COUNT PARAMETERS
# ==============================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================
# GET DEVICE
# ==============================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# FORMAT LOG OUTPUT
# ==============================
def log_epoch(epoch, total_epochs, loss, acc, sparsity, lam):
    print(f"Epoch [{epoch+1}/{total_epochs}] | "
          f"Loss: {loss:.4f} | "
          f"Acc: {acc:.2f}% | "
          f"Sparsity: {sparsity:.2f}% | "
          f"λ: {lam:.6f}")