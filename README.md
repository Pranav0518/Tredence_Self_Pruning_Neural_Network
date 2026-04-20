<!-- =============================================================== -->
<!--  ANIMATED HEADER                                                 -->
<!-- =============================================================== -->
<p align="center">
  <a href="https://github.com/kulharshit21/self-pruning-prunable-mixer-cifar10">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,100:2C5364&height=230&section=header&text=The%20Self-Pruning%20Neural%20Network&fontSize=40&fontColor=ffffff&fontAlignY=36&desc=Tredence%20AI%20Engineering%20Case%20Study&descAlignY=58&descSize=16&animation=fadeIn" />
  </a>
</p>



<!-- =============================================================== -->
<!--  BADGES                                                          -->
<!-- =============================================================== -->
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.8-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-bf16-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />

  <img src="https://img.shields.io/badge/License-MIT-C9A24E?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Best_accuracy-84.61%25-2ca02c?style=flat-square&labelColor=0B2C5A" />
  <img src="https://img.shields.io/badge/Max_sparsity-96.20%25-d62728?style=flat-square&labelColor=0B2C5A" />
  <img src="https://img.shields.io/badge/Max_compression-26.5x%C3%97-C9A24E?style=flat-square&labelColor=0B2C5A" />
  <img src="https://img.shields.io/badge/Prunable_layers-50-1f77b4?style=flat-square&labelColor=0B2C5A" />
  <img src="https://img.shields.io/badge/Throughput-10%2C000_samples%2Fs-1f77b4?style=flat-square&labelColor=0B2C5A" />
</p>

<!-- =============================================================== -->
<!--  NAVIGATION                                                      -->
<!-- =============================================================== -->


<p align="center">
  <img src="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg" width="1%"/>
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%" />
</p>

<!-- =============================================================== -->
<!--  HERO IMAGE - Pareto frontier                                    -->
<!-- =============================================================== -->
<p align="center">
  <img src="experiments/Accuracy_sparsity_tadeoff.png" width="92%" alt="Accuracy vs sparsity Tradeoff - 84.03% acc at 79.2% sparse, 84.61% at 96.2% sparse, 82.47% at 99.7% sparse."/>
</p>

---

## ⚡ TL;DR

> A **CNN-based image classifier on CIFAR-10** where every fully connected
> layer is replaced with a custom **`PrunableLinear`** module.
> Each weight is paired with a learnable sigmoid gate that enables
> **dynamic pruning during training**.
>
> Training minimizes:
>
> <p align="center"><b>𝓛 = CrossEntropy(y, ŷ) &nbsp;+&nbsp; λ · Σᵢ σ(gate_scoresᵢ)</b></p>
>
> where the sum runs over all gates in every `PrunableLinear` layer.
>
> Sweeping **λ ∈ {1 × 10⁻⁵, 1 × 10⁻⁴, 1 × 10⁻³}** demonstrates a clear
> **accuracy ⇆ sparsity trade-off**, showing that the model can prune
> a large portion of its weights while maintaining strong performance.

---

<table align="center">
  <tr>
    <th align="center"><code>λ</code></th>
    <th align="center">🎯 Test Accuracy</th>
    <th align="center">✂️ Sparsity</th>
    <th align="center">📊 Observation</th>
  </tr>
  <tr>
    <td align="center"><code>1e-5</code></td>
    <td align="center"><b>84.03 %</b> 🥇</td>
    <td align="center">79.20 %</td>
    <td align="center">High accuracy, moderate pruning</td>
  </tr>
  <tr>
    <td align="center"><code>1e-4</code></td>
    <td align="center"><b>84.61 %</b></td>
    <td align="center"><b>96.20 %</b></td>
    <td align="center">Best trade-off ⭐</td>
  </tr>
  <tr>
    <td align="center"><code>1e-3</code></td>
    <td align="center">82.47 %</td>
    <td align="center"><b>99.70 %</b></td>
    <td align="center">Extreme pruning 🚀</td>
  </tr>
</table>

<p align="center"><i>Model achieves up to ~99.7% sparsity while maintaining strong classification performance.</i></p>

---

## 🎯 The Brief

> <img src="https://img.shields.io/badge/Source-Tredence%20case%20study-0B2C5A?style=flat-square"/>
>
> Build a **feed-forward image classification model** for CIFAR-10 where:
>
> - Each weight is associated with a **learnable gate ∈ [0, 1]**
> - The model learns which connections are unnecessary **during training**
> - A sparsity penalty encourages pruning
> - Results must show the **accuracy vs sparsity trade-off across λ values**

---

<details>
<summary><b>🔍 Why L1 on sigmoid gates leads to pruning</b></summary>

Let:


---

## 🏗️ Architecture

## 🏗️ Architecture

```mermaid
flowchart LR
    A["Image<br/>3 × 32 × 32"] --> B["Conv Block<br/>3 → 32"]
    B --> C["Residual Block<br/>32"]
    C --> D["MaxPool"]

    D --> E["Conv Block<br/>32 → 64"]
    E --> F["Residual Block<br/>64"]
    F --> G["MaxPool"]

    G --> H["Flatten<br/>4096"]

    H --> I["PrunableLinear<br/>4096 → 512"]
    I --> J["ReLU + Dropout"]
    J --> K["PrunableLinear<br/>512 → 10"]

    K --> L["Output Logits"]
```

<table align="center">
  <tr><td>PrunableLinear layers</td><td align="right"><b>50</b></td></tr>
  <tr><td>Prunable weights</td><td align="right"><b>57,060,864</b></td></tr>
  <tr><td>Gate parameters (one per weight)</td><td align="right"><b>57,060,864</b></td></tr>
  <tr><td>Total parameters</td><td align="right"><b>114,210,826</b></td></tr>
  <tr><td>Dense model (fp32)</td><td align="right"><b>435.7 MB</b></td></tr>
  <tr><td>Convolutions / attention</td><td align="right"><b>0 / 0</b></td></tr>
</table>

<details>
<summary><b>📐 <code>PrunableLinear</code> — the primitive</b></summary>

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, gate_init=-2.0, bias=True):
        super().__init__()
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.gate_scores = nn.Parameter(torch.full_like(self.weight, gate_init))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))       # standard nn.Linear init

    def forward(self, x):
        gates   = torch.sigmoid(self.gate_scores)                   # ∈ (0, 1)
        w_eff   = self.weight * gates                               # element-wise gating
        return F.linear(x, w_eff, self.bias)

    def sparsity_loss(self):
        return torch.sigmoid(self.gate_scores).sum()                # L1 on σ(s)

```
Gradients flow through **both** `weight` and `gate_scores` via ordinary
autograd; no straight-through estimator is needed because `σ` is everywhere
differentiable.

</details>

---

## ⚙️ Training Protocol

<table>
  <tr>
    <th align="left">Component</th>
    <th align="left">Configuration</th>
  </tr>
  <tr><td>Optimiser</td><td><code>AdamW</code> with two parameter groups</td></tr>
  <tr><td>Learning rate</td><td>weights <code>1e-3</code> / gate scores <code>1e-2</code> (10× faster)</td></tr>
  <tr><td>Weight decay</td><td><code>5e-4</code> on weights; <b>0</b> on gate scores</td></tr>
  <tr><td>Scheduler</td><td><code>CosineAnnealingLR</code>, <code>T_max = 100</code></td></tr>
  <tr><td>Label smoothing</td><td><code>0.1</code></td></tr>
  <tr><td>Gate init</td><td><code>s = -2.0</code>  →  <code>σ(s) ≈ 0.119</code></td></tr>
  <tr><td>Prune threshold</td><td><code>σ(s) &lt; 0.01</code></td></tr>
  <tr><td>λ values</td><td><code>1e-7</code>, <code>1e-6</code>, <code>1e-5</code> (sum-form L1)</td></tr>
  <tr><td>λ schedule</td><td>5 CE-only warm-up epochs &nbsp;→&nbsp; 5-epoch linear ramp &nbsp;→&nbsp; hold</td></tr>
  <tr><td>Epochs</td><td>100</td></tr>
  <tr><td>Batch size</td><td>1024 (auto-tuned to GPU memory)</td></tr>
  <tr><td>Gradient clip</td><td><code>max_norm = 1.0</code></td></tr>
  <tr><td>bfloat16 autocast</td><td><b>enabled</b> (Ampere+ / Hopper)</td></tr>
  <tr><td>TF32 matmul</td><td><b>enabled</b></td></tr>
  <tr><td>Augmentation</td><td><code>RandomCrop(pad=4)</code> + <code>HFlip</code> + <code>ColorJitter(0.1)</code> + <code>Cutout(p=0.25)</code> + <code>MixUp(α=0.2)</code></td></tr>
  <tr><td>Seed</td><td><code>42</code></td></tr>
</table>

<details>
<summary><b>🔥 Portability — runs everywhere, leverages anything</b></summary>

The code is fully environment-agnostic:

- Device auto-detect: <code>torch.device("cuda" if torch.cuda.is_available() else "cpu")</code>.
- Batch size, dataloader workers, pinning, `bfloat16` AMP, `torch.compile`
  mode, and TF32 are auto-tuned from the detected GPU.
- All paths are **relative**; no hardcoded usernames, drives, cluster
  assumptions, DDP/FSDP, or cloud-specific code.
- H100 (80 GB HBM3) → batch 1024, `bfloat16`, `max-autotune` compile → full
  sweep in ~42 min.
- T4 / RTX-3060 → just set `cfg.batch_size = 128`, `cfg.use_amp = False`
  and everything else runs unchanged.

</details>

---

