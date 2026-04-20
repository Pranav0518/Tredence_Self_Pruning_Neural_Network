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
  <img src="experiments/Accuracy_sparsity_tradeoff.png" width="92%" alt="Accuracy vs sparsity Tradeoff - 84.03% acc at 79.2% sparse, 84.61% at 96.2% sparse, 82.47% at 99.7% sparse."/>
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
    <th align="center"> Test Accuracy</th>
    <th align="center"> Sparsity</th>
    <th align="center"> Observation</th>
  </tr>
  <tr>
    <td align="center"><code>1e-5</code></td>
    <td align="center"><b>84.03 %</b></td>
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
    <td align="center">Extreme pruning </td>
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

## g = sigmoid(s)

where  s  is the gate score.

The sparsity loss is:

## L_sp = sum(sigmoid(s_i))

Derivative:

## dL_sp / ds_i = sigmoid(s_i) * (1 - sigmoid(s_i))

### Interpretation

- This gradient pushes gate scores downward  
- For unimportant weights:
  - Classification loss gradient is weak  
  - Sparsity term dominates → gates → 0  
- For important weights:
  - Classification gradient keeps gates active  

### Result

- Gates naturally split into:
  - ~0 → pruned weights  
  - ~1 → important weights  

👉 This creates **automatic sparsity during training**

---

## 🧠 Why CNN + PrunableLinear (and not plain MLP)

A simple fully connected network on flattened CIFAR-10 images:

3 x 32 x 32 = 3072 \text {features}


fails to capture spatial structure and typically underperforms.

### Our Approach

- Use a **CNN backbone** for feature extraction  
- Apply pruning only in **fully connected layers**

### Benefits

- CNN preserves spatial information → higher accuracy  
- Prunable layers learn redundancy → sparsity  

Achieves both:
- Strong performance (~84%)  
- High compression (~99% sparsity)

### Key Insight

- Most redundancy lies in **dense layers**, not convolutional layers  
- This validates applying pruning selectively    

</details>

---

## Architecture

```mermaid
flowchart LR
    A["Image<br/>3 × 32 × 32"] --> B["Conv2D<br/>3 → 32"]
    B --> C["BatchNorm + ReLU"]
    C --> D["Residual Block<br/>32 channels"]
    D --> E["MaxPool<br/>16 × 16"]

    E --> F["Conv2D<br/>32 → 64"]
    F --> G["BatchNorm + ReLU"]
    G --> H["Residual Block<br/>64 channels"]
    H --> I["MaxPool<br/>8 × 8"]

    I --> J["Flatten<br/>4096"]

    J --> K["PrunableLinear<br/>4096 → 512"]
    K --> L["ReLU + Dropout"]
    L --> M["PrunableLinear<br/>512 → 10"]

    M --> N["Logits"]

    style A fill:#0B2C5A,color:#fff,stroke:#0B2C5A
    style N fill:#C9A24E,color:#fff,stroke:#C9A24E
    style K fill:#E9F1FB,color:#0B2C5A
    style M fill:#E9F1FB,color:#0B2C5A
```

<table align="center">
  <tr><td>Backbone</td><td align="right"><b>CNN + Residual Blocks</b></td></tr>
  <tr><td>PrunableLinear layers</td><td align="right"><b>2</b></td></tr>
  <tr><td>Prunable weights</td><td align="right"><b>Fully connected layers only</b></td></tr>
  <tr><td>Gate parameters</td><td align="right"><b>One per weight</b></td></tr>
  <tr><td>Pruning type</td><td align="right"><b>Unstructured (weight-level)</b></td></tr>
  <tr><td>Convolutions</td><td align="right"><b>Used for feature extraction</b></td></tr>
</table>

<details>

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
  <tr><td>Optimizer</td><td><code>AdamW</code> with parameter groups</td></tr>
  <tr><td>Learning rate</td><td>weights <code>1e-3</code> / gate scores <code>5e-3</code></td></tr>
  <tr><td>Weight decay</td><td><code>1e-4</code> on weights; <b>0</b> on gate scores</td></tr>
  <tr><td>Scheduler</td><td><code>CosineAnnealingLR</code>, <code>T_max = 50</code></td></tr>
  <tr><td>Loss</td><td>CrossEntropy + λ · L1(gates)</td></tr>
  <tr><td>Gate initialization</td><td><code>s = 2.0</code> → <code>σ(s) ≈ 0.88</code></td></tr>
  <tr><td>Prune threshold</td><td><code>σ(s) &lt; 0.01</code></td></tr>
  <tr><td>λ values</td><td><code>1e-5</code>, <code>1e-4</code>, <code>1e-3</code></td></tr>
  <tr><td>λ schedule</td><td>Linear increase over training epochs</td></tr>
  <tr><td>Epochs</td><td><code>50</code></td></tr>
  <tr><td>Batch size</td><td><code>256</code></td></tr>
  <tr><td>Data augmentation</td><td><code>RandomCrop</code> + <code>HorizontalFlip</code> + <code>ColorJitter</code></td></tr>
  <tr><td>Device</td><td>GPU (if available) / CPU fallback</td></tr>
</table>

<details>
<summary><b>🔥 Portability — runs on any standard setup</b></summary>

The implementation is designed to be simple and portable:

- Device auto-detect:
  <code>torch.device("cuda" if torch.cuda.is_available() else "cpu")</code>
- Runs on both GPU and CPU without code changes  
- Batch size can be adjusted based on available memory (default: <code>256</code>)  
- Uses standard PyTorch modules — no hardware-specific dependencies  
- All paths are relative, ensuring easy execution across environments  

### Practical Usage

- On GPU (recommended): faster training (~20 sec/epoch on CIFAR-10)  
- On CPU: slower but fully functional  
- Works seamlessly in:
  - Local environments  
  - Google Colab  
  - Standard Linux/Windows setups  

</details>
---

## 📊 Results

### Headline numbers


Every λ produces a *distinct and meaningful* operating point:

| Operating point | Best for |
|---|---|
| `λ = 1e-5` → **84.03 %** @ 79.2 % sparse | **Accuracy-focused**: strong performance with moderate pruning |
| `λ = 1e-4` → **84.61 %** @ **96.2 %** sparse | **Best trade-off** ⭐: highest accuracy with very high sparsity |
| `λ = 1e-3` → **82.47 %** @ **99.7 %** sparse | **Extreme compression** 🚀: near-total pruning with minimal accuracy drop |

---

### ✅ Sanity Checks

| # | Status | Assertion |
|---|:---:|---|
| 1 | ✅ **PASS** | Sparsity spans a wide range: 79 %, 96 %, 99 %. |
| 2 | ✅ **PASS** | Sparsity increases consistently with λ. |
| 3 | ✅ **PASS** | Test accuracy remains high across all λ values (>82%). |
| 4 | ✅ **PASS** | Model maintains performance even at extreme sparsity. |

---

### 🔬 Interpretation

- Increasing λ strengthens sparsity pressure → more weights are pruned  
- Moderate λ (1e-4) achieves the best balance between accuracy and compression  
- Even at ~99.7% sparsity, the model retains strong predictive ability  

---

### 🧠 Key Insight

The network successfully learns:

- Which weights are **important → retained**
- Which weights are **redundant → pruned**

This confirms that:

👉 **Pruning is happening during training, not as a post-processing step**

---

### ⚠️ Note on Hard Pruning

This implementation evaluates sparsity using a threshold:

```text id="4v0p6g"
σ(gate_scores) < 0.01
```
- We do not physically zero weights during evaluation
- However, near-zero gates effectively remove weights from computation

---

## 🧠 Deep Analysis

<details>
<summary><b>1. Stable accuracy under high sparsity — key observation</b></summary>

Across different λ values:

- λ = 1e-5 → 84.03 % @ 79.2 % sparsity  
- λ = 1e-4 → 84.61 % @ 96.2 % sparsity  
- λ = 1e-3 → 82.47 % @ 99.7 % sparsity  

Even as sparsity increases significantly, accuracy remains relatively stable.

### Interpretation

- The network contains **substantial redundancy**  
- Many weights can be removed without affecting predictions  
- Gates successfully identify **important vs redundant connections**

👉 This confirms effective **in-training pruning**

</details>

---

<details>
<summary><b>2. Redundancy is concentrated in fully connected layers</b></summary>

- Convolutional layers → feature extraction (remain dense)  
- Fully connected layers → classification (high redundancy)  

👉 Dense layers are significantly **over-parameterized**

</details>

---

<details>
<summary><b>3. Why <code>gate_init = 2.0</code> works well</b></summary>


σ′(s) = σ(s)(1 − σ(s))

The sigmoid derivative peaks at:

s = 0

At:

s = 2 → σ(s) ≈ 0.88

the gradient is still sufficiently large, so gates remain trainable.

<detail>
### Why not smaller initialization?
If initialized too low (e.g., σ ≈ 0.01):
Gates start near zero → already pruned, Gradient becomes very small, Learning becomes ineffective, Benefit of current initialization, Starts with most weights active, Allows the model to learn which weights to remove, Maintains strong gradient flow during early training
</details>
  
<details> <summary><b>4. λ calibration — intuition</b></summary>

Total loss:
L = CE + λ · Σ σ(gate_scores)
  
Understanding λ
Small λ → sparsity term is weak → accuracy dominates
Large λ → sparsity term is strong → aggressive pruning
Empirical behavior
λ	Effect
- 1e-5	Moderate pruning
- 1e-4	Best balance
- 1e-3	Extreme pruning
 
### Insight:
λ controls pressure toward sparsity, not exact sparsity level
Increasing λ gradually reveals the full accuracy–sparsity trade-off
</details>

<details> <summary><b>5. L1 on <code>σ(g)</code> vs exact L0 pruning</b></summary>

Exact L0 pruning
Non-differentiable
Computationally intractable
Practical alternative
L_sp = Σ σ(gate_scores)
Advantages
Fully differentiable
Works with standard gradient descent
Encourages sparsity naturally
Trade-off
Produces soft sparsity, not exact zeros
Requires thresholding (e.g., σ(g) < 0.01)
Key insight
λ controls the strength of sparsity pressure
Sparsity increases smoothly and predictably

👉 This makes L1 on sigmoid gates a strong and practical approximation to L0 pruning

</details>
---
