# GenAI Exercise 4 — PyTorch VAE on ChestMNIST
### Documentation & Reflections
*Written from the perspective of a master's student working through this exercise*

---

## 1. Overview and Task Understanding

This exercise asked me to implement a **Variational Autoencoder (VAE)** using PyTorch on the **ChestMNIST** dataset from MedMNIST. The key deliverables were:

- Load and preprocess the dataset
- Implement a VAE with encode, reparameterize, decode, and forward methods
- Define an appropriate loss function (reconstruction + KL divergence)
- Train the model and visualize reconstructions
- Generate synthetic samples and discuss the results

At first glance, this seemed like a straightforward extension of what we learned in the lectures about VAEs. However, working through the implementation revealed a number of subtle decisions and practical challenges.

---

## 2. Environment Setup — What Went Wrong First

My first hurdle was the environment itself. My system Python was **3.14.6**, and neither PyTorch nor TensorFlow has pre-built wheels for Python 3.14 yet (Python 3.14 was released just recently). TensorFlow had no matching distribution at all.

**Resolution:** I installed **Anaconda Navigator** and created a dedicated conda environment:

```bash
conda create -n genai_ex4 python=3.12 -y
```

Then installed the required packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install medmnist matplotlib numpy imageio jupyter ipykernel
```

**Second issue — GPU incompatibility:** My NVIDIA GeForce MX250 (a Pascal-generation GPU from 2017) has **CUDA Compute Capability 6.1**. Modern PyTorch 2.x requires CC ≥ 7.5 (Turing and newer). The GPU was visible to PyTorch (`torch.cuda.is_available()` returned `True`) but crashed at runtime with:

```
RuntimeError: GET was unable to find an engine to execute this computation
```

**Resolution:** I added a safe device detection that tests GPU usability before committing to it:

```python
device = torch.device('cpu')
if torch.cuda.is_available():
    try:
        torch.tensor([1.0]).cuda()
        device = torch.device('cuda')
    except RuntimeError:
        pass  # MX250 CC 6.1 not compatible — fall back to CPU
print(f"Using device: {device}")
```

This is a good reminder that `is_available()` only checks if a CUDA driver is installed, not whether the GPU is actually compatible with the compiled PyTorch build.

---

## 3. Dataset: ChestMNIST

**ChestMNIST** is part of the MedMNIST benchmark. It contains **chest X-ray images** resized to 28×28 grayscale, originally derived from the NIH ChestX-ray14 dataset. The labels are multi-label (14 disease categories), but for the generative task we ignore labels entirely.

- **Training set:** 78,468 images
- **Test set:** 22,433 images
- **Input shape:** (1, 28, 28) — single channel, 28×28 pixels
- **Pixel range after `ToTensor()`:** [0.0, 1.0]

**Preprocessing decision:** I used only `transforms.ToTensor()`, which normalizes pixels from [0, 255] to [0.0, 1.0]. This was intentional because:

1. The VAE decoder's final layer uses **Sigmoid**, which also outputs values in [0, 1]. For the Binary Cross-Entropy (BCE) reconstruction loss to make sense, inputs and outputs must be in the same range.
2. No color normalization to [-1, 1] was applied (that would require `tanh` in the decoder instead, and is more common in GANs).
3. No data augmentation was applied — we want the VAE to model the true data distribution, not an augmented version of it.

---

## 4. VAE Architecture Design

### Why Convolutional Layers?

The straightforward choice for 28×28 images would be a fully connected VAE, but convolutional layers are far more parameter-efficient and better at capturing spatial structure. They also generalize what we see in the literature for image-based VAEs.

### Encoder

```
Input: (batch, 1, 28, 28)
→ Conv2d(1→32, 3×3, stride=2, pad=1) + ReLU  →  (batch, 32, 14, 14)
→ Conv2d(32→64, 3×3, stride=2, pad=1) + ReLU  →  (batch, 64, 7, 7)
→ Flatten  →  (batch, 3136)
→ Linear(3136 → 16) = mu
→ Linear(3136 → 16) = logvar
```

The two conv layers each halve the spatial dimensions (stride=2), giving us a 3136-dimensional feature vector before the latent projections.

### Why latent_dim = 16?

I chose 16 as the latent dimension after considering:
- **Too small (e.g., 2):** The latent space wouldn't have enough capacity to encode the diverse pathological features in chest X-rays. Two dimensions might work for MNIST digits but not for medical images.
- **Too large (e.g., 256):** Risks posterior collapse (the KL term pushes unused dimensions to the prior, so many are wasted). Also increases the number of FC layer parameters significantly.
- **16** is a reasonable middle ground for 28×28 grayscale images. It's also consistent with what I found in VAE tutorials for MedMNIST datasets.

### Reparameterization Trick

This is the key innovation of VAEs over regular autoencoders. Instead of sampling directly from `q(z|x) = N(mu, sigma^2)` (which is non-differentiable), we write:

```
z = mu + eps * exp(0.5 * logvar)    where  eps ~ N(0, I)
```

The network learns `mu` and `logvar` (log-variance for numerical stability), and the randomness is expressed through `eps`, which doesn't depend on the parameters. This allows gradients to flow through `mu` and `logvar` during backpropagation.

### Decoder

```
z (batch, 16)
→ Linear(16 → 256) + ReLU
→ Linear(256 → 3136)             [no activation — signed values improve decoder expressiveness]
→ Reshape  →  (batch, 64, 7, 7)
→ ConvTranspose2d(64→32, 4×4, stride=2, pad=1) + ReLU  →  (batch, 32, 14, 14)
→ ConvTranspose2d(32→1, 4×4, stride=2, pad=1) + Sigmoid  →  (batch, 1, 28, 28)
```

I used `ConvTranspose2d` (transposed convolutions, or "deconvolutions") rather than upsampling + conv. The kernel size 4 with stride 2 and padding 1 exactly doubles the spatial dimensions: `(7-1)*2 - 2*1 + 4 = 14` ✓

**Why no activation after the second linear layer?** A ReLU here would clamp all 3136 feature map values to non-negative before the ConvTranspose2d layers, halving the representable input space for the deconvolution. Removing it lets the transpose-conv layers receive signed activations and use their full dynamic range. The final `Sigmoid` in `decoder_conv` still bounds the output to [0, 1].

**Total parameters: 962,817** — manageable for CPU training.

---

## 5. Loss Function

The VAE loss has two terms:

```
L = Reconstruction Loss + KL Divergence
  = BCE(recon_x, x) + KL[q(z|x) || p(z)]
```

### Reconstruction Loss (BCE)

Binary Cross-Entropy is appropriate for pixel values in [0, 1] (treating each pixel as a Bernoulli variable). We sum over all pixels and normalize by the batch size:

```python
BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
```

### KL Divergence

Since `q(z|x) = N(mu, exp(logvar))` and `p(z) = N(0, I)`, the KL divergence has a closed-form solution:

```
KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
```

This term encourages the learned latent distribution to stay close to the standard Gaussian prior, which is what allows us to sample from the prior and decode it into meaningful images.

**Trade-off:** A large KL term forces the model to make less use of the latent space (posterior collapse risk). A small KL term allows the encoder to "cheat" by encoding too much specific information. I kept them unweighted (beta=1 standard VAE) since the dataset is relatively simple.

---

## 6. Training

I used **Adam** with `lr=1e-3` for 10 epochs, as specified by the skeleton. Batch size was 128.

The `train_vae` function accepts `device` as an explicit parameter (defaulting to the device the model lives on via `next(model.parameters()).device`) and returns the per-epoch loss history as a list. After training completes, the loss curve is plotted with matplotlib, giving a visual confirmation of convergence without relying on console printouts alone.

**Observed loss progression (2-epoch test):**
- Epoch 1: ~473.84
- Epoch 2: ~467.16

The loss decreases monotonically, which is a good sign. For the full 10-epoch run, we'd expect the loss to continue dropping and plateau around epoch 5–8. The plotted loss curve makes it easy to spot divergence or stagnation at a glance.

**Training time on CPU:** Approximately 2–3 minutes per epoch with 78,468 images at batch size 128 (~614 batches/epoch). Total for 10 epochs: ~25–30 minutes on CPU.

---

## 7. Results: Reconstruction

After training, I displayed 8 pairs of original vs. reconstructed images from the test set. The reconstructions capture the overall structure (lungs, ribcage, general brightness) but appear blurrier than the originals. Row labels ("Original" / "Reconstructed") are placed using `ax.text(transform=ax.transAxes)` rather than `set_ylabel`, since `axis('off')` suppresses axis label rendering regardless of when `set_ylabel` is called.

**Why blurry?**  
The BCE reconstruction loss averages pixel errors independently. The model minimizes this by producing the "expected" reconstruction — an average over possible images given the latent code. This averaging effect creates blur. This is a fundamental limitation of VAEs with pixel-wise losses; GANs avoid this by using a discriminator.

---

## 8. Results: Generation

I sampled 16 latent vectors `z ~ N(0, I)` and decoded them. The generated images show lung-like structures, though they lack the clinical detail of real X-rays. The variety across samples demonstrates that the latent space learned a smooth manifold — different regions of the prior correspond to different image patterns.

**Discussion points:**
- The generated images don't correspond to specific disease states (we didn't condition on labels)
- A higher latent dimension or deeper architecture would improve detail
- The 28×28 resolution is inherently limited — this is a benchmark dataset size, not clinical resolution
- A conditional VAE (CVAE) would allow generating images with specific disease conditions

---

## 9. Key Takeaways

1. **Reparameterization is essential** — without it, you can't backpropagate through a sampling operation
2. **GPU compatibility is not guaranteed** — always test CUDA usability before committing to it
3. **VAEs produce smooth but blurry outputs** — this is the bias/variance trade-off of the evidence lower bound (ELBO) objective
4. **Environment management matters** — the conda environment with pinned Python 3.12 saved significant debugging time
5. **Medical images are interesting** — even at 28×28, you can see anatomical structures in the learned representations
6. **Decoder architecture matters** — a trailing ReLU before reshaping into transposed-conv layers constrains the feature space to non-negative values; removing it gives the deconvolution layers full signed dynamic range
7. **`axis('off')` hides labels** — use `ax.text(transform=ax.transAxes)` instead of `set_ylabel` when turning off axes; the latter is suppressed at render time regardless of call order
8. **Always return training metrics** — having `train_vae` return the loss history enables post-training plotting without re-running training

---

## 10. Reproducibility

- **Environment:** conda env `genai_ex4`, Python 3.12, PyTorch 2.11.0, medmnist 3.0.2
- **Device:** CPU (NVIDIA MX250 CC 6.1 not supported by PyTorch 2.11+)
- **Random seed:** not fixed (results will vary between runs, but the architecture and training are deterministic given a seed)
- **Dataset:** ChestMNIST from MedMNIST, downloaded via `medmnist.ChestMNIST(download=True)`