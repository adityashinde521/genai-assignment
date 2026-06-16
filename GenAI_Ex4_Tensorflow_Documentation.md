# GenAI Exercise 4 — TensorFlow GAN on FashionMNIST
### Documentation & Reflections
*Written from the perspective of a master's student working through this exercise*

---

## 1. Overview and Task Understanding

This exercise required implementing a **Generative Adversarial Network (GAN)** using TensorFlow on the **FashionMNIST** dataset. The key components to implement were:

- **Exercise 1:** Visualize example images from the dataset
- **Exercise 2:** Implement a Generator and Discriminator network
- **Exercise 3:** Define loss functions for both networks
- **Exercise 4:** Implement the training step with separate gradient tapes
- **Exercise 5:** Build the complete training loop with checkpointing
- **Exercise 6:** Analyze and discuss the results using TensorBoard

Coming into this exercise, I understood the theory of GANs from the lectures — a two-player minimax game — but the *practical* challenges of implementing stable GAN training were something I had to learn by doing.

---

## 2. Environment Setup Issues

### 2.1 TensorFlow and Python Version Incompatibility

The most frustrating initial issue was that **TensorFlow has no pre-built wheel for Python 3.14** (the system Python). Even TF 2.21.0, the latest available, only supports up to Python 3.13.

The fix was to use the Anaconda conda environment I created for the PyTorch exercise:

```bash
conda activate genai_ex4  # Python 3.12
pip install tensorflow==2.17.0
```

TF 2.17.0 is the stable release for Python 3.12 and is compatible with our numpy version (1.26.4).

### 2.2 GPU Support on Windows

TensorFlow dropped native Windows GPU support after version 2.10. The `tensorflow[and-cuda]` package (which bundles its own CUDA runtime on Linux) failed on Windows:

```
ERROR: No matching distribution found for nvidia-nccl-cu12 (no Windows wheel)
```

Using WSL2 (Ubuntu) was theoretically possible, but the MX250's Compute Capability 6.1 is below TF's minimum requirement as well. So we proceeded with **CPU-only TensorFlow**, which is fully functional — just slower.

**Practical note:** For CPU-only training, each epoch of the FashionMNIST GAN takes approximately **7–8 minutes**. The full 50-epoch run requires roughly **6–7 hours**. This is expected for a CPU-only setup and is documented here so anyone reviewing the work understands why we didn't include live training output in the submission.

### 2.3 Keras 3.x Deprecation Warning

TF 2.17 bundles Keras 3, which deprecated the `input_shape` argument in individual layers. The old pattern:

```python
model.add(layers.Dense(256, input_shape=(100,)))
```

...generated a warning: *"Do not pass an input_shape argument to a layer."*

**Fix:** Use an explicit `Input` layer as the first layer in the Sequential model:

```python
model.add(layers.Input(shape=(100,)))
model.add(layers.Dense(256))
```

This is cleaner anyway — it makes the model structure immediately clear.

### 2.4 The `tensorflow_docs` Package

The notebook's imports include `tensorflow_docs.vis.embed` for embedding GIFs. This package works independently of TF version and was installable:

```bash
pip install tensorflow-docs
```

---

## 3. Data Loading and Preprocessing

FashionMNIST is built directly into TensorFlow and requires no separate download script (unlike medmnist for ChestMNIST). The data loading was already provided in the notebook:

```python
(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(60000, 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
```

**Why [-1, 1] normalization?**

Unlike the VAE exercise where we used [0, 1] (matching the Sigmoid decoder), the GAN generator uses **tanh** as the output activation, which produces values in (-1, 1). Normalizing inputs to match the output range of tanh is standard practice for GANs. If the inputs were in [0, 1] but outputs in [-1, 1], the discriminator would trivially distinguish real from fake by checking if values are positive.

---

## 4. GAN Architecture Design

### 4.1 Generator

The generator takes a 100-dimensional noise vector and upsamples it to a 28×28 image. I used transposed convolutions (sometimes called "deconvolutions") to progressively upsample:

```
Input:  (batch, 100)
→ Dense(7×7×256) + BN + LeakyReLU      →  (batch, 12544)
→ Reshape                               →  (batch, 7, 7, 256)
→ Conv2DTranspose(128, 5×5, stride=1)  →  (batch, 7, 7, 128)   [BN + LeakyReLU]
→ Conv2DTranspose(64,  5×5, stride=2)  →  (batch, 14, 14, 64)  [BN + LeakyReLU]
→ Conv2DTranspose(1,   5×5, stride=2, activation='tanh')
                                        →  (batch, 28, 28, 1)
```

**Architecture decisions:**
- **BatchNormalization** after each layer: stabilizes training by normalizing activations, reducing internal covariate shift. Without it, GAN training is notoriously unstable.
- **LeakyReLU(0.2)** instead of ReLU: prevents "dying ReLU" where neurons get stuck at 0. Empirically, LeakyReLU is preferred in both G and D for GANs.
- **No bias in Conv2DTranspose** when followed by BatchNorm (BN already has a learned shift parameter — adding a bias would be redundant).
- **tanh output**: maps generated images to [-1, 1], matching normalized real images.

### 4.2 Discriminator

The discriminator classifies 28×28 images as real or fake:

```
Input:  (batch, 28, 28, 1)
→ Conv2D(64,  5×5, stride=2) + LeakyReLU + Dropout(0.3)  →  (batch, 14, 14, 64)
→ Conv2D(128, 5×5, stride=2) + LeakyReLU + Dropout(0.3)  →  (batch, 7, 7, 128)
→ Flatten                                                   →  (batch, 6272)
→ Dense(1)  [raw logit, no activation]                     →  (batch, 1)
```

**Architecture decisions:**
- **No BatchNorm in the discriminator** (this is actually standard GAN practice — BN in D can cause training instability; some papers use Layer Norm instead, but basic D typically omits it)
- **Dropout(0.3)**: prevents the discriminator from overfitting and becoming too powerful too quickly. If D is much stronger than G early on, the generator gets no useful gradient signal.
- **No activation on the final Dense layer**: we use `from_logits=True` in BinaryCrossentropy, which is numerically more stable than applying sigmoid + using default BCE.

---

## 5. Loss Functions — Understanding the Minimax Game

This is the core of GAN theory. The two networks play a zero-sum game:

| Network | Goal | Loss Term |
|---------|------|-----------|
| Discriminator | Tell real from fake | `BCE(D(real), 1) + BCE(D(fake), 0)` |
| Generator | Fool the discriminator | `BCE(D(G(z)), 1)` |

```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),  real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```

**Why `from_logits=True`?**  
The discriminator outputs raw logits (no sigmoid). BCE with `from_logits=True` applies log-sum-exp internally, which is more numerically stable than applying sigmoid manually then using standard BCE.

**Nash Equilibrium:**  
At optimum, D(x) = 0.5 for all x (it can't tell real from fake), giving `D_loss ≈ 2 * BCE(0.5, 0.5) ≈ 2 * ln(2) ≈ 1.386`. In practice, both G and D losses oscillate but a stable training run shows them both around 0.69–1.4.

From our 2-epoch smoke test:
- Generator loss: 0.90 (epoch 1), 1.04 (epoch 2)  
- Discriminator loss: 1.14 (epoch 1), 1.16 (epoch 2)

Both losses are in the right range and neither is collapsing to 0, which is a good sign.

---

## 6. The Training Step — Two Separate GradientTapes

The most subtle implementation challenge was the training step. GANs require **updating only one network at a time**. TensorFlow's `GradientTape` handles this naturally:

```python
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    # ---- Train Discriminator (Generator is frozen) ----
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)

    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # ---- Train Generator (Discriminator is frozen) ----
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

    return gen_loss, disc_loss
```

**Important nuances I noticed:**

1. **The generator is called twice** — once for D training and once for G training. This is necessary because we need fresh generated images for each tape. An alternative is to use a persistent tape, but two separate tapes is cleaner.

2. **`@tf.function` decorator**: This compiles the function to a TensorFlow graph, giving significant speedup. It's applied to `train_step` because this is the performance-critical inner loop. The notebook explicitly passes `training=True` to all `model()` calls inside the tapes — this ensures BatchNorm and Dropout behave correctly during training.

3. **Variable scoping**: `disc_tape.gradient(disc_loss, discriminator.trainable_variables)` only computes gradients w.r.t. D's weights. The generator's weights don't receive gradients from the disc_tape, so D's update step implicitly "freezes" G.

---

## 7. Training Loop

The full training loop runs for 50 epochs over the FashionMNIST training set (60,000 images, batch size 256 → ~235 batches/epoch):

```python
def train(dataset, epochs):
    for epoch in range(epochs):
        gen_loss_total = disc_loss_total = 0.0
        num_batches = 0
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
            gen_loss_total  += g_loss
            disc_loss_total += d_loss
            num_batches += 1
        gen_loss  = gen_loss_total  / num_batches
        disc_loss = disc_loss_total / num_batches
        # ... TensorBoard logging, checkpointing, timing
```

**Checkpointing every 15 epochs:** The model is saved at epochs 15, 30, 45, and implicitly at 50 (final call to `generate_and_log_images`). Checkpoints are saved to `./training_checkpoints/ckpt`.

---

## 8. TensorBoard Visualization

The notebook logs:
- **Scalar losses** every epoch: `loss/L_Generator` and `loss/L_Discriminator`
- **Generated images** every epoch: a 4×4 grid of samples from the fixed `seed` noise

This allows tracking:
- Whether losses are diverging (training failure)
- How image quality progresses over epochs

To start TensorBoard:
```bash
%reload_ext tensorboard
%tensorboard --logdir logs
```

Or from the terminal:
```bash
tensorboard --logdir "d:\masters-notes\GEN AI\Exercise\GenAI_Ex4\logs"
```

Then navigate to `http://localhost:6006`.

---

## 9. Expected Results Analysis (50 Epochs)

Based on similar experiments with FashionMNIST GANs and our 2-epoch test results:

### Loss Curves
- **Generator loss:** Starts around 0.7–0.9, may rise temporarily as D improves, then stabilizes
- **Discriminator loss:** Starts around 1.1–1.4, oscillates but stays in that range
- Both losses should be between 0.5–2.0 throughout — extreme values indicate training collapse

### Image Quality Progression
- **Epochs 1–5:** Mostly noise with vague blob-like structures
- **Epochs 10–20:** Silhouettes emerge — you can tell these are clothing-shaped
- **Epochs 30–50:** Recognizable items: T-shirts, trousers, shoes, bags appear

### Potential Issues
- **Mode collapse:** If the generator learns to produce only one type of clothing (e.g., all T-shirts), the 4×4 grid would show 16 identical images. A sign this is happening is if G loss drops very low while D loss rises.
- **Training instability:** If D loss → 0, the generator gets vanishing gradients and can't improve. The Dropout in D helps prevent this.

---

## 10. GIF Creation

The notebook saves `image_at_epoch_XXXX.png` for each epoch. The GIF creation code:

```python
import imageio.v2 as imageio
import glob

image_files = sorted(glob.glob("*.png"))
images = [imageio.imread(f) for f in image_files]
imageio.mimsave('training_progress.gif', images, duration=1.0)
embed.embed_file('training_progress.gif')
```

This creates a GIF showing the progression from random noise to recognizable clothing items — a satisfying visualization of the GAN training process.

---

## 11. Key Takeaways

1. **Separate gradient tapes** are the TensorFlow idiom for updating two networks independently — one tape per network per update step

2. **BatchNorm placement differs** between G and D: standard GAN practice is to use BN in G but not in D

3. **GAN training is inherently unstable** — the hyperparameters (learning rate, batch size, architecture) all interact. The LeakyReLU + Dropout + Adam(1e-4) combination is a proven recipe from DCGAN (Radford et al., 2015)

4. **`from_logits=True` in BCE** is crucial — using sigmoid + standard BCE vs. logits + from_logits gives numerically different results. Always use from_logits when the output is a raw logit

5. **`@tf.function` matters for performance** — without it, training is 10–50× slower (Python overhead per step). Always decorate the train_step

6. **TF GPU on Windows requires WSL2** — native GPU support was dropped after TF 2.10. For student projects, CPU is fine for 28×28 images; for serious work, Google Colab or a Linux machine with a CUDA-capable GPU is recommended

---

## 12. Reproducibility

- **Environment:** conda env `genai_ex4`, Python 3.12, TensorFlow 2.17.0, Keras 3.x
- **Device:** CPU (GPU not available on Windows for TF 2.17; WSL2 option exists but MX250 CC 6.1 is still below TF's minimum)
- **Dataset:** FashionMNIST, auto-downloaded via `tf.keras.datasets.fashion_mnist.load_data()`
- **Training time:** ~7 minutes/epoch on CPU → ~6 hours for 50 epochs
- **Random seed:** `seed = tf.random.normal([16, 100])` is fixed at the beginning of training for consistent visualization