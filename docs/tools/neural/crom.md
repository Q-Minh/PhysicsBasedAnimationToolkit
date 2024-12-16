# Continuous Reduced Order Model (CROM) for PDE Simulations

This script implements a **Continuous Reduced Order Model (CROM)** for Partial Differential Equation (PDE) simulations. It leverages an encoder-decoder architecture to create latent representations of high-dimensional PDE solutions, enabling efficient and accurate approximation and sampling.

## Overview

The CROM framework consists of three main components:

1. **ShapeDataset**: Handles loading and preprocessing of mesh data for PDE simulations.
2. **Encoder-Decoder Architecture**: Encodes PDE solutions into latent representations and reconstructs outputs using a decoder.
3. **Gauss-Newton and Sampling Algorithms**: Provides robust sampling and efficient latent space optimization for simulation.

The script supports training, sampling, and running the model for PDE simulation tasks.

---

## Prerequisites

The following Python libraries are required:

- `torch`
- `torchvision`
- `meshio`
- `numpy`
- `argparse`
- `random`
- `math`
- `tensorboard`

Install dependencies using `pip`:

```bash
pip install torch torchvision meshio numpy tensorboard
```

---

## Command-Line Arguments

The script accepts the following command-line arguments:

| Argument                       | Description                                                                 | Default      |
|--------------------------------|-----------------------------------------------------------------------------|--------------|
| `-i`, `--input`               | Path to input data directory                                                | Required     |
| `-o`, `--output`              | Path to output directory                                                    | `.`          |
| `-d`, `--output-dimensions`   | Dimensionality of the PDE solution output                                   | `1`          |
| `-m`, `--input-dimensions`    | Dimensionality of the PDE solution input                                    | `3`          |
| `-r`, `--latent-dimensions`   | Dimensionality of the latent space                                          | `64`         |
| `--epochs`                    | Number of training epochs                                                   | `100`        |
| `--batch-size`                | Batch size for training                                                     | `32`         |
| `--learning-rate`             | Learning rate for training                                                  | `1e-4`       |
| `--mode`                      | Mode of operation: `train`, `sample`, or `run`                              | `train`      |
| `--sampling-target-accuracy`  | Target accuracy for CROM sampling                                           | `1e-2`       |

Example usage for training:

```bash
python crom_simulation.py -i data/input -o results --epochs 50 --batch-size 16 --mode train
```

---

## Workflow Overview

### 1. Data Preprocessing

The `ShapeDataset` class loads and processes mesh data for PDE simulations. The meshes are preprocessed to normalize dimensions and prepare them for encoding.

```python
class ShapeDataset(Dataset):
    def __init__(self, mesh_path):
        mesh_files = [file for file in os.listdir(mesh_path) if file.endswith(".mesh") or file.endswith(".msh")]
        mesh_files = sorted(mesh_files, key=lambda file: int(file.split(".")[0]))
        meshes = [meshio.read(os.path.join(mesh_path, file)) for file in mesh_files]
        self.V = [mesh.points for mesh in meshes]

    def __len__(self):
        return len(self.V)

    def __getitem__(self, idx):
        return self.V[idx], self.V[0]  # Returns current and reference mesh
```

---

### 2. Encoder-Decoder Architecture

The **Encoder** compresses PDE solutions into a low-dimensional latent space. The **Decoder** reconstructs the solution from the latent representation.

#### Encoder:

```python
class Encoder(nn.Module):
    def __init__(self, p, d, r, conv_kernel_size=6, conv_stride_size=4):
        super().__init__()
        self.convolution = nn.ModuleList()
        L = p
        while L > 32:
            L = int((L - (conv_kernel_size - 1) - 1) / conv_stride_size + 1)
            self.convolution.append(nn.Conv1d(d, d, conv_kernel_size, conv_stride_size))
            self.convolution.append(nn.ELU())
        self.linear = nn.ModuleList([
            nn.Linear(L * d, 32),
            nn.ELU(),
            nn.Linear(32, r)
        ])

    def forward(self, f):
        X = torch.swapaxes(f, 1, 2)
        for layer in self.convolution:
            X = layer(X)
        X = X.flatten(1, 2)
        for layer in self.linear:
            X = layer(X)
        return X
```

#### Decoder:

```python
class Decoder(nn.Module):
    def __init__(self, din, dout, r, nlayers=5, beta=512):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(din + r, beta * dout), nn.ELU()
        ])
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(beta * dout, beta * dout))
            self.layers.append(nn.ELU())
        self.layers.append(nn.Linear(beta * dout, dout))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
```

---

### 3. Training

The training process optimizes the encoder-decoder architecture using the Mean Squared Error (MSE) loss function.

```python
def train(args):
    dataset = ShapeDataset(args.input)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    encoder = Encoder(p=dataset.V[0].shape[0], d=args.odims, r=args.ldims)
    decoder = Decoder(din=args.idims, dout=args.odims, r=args.ldims)
    crom = CROM(encoder, decoder)
    optimizer = optim.Adam(crom.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        for fb, Xb in train_loader:
            optimizer.zero_grad()
            gb = crom(fb, Xb)
            loss = criterion(gb, fb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

---

### 4. Robust Sampling

The script implements a sampling algorithm to select the most representative integration samples based on residual metrics.

```python
def robust_sampling(dataset, encoder, decoder, target_accuracy=1e-2):
    P = encoder.p
    M = [random.randint(0, P - 1)]
    r = sampling_residual(M, dataset, encoder, decoder)
    while sampling_metric(r) >= target_accuracy:
        largest_residuals = np.argsort(r.detach().numpy())[-10:]
        best_sample = largest_residuals[np.argmin([sampling_metric(r) for r in largest_residuals])]
        M.append(best_sample)
    return M
```

---

### 5. Simulation

The `simulate` function integrates the PDE solution in time using dynamics and latent space representations.

```python
def simulate(q0, decoder, X, integrate_pde, dt=0.01, T=500):
    q = q0
    for t in range(T):
        Xhat = concat(X, q)
        f = decoder(Xhat)
        f = integrate_pde(f, X, dt)
        q = gauss_newton(decoder, q, f, X)
```

---

## Modes of Operation

The script supports three modes:

1. **Train**: Train the CROM model on the dataset.
   ```bash
   python crom_simulation.py --mode train
   ```

2. **Sample**: Perform robust sampling to extract integration samples.
   ```bash
   python crom_simulation.py --mode sample
   ```

3. **Run**: Use the trained model to run PDE simulations.
   ```bash
   python crom_simulation.py --mode run
   ```

---

## References

This script is inspired by recent advancements in continuous reduced-order modeling and robust sampling techniques for efficient PDE simulation and approximation.

