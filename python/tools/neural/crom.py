import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import meshio
import os
import random


class ShapeDataset(Dataset):
    def __init__(self, mesh_path):
        mesh_files = [file for file in os.listdir(mesh_path) if not os.path.isdir(
            file) and (file.endswith(".mesh") or file.endswith(".msh"))]
        mesh_files.sort()
        meshes = [meshio.read(os.path.join(mesh_path, file))
                  for file in mesh_files]
        V, C = [mesh.points for mesh in meshes], [
            mesh.cells_dict["tetra"] for mesh in meshes]
        # have_valid_topology = len(
        #     [Vi for Vi in V if Vi.shape == V[0].shape]) == 0
        # assert (have_valid_topology)
        self.V = V

    def __len__(self):
        return len(self.V)

    def __getitem__(self, idx):
        return self.V[idx].T, self.V[idx].T


class Encoder(nn.Module):

    def __init__(self, p: int, d: int, r: int, conv_kernel_size: int = 6, conv_stride_size: int = 4):
        """

        Args:
            p (int): Number of samples of the PDE solution
            d (int): PDE solution's output dimensions
            r (int): Latent dimensions
            conv_kernel_size (int, optional): 1D convolution kernel size. Defaults to 6.
            conv_stride_size (int, optional): 1D convolution stride size. Defaults to 4.
        """
        super().__init__()
        self.p = p
        self.d = d
        self.r = r
        self.k = conv_kernel_size
        self.s = conv_stride_size

        self.convolution = []
        self.linear = []
        L = p
        # See CROM paper for the value 32
        while L > 32:
            # See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            Lout = (L - (self.k - 1) - 1) / self.s + 1
            if L <= 32:
                break

            L = int(Lout)
            self.convolution.append(nn.Conv1d(self.d, self.d, self.k, self.s))
            self.convolution.append(nn.ELU())

        self.linear.append(nn.Linear(L * self.d, 32))
        self.linear.append(nn.ELU())
        self.linear.append(nn.Linear(32, self.r))

    def forward(self, X: torch.Tensor):
        for layer in self.convolution:
            X = layer(X)
        X = X.flatten(1, 2)
        for layer in self.linear:
            X = layer(X)
        return X


class Decoder(nn.Module):

    def __init__(self, din: int, dout: int, r: int, nlayers: int = 5, beta: int = 512):
        """

        Args:
            din (int): PDE solution's input dimensions
            dout (int): PDE solution's output dimensions
            r (int): Latent dimensions
            nlayers (int, optional): Number of MLP layers. Defaults to 5.
            beta (int, optional): Learning capacity. Defaults to 512.
        """
        super().__init__()

        self.din = din
        self.dout = dout
        self.r = r
        self.beta = beta

        self.layers = []
        self.layers.append(nn.Linear(self.din + r, beta*self.dout))
        self.layers.append(nn.ELU())
        for l in range(nlayers - 2):
            self.layers.append(nn.Linear(beta*self.dout, beta*self.dout))
            self.layers.append(nn.ELU())
        self.layers.append(nn.Linear(beta*self.dout, self.dout))

    def forward(self, X: torch.Tensor):
        for layer in self.layers:
            X = layer(X)
        return X


class CROM(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, f: torch.Tensor, X: torch.Tensor):
        Q = self.encoder(f)
        # Q = |#batch|x|#latent|
        Q = Q[:, :, torch.newaxis].expand(-1, -1, self.encoder.p)
        Q = Q.swapaxes(1, 2)
        X = X.swapaxes(1, 2)
        # Q = |#batch|x|#samples|x|#latent|
        # X = |#batch|x|#samples|x|#din|
        Xhat = torch.cat((X, Q), dim=-1)
        # Xhat = |#batch|x|#samples|x|#din + #latent|
        Xhat = Xhat.flatten(0, 1)
        # Xhat = |#batch * #samples|x|#din + #latent|
        g = self.decoder(Xhat)
        return g


def parse_cli():
    parser = argparse.ArgumentParser(
        prog="Continuous Reduced Order Model",
    )
    parser.add_argument("-i", "--input", help="Path to input data", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output", type=str,
                        dest="output", default=".")
    parser.add_argument("-d", "--output-dimensions", help="PDE solution dimensionality", type=int,
                        dest="odims", default=1)
    parser.add_argument("-m", "--input-dimensions", help="PDE solution embedding dimensionality", type=int,
                        dest="idims", default=3)
    parser.add_argument("-r", "--latent-dimensions", help="Latent PDE solution dimensionality", type=int,
                        dest="ldims", default=64)
    parser.add_argument("--manifold-hidden-layers", help="Number of hidden layers for the manifold parameterization", type=int,
                        dest="glayers", default=5)
    parser.add_argument("-b", "--beta", help="(Width / d) hidden layer dimensions", type=int,
                        dest="beta", default=512)
    parser.add_argument("-k", "--encoder-conv-kernel-size", help="Kernel size for the encoder's 1D convolution", type=int,
                        dest="encoder_conv_kernel_size", default=6)
    parser.add_argument("-s", "--encoder-conv-stride-size", help="Stride size for the encoder's 1D convolution", type=int,
                        dest="encoder_conv_stride_size", default=4)
    parser.add_argument("--epochs", help="Number of training epochs", type=int,
                        dest="epochs", default=100)
    parser.add_argument("--learning-rate", help="Number of training epochs", type=float,
                        dest="learning_rate", default=1e-4)
    parser.add_argument("--batch-size", help="Training batch size", type=int,
                        dest="batch_size", default=32)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli()

    # Setup reproducible training
    seed = 0
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":2048:8"

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    worker_init_fn = seed_worker
    generator = torch.Generator().manual_seed(seed)

    # Load data
    dataset = ShapeDataset(args.input)
    # train, test = random_split(dataset, [0.8, 0.2], generator=generator)
    # train = DataLoader(train, batch_size=args.batch_size,
    #                    shuffle=True, worker_init_fn=worker_init_fn)
    train = DataLoader(dataset, batch_size=args.batch_size,
                       shuffle=True, worker_init_fn=worker_init_fn)
    p = dataset.V[0].shape[0]

    # Build encoder/decoder joint training network
    encoder = Encoder(p, args.odims, args.ldims,
                      conv_kernel_size=args.encoder_conv_kernel_size,
                      conv_stride_size=args.encoder_conv_stride_size)
    decoder = Decoder(args.idims, args.odims, args.ldims,
                      nlayers=args.glayers, beta=args.beta)
    crom = CROM(encoder, decoder)
    optimizer = optim.Adam(
        [{"params": encoder.parameters(), "lr": args.learning_rate},
         {"params": decoder.parameters(), "lr": args.learning_rate}])
    criterion = nn.MSELoss()

    # Train encoder/decoder via simple L2 reconstruction error
    writer = SummaryWriter(args.output)
    for epoch in range(args.epochs):
        for b, (fb, Xb) in enumerate(train):
            optimizer.zero_grad()
            gb = crom(fb, Xb)
            fb = fb.swapaxes(1, 2).flatten(0, 1)
            L = criterion(gb, fb)
            L.backward()
            optimizer.step()
            writer.add_scalar("Train MSE Loss", L, epoch + b / len(train))
    writer.close()
