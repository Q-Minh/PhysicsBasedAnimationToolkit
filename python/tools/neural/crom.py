import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch


class Encoder(nn.Module):

    def __init__(self, p: int, d: int, r: int, conv_kernel_size: int = 6, conv_stride_size: int = 4):
        super().__init__()
        self.p = p
        self.d = d
        self.r = r
        self.k = conv_kernel_size
        self.s = conv_stride_size

        self.layers = []
        L = p
        # See CROM paper for the value 32
        while L > 32:
            # See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            Lout = (L - (self.k - 1) - 1) / self.s + 1
            if L <= 32:
                break

            L = Lout
            self.layers.append(nn.Conv1d(self.d, self.d, self.k, self.s))
            self.layers.append(nn.ELU())

        self.layers.append(nn.Linear(L * self.d, 32))
        self.layers.append(nn.ELU())
        self.layers.append(nn.Linear(32, self.r))

    def forward(self, X: torch.Tensor):
        for layer in self.layers:
            X = layer(X)
        return X

    @property
    def p(self):
        return self.p

    @property
    def d(self):
        return self.d

    @property
    def r(self):
        return self.r

    @property
    def k(self):
        return self.k

    @property
    def s(self):
        return self.s


class Decoder(nn.Module):

    def __init__(self, din: int, dout: int, r: int, nlayers: int = 5, beta: int = 512):
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

    @property
    def din(self):
        return self.din

    @property
    def dout(self):
        return self.dout

    @property
    def r(self):
        return self.r

    @property
    def beta(self):
        return self.beta

    @property
    def nlayers(self):
        return len(self.layers)


class CROM(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = nn.MSELoss()

    def forward(self, X: torch.Tensor):
        din = self.decoder.din
        dout = self.decoder.dout
        T = int(X.shape[1] / (din + dout))
        finds = [din + t*(din+dout) + j
                 for j in range(dout)
                 for t in range(T)]
        Xinds = [t*(din+dout) + j
                 for j in range(din)
                 for t in range(T)]
        # X is #samples x |decoder.din + decoder.dout|
        Q = self.encoder(X[:, finds])
        loss = []
        for sample in range(self.encoder.p):
            # WARNING: Check that this reshape is column order!
            Xt = X[sample, Xinds].reshape(din, T)
            Xhat = torch.cat((Xt, Q), dim=-1)
            gt = self.decoder(Xhat)
            ft = X[sample, finds].reshape(dout, T)
            loss.append(self.loss(gt, ft))

        mu = sum(loss) / len(loss)
        return mu


if __name__ == "__main__":
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

    # TODO: Construct dataset from simulation data!
    X = []
    p = X.shape[0]
    # TODO: Compute number of data points from data set
    ndata = X.shape[1]
    bsize = args.batch_size

    encoder = Encoder(p, args.odims, args.ldims,
                      conv_kernel_size=args.encoder_conv_kernel_size, conv_stride_size=args.encoder_conv_stride_size)
    decoder = Decoder(args.idims, args.odims, args.ldims,
                      nlayers=args.glayers, beta=args.beta)
    crom = CROM(encoder, decoder)
    optimizer = optim.Adam(crom.parameters(), args.learning_rate)

    for epoch in range(args.epochs):
        for b in range(0, ndata - bsize, bsize):
            Xb = X[:,b:(b+bsize)]
            L = crom(Xb)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()