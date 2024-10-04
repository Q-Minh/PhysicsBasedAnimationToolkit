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
import math
from collections.abc import Callable


class ShapeDataset(Dataset):
    def __init__(self, mesh_path):
        mesh_files = [file for file in os.listdir(mesh_path) if not os.path.isdir(
            file) and (file.endswith(".mesh") or file.endswith(".msh"))]
        mesh_files = sorted(
            mesh_files, key=lambda file: int(file.split(".")[0]))
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
        return self.V[idx], self.V[0]  # 2-tuple of |#vertices|x|#dimensions|


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
        self.linear = nn.ModuleList(self.linear)
        self.convolution = nn.ModuleList(self.convolution)

    def forward(self, f: torch.Tensor):
        """

        Args:
            f (torch.Tensor): |#batch|x|#samples|x|#output dims|

        Returns:
            torch.Tensor: latent code
        """
        X = torch.swapaxes(f, 1, 2)
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
        self.layers = nn.ModuleList(self.layers)

    def forward(self, X: torch.Tensor):
        """

        Args:
            X (torch.Tensor): |#batch|x|#latent + #input dims|

        Returns:
            torch.Tensor: 
        """
        for layer in self.layers:
            X = layer(X)
        return X


class CROM(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, f: torch.Tensor, X: torch.Tensor):
        # Q = |#batch|x|#latent|
        Q = self.encoder(f)
        # Q = |#batch|x|#latent|x|#samples|
        Q = Q.unsqueeze(dim=2).expand(-1, -1, self.encoder.p)
        # Q = |#batch|x|#samples|x|#latent|
        Q = Q.swapaxes(1, 2)
        # X = |#batch|x|#samples|x|#din|
        Xhat = torch.cat((X, Q), dim=-1)
        # Xhat = |#batch|x|#samples|x|#din + #latent|
        Xhat = Xhat.flatten(0, 1)
        # Xhat = |#batch * #samples|x|#din + #latent|
        g = self.decoder(Xhat)
        return g


def line_search(alpha0: float,
                xk: np.ndarray,
                dx: np.ndarray,
                gk: np.ndarray,
                f: Callable[[np.ndarray], float],
                maxiters: int = 20,
                c: float = 1e-4,
                tau: float = 0.5):
    alphaj = alpha0
    Dfk = gk.dot(dx)
    fk = f(xk)
    for j in range(maxiters):
        fx = f(xk + alphaj*dx)
        flinear = fk + alphaj * c * Dfk
        if fx <= flinear:
            break
        alphaj = tau*alphaj
    return alphaj


def concat(X, q):
    """

    Args:
        q (torch.Tensor): |#latent dims| latent vector
        X (torch.Tensor): |#points|x|#input dims| sample positions

    Returns:
        torch.Tensor: Concatenation of q, duplicated #points times, with X
    """
    M = X.shape[0]
    # Q = |#latent|
    Q = q.unsqueeze(dim=0).expand(M, -1)
    # Q = |#samples|x|#latent|
    # X = |#samples|x|#din|
    Xhat = torch.cat((X, Q), dim=-1)
    # Xhat = |#samples|x|#din + #latent|
    return Xhat


def gauss_newton(
        decoder: Decoder,
        q0: torch.Tensor,
        f: torch.Tensor,
        X: torch.Tensor,
        maxiters: int = 10):
    """Solve non-linear least squares problem given an initial 
    iterate q0, sampled function values f taken at sample positions X, 
    and the non-linear function decoder(q).

    Args:
        decoder: g(X,q)
        q0: |#latent dims| initial latent code vector
        f: |#integration samples|x|#output dims| sampled function values at X
        X: |#integration samples|x|#input dims| sample positions
        maxiters (int, optional): Maximum Gauss-Newton iterations. Defaults to 10.

    Returns:
        torch.Tensor: The latent code q corresponding to f
    """
    def L2(q):
        g = decoder(concat(X, q))
        return torch.linalg.norm(g - f)

    def residual(X, q):
        # Xhat -> |#integration samples|x|#input dims + #latent|
        Xhat = concat(X, q)
        r = torch.linalg.norm(decoder(Xhat) - f, dim=1)
        return r

    q = q0
    for _ in range(maxiters):
        r = residual(X, q)
        # J: 1x|#latent|
        drdX, drdq = torch.autograd.functional.jacobian(residual, (X, q))
        J = drdq
        # A: |#latent|x|#latent|
        A = J.T @ J
        b = -J.T @ r
        dq = torch.linalg.solve(A, b)
        step = line_search(1., q, dq, -b, L2)
        q += dq * step
    return q


def sampling_metric(r: torch.Tensor):
    return torch.mean(r) + torch.max(r)


def sampling_residual(M: list, dataset: Dataset, encoder: Encoder, decoder: Decoder,
                      gn_max_iters: int = 100):
    # ft: |#training samples|x|#output dims|
    # XT: |#training samples|x|#input dims|
    fT, XT = dataset[-1]
    fT, XT = torch.from_numpy(fT), torch.from_numpy(XT)
    # fMT: |#integration samples|x|#output dims|
    fMT = fT[M, :]
    # XMT: |#integration samples|x|#input dims|
    XMT = XT[M, :]
    # Get latent code from previous frame to initialize network inversion
    fTm1, XTm1 = dataset[-2]
    fTm1, XTm1 = torch.from_numpy(fTm1), torch.from_numpy(XTm1)
    qTm1 = encoder(fTm1.unsqueeze(0))[0, :]
    # qM: |#latent|
    qM = gauss_newton(decoder, qTm1, fMT, XMT, maxiters=gn_max_iters)
    # Xhat: |#training samples|x|#input dims + #latent|
    Xhat = concat(XT, qM)
    # r: |#training samples|
    r = torch.linalg.norm(decoder(Xhat) - fT, dim=1)
    return r


def robust_sampling(
        dataset: Dataset,
        encoder: Encoder,
        decoder: Decoder,
        target_accuracy: float = 1e-2,
        Q: int = 10,
        gn_max_iters: int = 100):
    """Robust sampling to get runtime integration samples from CROM paper

    Args:
        dataset (Dataset): Training data set
        encoder (Encoder): e(XP) -> q
        decoder (Decoder): g(q,X) -> f
        target_accuracy (float, optional): Desired 
        PDE solution accuracy using integration samples. Defaults to 1e-2.
        Q (int, optional): Number of largest residual 
        training samples to choose from at every sampling iteration. 
        Defaults to 10.
        gn_max_iters (int, optional): Max Gauss-Newton network inversion iterations

    Returns:
        list: Subset (list) of training data sample position indices to 
        use as integration samples at inference. 
    """
    P = encoder.p
    k = random.randint(0, P - 1)
    M = []
    M.append(k)
    S = np.setdiff1d(list(range(P)), [k])
    r = sampling_residual(M, dataset, encoder, decoder, gn_max_iters)
    while sampling_metric(r) >= target_accuracy:
        print(f"# samples={len(M)}")
        largest = np.argsort(r[S].detach().numpy())[-Q:]
        mQ = [0]*Q
        for i, m in enumerate(largest):
            M.append(S[m])
            rQ = sampling_residual(M, dataset, encoder, decoder, gn_max_iters)
            mQ[i] = sampling_metric(rQ)
            M.pop()
        m = S[largest[np.argmin([mQi.item() for mQi in mQ])]]
        M.append(m)
        S = np.setdiff1d(S, [m])
        r = sampling_residual(M, dataset, encoder, decoder, gn_max_iters)
    print(f"Samples:\n{M}")
    return M


class Dynamics:
    def __init__(self, v: torch.Tensor, f: torch.Tensor, rho: torch.Tensor, Y: float = 1e6, nu: float = 0.45):
        self.v = v
        self.f = f
        self.rho = rho
        self.mu = Y / (2. * (1. + nu))
        self.llambda = (Y * nu) / ((1. + nu) * (1. - 2. * nu))

    def I1(self, S: torch.Tensor):
        return S.trace()

    def I2(self, F: torch.Tensor):
        return (F.T @ F).trace()

    def I3(self, F: torch.Tensor):
        return F.det()

    def stvk(self, F):
        I = torch.eye(F.shape[0])
        FtF = F.T @ F
        E = (FtF - I) / 2
        trE = E.trace()
        EtE = E.T @ E
        EddotE = EtE.trace()
        return self.mu*EddotE + (self.llambda / 2) * trE**2

    def neohookean(self, F):
        alpha = 1 + self.mu / self.llambda
        d = F.shape[1]
        return (self.mu / 2) * (self.I2(F) - d) + (self.llambda / 2) * (self.I3(F) - alpha)**2

    def __call__(self, xt: torch.Tensor, X: torch.Tensor, dt: float):
        # Default to BDF1 for now, but should support more integration schemes
        vt = self.v
        fext = self.f
        rho = self.rho
        din = X.shape[1]
        nsamples = X.shape[0]
        fint = torch.zeros_like(xt)
        for s in range(nsamples):
            F = torch.zeros((din, din), requires_grad=True)
            # Compute F = dx/dX
            for d in range(din):
                F[d, :] = torch.autograd.grad(
                    xt[s, d], X[s, :], create_graph=True)
            # Evaluate elastic potential Psi(F)
            Psi = self.neohookean(F)
            # Compute internal stress as dPsi/dF
            dPsidF = torch.autograd.grad(
                Psi, F, create_graph=True, retain_graph=True)
            # Compute internal forces as div(dPsi/dF)
            for d in range(din):
                fint[s, d] = torch.autograd.grad(dPsidF[:, d], X[s, :])
        x = xt + dt*vt + dt**2 * (fint + fext) / rho
        self.v = (x - xt) / dt
        return x


def simulate(
        q0: torch.Tensor,
        decoder: Decoder,
        X: torch.Tensor,
        Xfull: torch.Tensor,
        integrate_pde: Callable[[torch.Tensor, float], torch.Tensor],
        dt: float = 0.01,
        T: int = 500):
    """_summary_

    Args:
        q0 (torch.Tensor): _description_
        decoder (Decoder): _description_
        X (torch.Tensor): _description_
        Xfull (torch.Tensor): _description_
        integrate_pde (Callable[[torch.Tensor, float], torch.Tensor]): _description_
        dt (float, optional): _description_. Defaults to 0.01.
        T (int, optional): _description_. Defaults to 500.
    """
    din = X.shape[1]
    q = q0
    for t in range(T):
        Xhat = concat(X, q)
        f = decoder(Xhat)
        f = integrate_pde(f, Xhat[:, din:], dt)
        q = gauss_newton(decoder, q, f, X)
        # Obtain full-resolution PDE solution at current time step
        # ffull = decoder(concat(Xfull, q))


def train(args):
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
    X0 = dataset.V[0]
    p = X0.shape[0]

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
        losses = []
        for b, (fb, Xb) in enumerate(train):
            optimizer.zero_grad()
            gb = crom(fb, Xb)
            fb = fb.flatten(0, 1)
            L = criterion(gb, fb)
            losses.append(L.item())
            L.backward()
            optimizer.step()
        Lmean = sum(losses) / len(losses)
        writer.add_scalar("Train MSE Loss", Lmean, epoch)
        print(f"Loss={Lmean}, epoch={epoch}")
        torch.save(crom.decoder, f"{args.output}/decoder.pt")
        torch.save(crom.encoder, f"{args.output}/encoder.pt")
    writer.close()
    q0 = crom.encoder(torch.from_numpy(X0).unsqueeze(0))
    torch.save(q0, f"{args.output}/q.pt")


def sample(args):
    encoder = torch.load(f"{args.input}/encoder.pt")
    decoder = torch.load(f"{args.input}/decoder.pt")
    dataset = ShapeDataset(args.input)
    M = robust_sampling(dataset, encoder, decoder,
                        target_accuracy=args.sampling_target_accuracy,
                        Q=args.sampling_batch_samples,
                        gn_max_iters=args.sampling_max_gn_iters)
    f0, X0 = dataset[0]
    XM = X0[M, :]
    torch.save(XM, f"{args.output}/XM.pt")


def run(args):
    decoder = torch.load(f"{args.input}/decoder.pt")
    q0 = torch.load(f"{args.input}/q.pt")
    XM = torch.load(f"{args.output}/XM.pt")
    v = torch.zeros_like(XM)
    rho = torch.ones(XM.shape[0])
    g = -9.81
    f = torch.zeros_like(XM)
    f[:, -1] = rho*g
    simulate(q0, decoder, XM, None, Dynamics(v, f, rho))


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
    parser.add_argument("--sampling-max-gn-iters", help="Number of Gauss-Newton iterations in sampling mode", type=int,
                        dest="sampling_max_gn_iters", default=100)
    parser.add_argument("--sampling-target-accuracy", help="Target accuracy in CROM's smart sampling", type=float,
                        dest="sampling_target_accuracy", default=1e-2)
    parser.add_argument("--sampling-batch-samples", help="Number of samples with max residual to consider in sampling mode", type=int,
                        dest="sampling_batch_samples", default=10)
    parser.add_argument("--mode", help="One of train | sample | run", type=str,
                        dest="mode", default="train")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli()
    if args.mode == "train":
        train(args)
    if args.mode == "sample":
        sample(args)
    if args.mode == "run":
        run(args)
