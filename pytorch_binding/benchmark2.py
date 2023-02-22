"""
This benchmark script is designed to compare
    rnnt_loss(..., gather=True) and 
    rnnt_loss(..., compact=True)

Differed to benchmark.py, this script benchmarks both the overhead
of rnn-t loss computation along with a typical joint net.
"""
import sys
import argparse
import pickle

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


class Joint(nn.Module):
    def __init__(self, hdim: int, V: int) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hdim, V)
        )
        self.requires_packing = False

    def requires_packing_(self, requires_packing: bool = True):
        self.requires_packing = requires_packing

    def forward(self, f: torch.Tensor, g: torch.Tensor, lf: torch.Tensor = None, lg: torch.Tensor = None):
        """
        f : (N, T, H), output of encoder
        g : (N, U+1, H), output of predictor
        lf: (N, ), lengths of seqs in f
        lg: (N, ), lengths of seqs in g
        """
        if self.requires_packing:
            assert lf is not None
            assert lg is not None

            H = f.size(-1)
            # x: (sum(T_i * (U_i+1)), H)
            x = torch.cat(
                [
                    (f[i, :lf[i]].unsqueeze(1) +
                     g[i, :lg[i]+1].unsqueeze(0)).view(-1, H)
                    for i in range(f.size(0))
                ],
                dim=0
            )
        else:
            # x: (N, T, U+1, H)
            x = f.unsqueeze(2) + g.unsqueeze(1)

        return self.ff(x).log_softmax(dim=-1)


def gen_data_from_real(N: int, V: int, H: int):
    with open("libri-dev.pkl", 'rb') as fib:
        xn, yn = pickle.load(fib)

    indices = torch.randint(0, xn.shape[0], (N, ))
    xn = torch.tensor(xn, dtype=torch.int, device=0)[indices]
    yn = torch.tensor(yn, dtype=torch.int, device=0)[indices]
    T = max(xn)
    U = max(yn)

    f = torch.randn(N, T, H, dtype=torch.float, device=0)
    g = torch.randn(N, U+1, H, dtype=torch.float, device=0)
    ys = torch.randint(1, V, (N, U), dtype=torch.int, device=0)

    return f, g, ys, xn, yn


def gen_data(N: int, T: int, U: int, V: int, H: int, rand_length: bool = True):

    f = torch.randn(N, T, H, dtype=torch.float, device=0)
    g = torch.randn(N, U+1, H, dtype=torch.float, device=0)
    ys = torch.randint(1, V, (N, U), dtype=torch.int, device=0)

    if rand_length:
        lf = torch.randint(T // 2, T+1, (N,), dtype=torch.int, device=0)
        lg = torch.randint(U // 2, U+1, (N,), dtype=torch.int, device=0)
        lf = lf + T - lf.max()
        lg = lg + U - lg.max()
    else:
        lf = torch.full((N, ), fill_value=T, dtype=torch.int, device=0)
        lg = torch.full((N, ), fill_value=U, dtype=torch.int, device=0)

    return f, g, ys, lf, lg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RNN-T loss implementation")
    parser.add_argument("--loss", type=str, required=True,
                        help="The target implementation")
    parser.add_argument("--random-length", action="store_true", default=False,
                        help="The random length")
    parser.add_argument("--fwd-only", action="store_true", default=False,
                        help="If True, benchmarking forward pass only, "
                        "otherwise both forward & backward are benchmarked.")

    args = parser.parse_args()
    torch.manual_seed(0)

    if args.loss == "warp-rnnt":
        from warp_rnnt import rnnt_loss

        def run_loss(xs, ys, xn, yn):
            return rnnt_loss(xs, ys, xn, yn, gather=False)

    elif args.loss == "warp-rnnt-gather":
        from warp_rnnt import rnnt_loss

        def run_loss(xs, ys, xn, yn):
            return rnnt_loss(xs, ys, xn, yn, gather=True)

    elif args.loss == "warp-rnnt-compact":
        from warp_rnnt import rnnt_loss

        def run_loss(xs, ys, xn, yn):
            ys = torch.cat([ys[i, :yn[i]] for i in range(ys.size(0))])
            return rnnt_loss(xs, ys, xn, yn, compact=True)
    else:
        raise ValueError(f"Unrecognized type of loss:{args.loss}")

    H = 512
    for E, T, U, V in [(10, 150, 40, 28), (10, 150, 20, 5000), (10, 1500, 300, 50)]:
        for N in [1, 16, 32, 64, 128]:
            joiner = Joint(H, V).cuda()
            if args.loss.split('-')[-1] == 'compact':
                joiner.requires_packing_()
            f, g, ys, xn, yn = gen_data(
                N, T, U, V, H,
                rand_length=args.random_length
            )
            if args.fwd_only:
                joiner.requires_grad_(False)
            else:
                f.requires_grad_(True)
                g.requires_grad_(True)

            print(
                f"H={H:<5} T={T:<5} U={U:<5} V={V:<5} N={N:<5} shuffle-length={args.random_length}  fwd-only={args.fwd_only}")
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function(args.loss):
                    for i in range(E):
                        xs = joiner(f, g, xn, yn)
                        loss = run_loss(xs, ys, xn, yn)
                        if not args.fwd_only:
                            (loss.mean(dim=0)).backward()
                            joiner.zero_grad()
                            f.grad = None
                            g.grad = None

            sys.stdout.write(prof.key_averages().table(
                sort_by="cpu_time_total", row_limit=10))
            sys.stdout.write(
                f"Max CUDA Mem allocated: {torch.cuda.max_memory_allocated()/1e6:.2f}MB\n\n")
            del f, g, xs, ys, xn, yn, loss
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            sys.stdout.flush()
