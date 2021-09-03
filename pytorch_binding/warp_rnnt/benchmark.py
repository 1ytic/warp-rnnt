import argparse

import torch
import torch.nn.functional as F

from timeit import default_timer as timer


def compactTensor(xs: torch.Tensor, ys: torch.Tensor, xn: torch.Tensor, yn: torch.Tensor):

    assert xs.dim() == 4
    assert ys.dim() == 2

    N, T, Up, V = xs.size()
    assert ys.size() == (N, Up-1)
    assert xn.size(0) == N
    assert yn.size(0) == N

    _ys = torch.cat([ys[i, :yn[i]] for i in range(N)])
    _xs = [xs[i, :xn[i], :yn[i]+1, :].contiguous() for i in range(N)]
    _xs = torch.cat([x.view(-1, V) for x in _xs], dim=0)

    return _xs, _ys


def run_benchmark(loss, E, N, T, U, V, random_length=False, device="cuda", compact=False):

    torch.manual_seed(N)

    elapsed_time = 0

    for i in range(E):

        xs = torch.randn((N, T, U, V), dtype=torch.float32,
                         device=0, requires_grad=True)
        ys = torch.randint(1, V, (N, U-1), dtype=torch.int, device=0)

        if random_length:
            xn = torch.randint(T // 2, T+1, (N,), dtype=torch.int, device=0)
            yn = torch.randint(U // 2, U, (N,), dtype=torch.int, device=0)
            xn = xn + T - xn.max()
            yn = yn + U-1 - yn.max()
        else:
            xn = torch.ones((N,), dtype=torch.int, device=0) * T
            yn = torch.ones((N,), dtype=torch.int, device=0) * (U-1)

        if compact:
            xs, ys = compactTensor(xs, ys, xn, yn)

        if device == "cuda":
            xs = xs.cuda()
            ys = ys.cuda()
            xn = xn.cuda()
            yn = yn.cuda()
            torch.cuda.synchronize()  # sync before start the timer

        t = timer()

        costs = loss(xs, ys, xn, yn)

        if device == "cuda":
            torch.cuda.synchronize()  # sync before stop the timer

        elapsed_time += timer() - t

        del xs, ys, xn, yn, costs

        if device == "cuda":
            torch.cuda.empty_cache()

    return elapsed_time * 1000 / E


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Benchmark RNN-T loss implementation")
    parser.add_argument("--loss", type=str, required=True,
                        help="The target implementation")
    parser.add_argument("--device", type=str, required=False,
                        help="The target implementation", default="cuda")
    parser.add_argument("--random_length", type=bool,
                        required=False, help="The random length", default=True)

    args = parser.parse_args()

    if args.loss == "warp-rnnt":
        from warp_rnnt import rnnt_loss

        def run_loss(xs, ys, xn, yn):
            return rnnt_loss(F.log_softmax(xs, -1), ys, xn, yn, gather=False)

    elif args.loss == "warp-rnnt-gather":
        from warp_rnnt import rnnt_loss

        def run_loss(xs, ys, xn, yn):
            return rnnt_loss(F.log_softmax(xs, -1), ys, xn, yn, gather=True)

    elif args.loss == "warp-rnnt-compact":
        from warp_rnnt import rnnt_loss

        def run_loss(xs, ys, xn, yn):
            return rnnt_loss(F.log_softmax(xs, -1), ys, xn, yn, gather=False, compact=True)

    elif args.loss == "warp-rnnt-gather-compact":
        from warp_rnnt import rnnt_loss

        def run_loss(xs, ys, xn, yn):
            return rnnt_loss(F.log_softmax(xs, -1), ys, xn, yn, gather=True, compact=True)

    elif args.loss == "warprnnt_pytorch":
        from warprnnt_pytorch import rnnt_loss

        def run_loss(xs, ys, xn, yn):
            return rnnt_loss(xs, ys, xn, yn, reduction='none')

    elif args.loss == "Transducer":
        from transducer import Transducer
        fn = Transducer(blank_label=0)

        def run_loss(xs, ys, xn, yn):
            return fn.apply(F.log_softmax(xs, dim=-1), ys.view(-1), xn, yn)
    else:
        raise ValueError("Unknown RNN-T loss")
# (100, 150, 40, 28), (50, 150, 20, 5000), (10, 1500, 300, 50), (10, 400, 100, 1024)
    for E, T, U, V in [(100, 150, 40, 28), (50, 150, 20, 5000), (10, 1500, 300, 50), (10, 400, 100, 1024)]:
        for N in [1, 16, 32, 64, 128]:
            if (E, T, U, V, N) in [(10, 1500, 300, 50, 128), (10, 400, 100, 1024, 128)]:
                torch.cuda.reset_peak_memory_stats()
                continue
            print(f"T={T}\tU={U}\tV={V}\tN={N}\t", end="")
            try:
                time = run_benchmark(
                    run_loss,
                    E=E,
                    N=N,
                    T=T,
                    U=U,
                    V=V,
                    random_length=args.random_length,
                    device=args.device,
                    compact=args.loss.split('-')[-1] == 'compact'
                )
                print(
                    f"time={time:<7.2f}\tmemory={torch.cuda.max_memory_allocated(0)/1e6:.0f}")
            except RuntimeError as e:
                print(f"error={e}")
                break
            torch.cuda.reset_peak_memory_stats()
        print()
