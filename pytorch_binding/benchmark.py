import argparse

import torch
import torch.nn.functional as F

from timeit import default_timer as timer


def run_benchmark(loss, E, N, T, U, V, random_length=False, device="cuda"):

    torch.manual_seed(N)

    elapsed_time = 0

    for i in range(E):

        xs = torch.randn((N, T, U, V), dtype=torch.float32, requires_grad=True)
        ys = torch.randint(1, V, (N, U-1), dtype=torch.int)

        if random_length:
            xn = torch.randint(T // 2, T+1, (N,), dtype=torch.int)
            yn = torch.randint(U // 2, U, (N,), dtype=torch.int)
            xn = xn + T - xn.max()
            yn = yn + U-1 - yn.max()
        else:
            xn = torch.ones((N,), dtype=torch.int) * T
            yn = torch.ones((N,), dtype=torch.int) * (U-1)

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

    parser = argparse.ArgumentParser(description="Benchmark RNN-T loss implementation")
    parser.add_argument("--loss", type=str, required=True, help="The target implementation")
    parser.add_argument("--device", type=str, required=False, help="The target implementation", default="cuda")
    parser.add_argument("--random_length", type=bool, required=False, help="The random length", default=False)

    args = parser.parse_args()

    if args.loss == "warp-rnnt":
        from warp_rnnt import rnnt_loss
        def run_loss(xs, ys, xn, yn):
            return rnnt_loss(F.log_softmax(xs, -1), ys, xn, yn, gather=False)

    elif args.loss == "warp-rnnt-gather":
        from warp_rnnt import rnnt_loss
        def run_loss(xs, ys, xn, yn):
            return rnnt_loss(F.log_softmax(xs, -1), ys, xn, yn, gather=True)

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

    for E, T, U, V in [(100, 150, 40, 28), (50, 150, 20, 5000), (10, 1500, 300, 50)]:
        for N in [1, 16, 32, 64, 128]:
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
                )
                print(f"time={time:.2f}")
            except RuntimeError as e:
                print(f"error={e}")
                break
        print()
