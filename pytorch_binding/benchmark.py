import sys

import torch
import torch.nn.functional as F

#from warp_rnnt import rnnt_loss as loss1
#from warprnnt_pytorch import rnnt_loss as loss2
from transducer.functions.transducer import Transducer

from timeit import default_timer as timer


def run_loss1(xs, ys, xn, yn):
    xs = F.log_softmax(xs, -1)
    return loss1(xs, ys, xn, yn)


def run_loss2(xs, ys, xn, yn):
    return loss2(xs, ys, xn, yn, reduction='none')


def run_loss3(xs, ys, xn, yn):
    xs = F.log_softmax(xs, -1)
    fn = Transducer(blank_label=0)
    return fn(xs, ys.view(-1), xn, yn)


def run_benchmark(loss, E, N, T, U, V):

    torch.manual_seed(N)

    elapsed_time = 0

    for i in range(E):

        xs = torch.randn((N, T, U, V), dtype=torch.float32, requires_grad=True)
        ys = torch.randint(1, V, (N, U-1), dtype=torch.int)

        #xn = torch.randint(T // 2, T+1, (N,), dtype=torch.int)
        #yn = torch.randint(U // 2, U, (N,), dtype=torch.int)
        #xn = xn + T - xn.max()
        #yn = yn + U-1 - yn.max()

        xn = torch.ones((N,), dtype=torch.int) * T
        yn = torch.ones((N,), dtype=torch.int) * (U-1)

        #xs = xs.cuda()
        #ys = ys.cuda()
        #xn = xn.cuda()
        #yn = yn.cuda()

        t = timer()

        costs = loss(xs, ys, xn, yn)

        elapsed_time += timer() - t

        del xs, ys, xn, yn, costs

        torch.cuda.empty_cache()

    elapsed_time = elapsed_time * 1000 / E

    print("%d: %.2f" % (N, elapsed_time))


def run_benchmark_safe(loss, E, N, T, U, V):
    try:
        run_benchmark(loss, E, N, T, U, V)
    except RuntimeError:
        exc_type, value, traceback = sys.exc_info()
        print(value)


for n in [1, 16, 32, 64, 128]:
    for loss in [run_loss3]:
        #run_benchmark(loss, E=100, N=n, T=150, U=40, V=28)
        #run_benchmark_safe(loss, E=10, N=n, T=150, U=20, V=5000)
        run_benchmark_safe(loss, E=10, N=n, T=1500, U=300, V=50)
