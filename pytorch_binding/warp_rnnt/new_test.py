import torch
import numpy as np
import warp_rnnt._C as core
from warp_rnnt import rnnt_loss


xs = torch.tensor([], dtype=torch.float32)
ys = torch.tensor([], dtype=torch.int)
xn = torch.tensor([], dtype=torch.int)
yn = torch.tensor([], dtype=torch.int)


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


def reverseCompact(xs: torch.Tensor, ys: torch.Tensor, xn: torch.Tensor, yn: torch.Tensor):

    N, T, U, V = xn.size(0), xn.max(), yn.max(), xs.size(-1)
    _xs = xs.new_zeros((N, T, U+1, V))
    _ys = ys.new_zeros((N, U))

    offset = 0
    offset_y = 0
    for n in range(N):
        Ti, Uip = xn[n], yn[n]+1
        _xs[n, :Ti, :Uip, :] = xs[offset:offset+Ti*Uip, :].view(Ti, Uip, V)

        _ys[n, :Uip-1] = ys[offset_y:offset_y+Uip-1].view(-1)
        offset += Ti*Uip
        offset_y += Uip-1

    return _xs, _ys


def test_calls():
    n = 20
    t = 32
    u = 16
    v = 3
    cnt = 0
    for i in range(1):
        torch.manual_seed(i)
        xn = torch.tensor([t] * n, dtype=torch.int, device=0)
        yn = torch.randint(1, u, (n,), dtype=torch.int, device=0)
        ys = torch.randint(1, v, (yn.sum(), ), dtype=torch.int, device=0)
        xs = torch.randn(((xn*(yn+1)).sum(), v), dtype=torch.float32,
                         device=0).log_softmax(dim=-1)

        # xn = torch.tensor([2], dtype=torch.int, device=0)
        # yn = torch.tensor([1], dtype=torch.int, device=0)
        # xs = torch.tensor([[-1.6230, -0.8267, -1.0073],
        #                    [-1.3808, -2.7746, -0.3765],
        #                    [-2.2364, -0.6881, -0.9399],
        #                    [-0.5337, -1.5759, -1.5763]], dtype=torch.float32, device=0)
        # ys = torch.tensor([1], dtype=torch.int, device=0)

        cumSum = torch.cumsum(xn * (yn+1), dim=0)
        _costs, _grads, _, _ = core.rnnt_loss_compact_forward(xs, ys, xn, yn)
        real_grads = core.rnnt_loss_compact_backward(torch.ones_like(_costs),
                                                     _grads, cumSum.to(torch.int32), _grads, xs.size(-1), 0)

        costs, grads, loc, blank = core.rnnt_loss_compact_forward(
            xs, ys, xn, yn, -1)
        real_grads_gather = core.rnnt_loss_compact_backward(torch.ones_like(costs),
                                                            grads, cumSum.to(torch.int32), loc, xs.size(-1), blank)

        if not torch.all(real_grads == real_grads_gather):
            print(xn)
            print(yn)
            print(xs.size())
            print(ys)
            print(_costs)
            print(costs)
            print(_grads)
            print(real_grads)
            print(real_grads_gather)
            break

        cnt += torch.all(real_grads == real_grads_gather)

        # xs.requires_grad = False
        # xs, ys = reverseCompact(xs, ys, xn, yn)
        xs.requires_grad = True
        torch.autograd.gradcheck(
            rnnt_loss, (xs, ys, xn, yn, False, 'mean', 0, False, 0.0, True))

    print("Gather mode produces {} same results as non-gather one.".format(cnt))


def test_compute():

    NTest = 3

    for seed in range(NTest):
        torch.manual_seed(seed)
        N = torch.randint(1, 20, (1,)).item()
        T = torch.randint(5, 512, (1,)).item()
        U = torch.randint(1, 512, (1,)).item()
        V = torch.randint(3, 128, (1,)).item()

        xs = torch.randn((N, T, U, V), dtype=torch.float32,
                         device=0).log_softmax(dim=-1)
        ys = torch.randint(1, V, (N, U-1), dtype=torch.int, device=0)
        xn = torch.randint(T // 2, T+1, (N,), dtype=torch.int, device=0)
        yn = torch.randint(U // 2, U, (N,), dtype=torch.int, device=0)
        xn = xn + T - xn.max()
        yn = yn + U-1 - yn.max()

        # print("{0} Test loaded sample {0}".format("="*10))
        # checkpoint = torch.load(
        #     'CheckForTestRNNT.pt', map_location='cuda:0')
        # xs, ys, xn, yn = tuple(checkpoint.values())
        # xs.requires_grad = False
        # ys.requires_grad = False
        # xs, ys = reverseCompact(xs, ys, xn, yn)

        ys = ys.to(dtype=torch.int)
        xn, yn = xn.to(dtype=torch.int, device=0), yn.to(
            dtype=torch.int, device=0)
        print("xs size: ", xs.size())
        print("ys size: ", ys.size())
        print("lx size: ", xn.size())
        print("ly size: ", yn.size())
        xs.requires_grad = True

        m_cost = rnnt_loss(xs, ys, xn, yn, gather=False, compact=False)
        m_cost.sum().backward()
        m_grad = xs.grad.data.detach()
        xs.grad = None

        _xs, _ys = compactTensor(xs, ys, xn, yn)
        t_cost = rnnt_loss(_xs, _ys, xn, yn, gather=True, compact=True)
        t_cost.sum().backward()
        t_grad = xs.grad.data.detach()

        print("backward diff 1-order norm: {:.4e}".format(
            torch.sum(torch.abs(m_grad - t_grad)).item()))

        print("correctness: forward | backward : {} | {}\n".format(
            torch.all(m_cost == t_cost).item(), torch.all(m_grad == t_grad).item()))


if __name__ == "__main__":
    # test_compute()
    test_calls()
