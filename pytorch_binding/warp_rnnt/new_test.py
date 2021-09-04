import torch
import numpy as np
import warp_rnnt._C as core


xs = torch.tensor([], dtype=torch.float32)
ys = torch.tensor([], dtype=torch.int)
xn = torch.tensor([], dtype=torch.int)
yn = torch.tensor([], dtype=torch.int)


def gather(log_probs: torch.Tensor, labels: torch.Tensor, frames_lengths, labels_lengths, blank=0):
    offset = 0      # offset
    index = log_probs.new_full(
        log_probs.size()[:-1] + (2,), fill_value=blank, dtype=torch.long)
    cumsumLabel = labels_lengths.cumsum(dim=0, dtype=torch.long)
    padded_labels = [torch.nn.functional.pad(
        labels[cumsumLabel[i] - labels_lengths[i]:cumsumLabel[i]], (0, 1), value=blank) for i in range(labels_lengths.size(0))]

    # FIXME: is there any faster way?
    for Ti, Ui, pad_l in zip(frames_lengths, labels_lengths, padded_labels):
        _local = Ti * (Ui + 1)
        index[offset:offset+_local, 1] = pad_l.repeat(Ti, 1).view(-1)
        offset += _local

    # print(index)
    log_probs = log_probs.gather(dim=1, index=index)
    return log_probs


def test_calls():
    n = 3
    t = 5
    u = 2
    v = 3
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

        print(xn)
        print(yn)
        print(xs.size())
        print(ys)
        _costs, _grads, _, _ = core.rnnt_loss_compact_forward(xs, ys, xn, yn)
        print(_costs)

        costs, grads, loc, blank = core.rnnt_loss_compact_forward(
            xs, ys, xn, yn, -1)
        print(costs)

        cumSum = torch.cumsum(xn * (yn+1), dim=0)
        real_grads = core.rnnt_loss_compact_backward(torch.ones_like(_costs),
                                                     _grads, cumSum.to(torch.int32), _grads, xs.size(-1), 0)

        real_grads_gather = core.rnnt_loss_compact_backward(torch.ones_like(costs),
                                                            grads, cumSum.to(torch.int32), loc, xs.size(-1), blank)
        print("Backward correctness of rnn-t compact:",
              torch.all(real_grads == _grads).item())
        print("Backward correctness of rnn-t compact gather:",
              torch.all(real_grads_gather == _grads).item())

        # print(real_grads)
        print(grads)
        print(real_grads_gather)


if __name__ == "__main__":
    test_calls()
