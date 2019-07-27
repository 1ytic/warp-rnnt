import torch
import warp_rnnt._C as warp_rnnt_core


class _WRNNT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, labels, frames_lengths, labels_lengths, blank=0):
        costs, ctx.grads = warp_rnnt_core.rnnt_loss(
            xs=log_probs, ys=labels,
            xn=frames_lengths, yn=labels_lengths,
            blank=blank,
        )
        return costs

    @staticmethod
    def backward(ctx, grads_output):
        grads_output = grads_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul_(grads_output), None, None, None, None, None


def rnnt_loss(
        log_probs,  # type: torch.FloatTensor
        labels,  # type: torch.IntTensor
        frames_lengths,  # type: torch.IntTensor
        labels_lengths,  # type: torch.IntTensor
        average_frames=False,  # type: bool
        reduction=None,  # type: Optional[AnyStr]
        blank=0,  # type: int
):
    """The CUDA-Warp RNN-Transducer loss.

    Args:
      log_probs (torch.Tensor): Input tensor (float) with shape
        (T, N, U, V) where T is the maximum number of input frames, N is the
        minibatch size, U is the maximum number of output labels and V is
        the vocabulary of labels (including the blank).
      labels (torch.IntTensor): Tensor with shape (N, U-1) representing the
        reference labels for all samples in the minibatch.
      frames_lengths (torch.IntTensor): Tensor with shape (N,) representing the
        number of frames for each sample in the minibatch.
      labels_lengths (torch.IntTensor): Tensor with shape (N,) representing the
        length of the transcription for each sample in the minibatch.
      average_frames (bool, optional): Specifies whether the loss of each
        sample should be divided by its number of frames. Default: ``False''.
      reduction (string, optional): Specifies the type of reduction.
        Default: None.
      blank (int, optional): label used to represent the blank symbol.
        Default: 0.
    """
    # type: (...) -> torch.Tensor

    assert average_frames is None or isinstance(average_frames, bool)
    assert reduction is None or reduction in ("none", "mean", "sum")
    assert isinstance(blank, int)

    assert not labels.requires_grad, "labels does not require gradients"
    assert not frames_lengths.requires_grad, "frames_lengths does not require gradients"
    assert not labels_lengths.requires_grad, "labels_lengths does not require gradients"

    costs = _WRNNT.apply(log_probs, labels, frames_lengths, labels_lengths, blank)

    if average_frames:
        costs = costs / frames_lengths.to(log_probs)

    if reduction == "sum":
        return costs.sum()
    elif reduction == "mean":
        return costs.mean()
    return costs
