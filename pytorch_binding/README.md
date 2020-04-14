# PyTorch bindings for CUDA-Warp RNN-Transducer

```python
def rnnt_loss(log_probs: torch.FloatTensor,
              labels: torch.IntTensor,
              frames_lengths: torch.IntTensor,
              labels_lengths: torch.IntTensor,
              average_frames: bool = False,
              reduction: Optional[AnyStr] = None,
              blank: int = 0,
              gather: bool = False) -> torch.Tensor:

    """The CUDA-Warp RNN-Transducer loss.

    Args:
        log_probs (torch.FloatTensor): Input tensor with shape (N, T, U, V)
            where N is the minibatch size, T is the maximum number of
            input frames, U is the maximum number of output labels and V is
            the vocabulary of labels (including the blank).
        labels (torch.IntTensor): Tensor with shape (N, U-1) representing the
            reference labels for all samples in the minibatch.
        frames_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        labels_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.
        average_frames (bool, optional): Specifies whether the loss of each
            sample should be divided by its number of frames.
            Default: False.
        reduction (string, optional): Specifies the type of reduction.
            Default: None.
        blank (int, optional): label used to represent the blank symbol.
            Default: 0.
        gather (bool, optional): Reduce memory consumption.
            Default: False.
    """
```

## Requirements

- C++11 or C++14 compiler (tested with GCC 5.4).
- Python: 3.5, 3.6, 3.7 (tested with version 3.6).
- [PyTorch](http://pytorch.org/) >= 1.0.0 (tested with version 1.1.0).
- [CUDA Toolkit](https://developer.nvidia.com/cuda-zone) (tested with version 10.0).



## Install

The following setup instructions compile the package from the source code locally.

### From Pypi

```bash
pip install warp_rnnt
```

### From GitHub

```bash
git clone https://github.com/1ytic/warp-rnnt
cd warp-rnnt/pytorch_binding
python setup.py install
```
