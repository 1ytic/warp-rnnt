![PyPI](https://img.shields.io/pypi/v/warp-rnnt.svg)
[![Downloads](https://pepy.tech/badge/warp-rnnt)](https://pepy.tech/project/warp-rnnt)

# CUDA-Warp RNN-Transducer
A GPU implementation of RNN Transducer (Graves [2012](https://arxiv.org/abs/1211.3711), [2013](https://arxiv.org/abs/1303.5778)).
This code is ported from the [reference implementation](https://github.com/awni/transducer/blob/master/ref_transduce.py) (by Awni Hannun)
and fully utilizes the CUDA warp mechanism.

The main bottleneck in the loss is a forward/backward pass, which based on the dynamic programming algorithm.
In particular, there is a nested loop to populate a lattice with shape (T, U),
and each value in this lattice depend on the two previous cells from each dimension (e.g. [forward pass](https://github.com/awni/transducer/blob/6b37e98c21551c7ed2181e2f526053bae8ae94d2/ref_transduce.py#L56)).

CUDA executes threads in groups of 32 parallel threads called [warps](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture).
Full efficiency is realized when all 32 threads of a warp agree on their execution path.
This is exactly what is used to optimize the RNN Transducer. The lattice is split into warps in the T dimension.
In each warp, variables between threads exchanged using a fast operations.
As soon as the current warp fills the last value, the next two warps (t+32, u) and (t, u+1) are start running. 
A schematic procedure for the forward pass is shown in the figure below, where T - number of frames, U - number of labels, W - warp size.
The similar procedure for the backward pass runs in parallel.

![](lattice.gif)


## Performance
NVIDIA Profiler shows advantage of the _warp_ implementation over the _non-warp_ implementation.

This warp implementation:
![](warp-rnnt.nvvp.png)

Non-warp implementation [warp-transducer](https://github.com/HawkAaron/warp-transducer):
![](warp-transducer.nvvp.png)

Unfortunately, in practice this advantage disappears because the memory operations takes much longer. Especially if you synchronize memory on each iteration.

|                         |    warp_rnnt (gather=False)    |    warp_rnnt (gather=True)    | [warprnnt_pytorch](https://github.com/HawkAaron/warp-transducer/tree/master/pytorch_binding) | [transducer (CPU)](https://github.com/awni/transducer) |
| :---------------------- | ------------------: | ------------------: | ------------------: | ------------------: |
|  **T=150, U=40, V=28**  | 
|         N=1             |       0.50 ms       |       0.54 ms       |       0.63 ms       |       1.28 ms       |
|         N=16            |       1.79 ms       |       1.72 ms       |       1.85 ms       |       6.15 ms       |
|         N=32            |       3.09 ms       |       2.94 ms       |       2.97 ms       |      12.72 ms       |
|         N=64            |       5.83 ms       |       5.54 ms       |       5.23 ms       |      23.73 ms       |
|         N=128           |      11.30 ms       |      10.74 ms       |       9.99 ms       |      47.93 ms       |
| **T=150, U=20, V=5000** |
|         N=1             |       0.95 ms       |       0.80 ms       |       1.74 ms       |      21.18 ms       |
|         N=16            |       8.74 ms       |       6.24 ms       |      16.20 ms       |     240.11 ms       |
|         N=32            |      17.26 ms       |      12.35 ms       |      31.64 ms       |     490.66 ms       |
|         N=64            |    out-of-memory    |    out-of-memory    |    out-of-memory    |     944.73 ms       |
|         N=128           |    out-of-memory    |    out-of-memory    |    out-of-memory    |    1894.93 ms       |
| **T=1500, U=300, V=50** |
|         N=1             |       5.89 ms       |       4.99 ms       |      10.02 ms       |     121.82 ms       |
|         N=16            |      95.46 ms       |      78.88 ms       |      76.66 ms       |     732.50 ms       |
|         N=32            |    out-of-memory    |     157.86 ms       |     165.38 ms       |    1448.54 ms       |
|         N=64            |    out-of-memory    |    out-of-memory    |     out-of-memory   |    2767.59 ms       |

[Benchmarked](pytorch_binding/benchmark.py) on a GeForce RTX 2070 Super GPU, Intel i7-10875H CPU @ 2.30GHz.

## Note

- This implementation assumes that the input is log_softmax.

- In addition to alphas/betas arrays, counts array is allocated with shape (N, U * 2), which is used as a scheduling mechanism.

- [core_gather.cu](core_gather.cu) is a memory-efficient version that expects log_probs with the shape (N, T, U, 2) only for blank and labels values. It shows excellent performance with a large vocabulary.

- Do not expect that this implementation will greatly reduce the training time of RNN Transducer model. Probably, the main bottleneck will be a trainable joint network with an output (N, T, U, V).

- Also, there is a restricted version, called [Recurrent Neural Aligner](https://github.com/1ytic/warp-rna), with assumption that the length of input sequence is equal to or greater than the length of target sequence.


## Install
There are two bindings for the core algorithm:
- [pytorch_binding](pytorch_binding)
- [tensorflow_binding](tensorflow_binding)


## Reference
- Awni Hannun [transducer](https://github.com/awni/transducer)

- Mingkun Huang [warp-transducer](https://github.com/HawkAaron/warp-transducer)
