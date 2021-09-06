#!/bin/bash

result=BenchMarkResults.txt

if [ -f $result ]; then
    echo "$result file exists."
    exit 0
fi

mkdir .benchmarktmp
cp benchmark.py .benchmarktmp/
for loss in warp-rnnt warp-rnnt-compact warp-rnnt-gather warp-rnnt-gather-compact; do
    echo $loss
    CUDA_VISIBLE_DEVICES=0 python .benchmarktmp/benchmark.py --random_length \
        --loss=$loss || exit 1
    echo ""
done > $result

/bin/rm -r .benchmarktmp/
