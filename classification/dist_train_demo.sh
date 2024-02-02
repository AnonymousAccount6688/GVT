#!/bin/bash
# Copyright Ross Wightman (https://github.com/rwightman)

NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=11223 /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/ddp_demo.py "$@"
