#!/bin/bash

#export nnUNet_raw=/afs/crc.nd.edu/user/y/ypeng4/data/raw_data
#export nnUNet_preprocessed=/afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data
#export nnUNet_results=/afs/crc.nd.edu/user/y/ypeng4/data/trained_models

export HOME=/afs/crc.nd.edu/user/y/ypeng4
export PYTHONPATH=/afs/crc.nd.edu/user/y/ypeng4/nnUNet:$PYTHONPATH

source /scratch365/ypeng4/software/bin/anaconda/bin/activate python310

input_raw_dir=/tmp/ypeng4/raw_data
input_pre_dir=/tmp/ypeng4/preprocessed_data

output_dir=/afs/crc.nd.edu/user/y/ypeng4/data/trained_models
output_dir_tmp=/tmp/ypeng4/data/trained_models

mkdir -p $input_raw_dir/imagenet1k
mkdir -p $input_pre_dir/imagenet1k
mkdir -p $output_dir_tmp/imagenet1k

#cp -r $HOME/data/preprocessed_data/Dataset124_ISIC2018 $input_pre_dir
echo "starting copying imagenet1k"

#cp -r $HOME/data/raw_data/imagenet1k $input_raw_dir

export nnUNet_raw=$input_raw_dir
export nnUNet_preprocessed=$input_pre_dir

#export nnUNet_results=$output_dir_tmp
export nnUNet_results=$output_dir

#/scratch365/ypeng4/software/bin/anaconda/bin/conda init
#/scratch365/ypeng4/software/bin/anaconda/bin/conda activate python310

export LD_LIBRARY_PATH=/scratch365/ypeng4/software/bin/anaconda/lib:$LD_LIBRARY_PATH
export CPATH=/scratch365/ypeng4/software/bin/anaconda/include

CUDA_VISIBLE_DEVICES=0,1,2,3 /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/dist_train.sh \
4 -c /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/configs/hvt_tiny.yml \
/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/imagenet1k \
--model hvt_tiny \
--amp \
--output /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/output/hvt \
--experiment imagenet_hvt_tiny-bs1024_12 --project imagenet --log-wandb \
--resume /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/output/hvt/imagenet_hvt_tiny-bs1024_12/last.pth.tar


#/afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer_1/classification/dist_train.sh \
#$2 -c /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer_1/classification/configs/hvt_tiny.yml \
#--log-wandb --model hvt_tiny \
#--amp \
#--nnodes=2 \
#--rdzv-id=$JOB_ID \
#--rdzv-backend=c10d \
#--rdzv-endpoint=$HOST_NODE_ADDR \
#--output /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer_1/classification/output/hvt \
#--experiment imagenet_hvt_tiny-bs256_2 --project imagenet \
#/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/imagenet1k
# --resume /afs/crc.nd.edu/user/y/ypeng4/pytorch-image-models/output/20230917-201120-hvt_tiny-224/last.pth.tar
# --resume /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer_1/classification/output/hvt/imagenet-bs1024_2/last.pth.tar

#cp -r $output_dir_tmp/Dataset122_ISIC2017 $output_dir
#rm -rf $output_dir_tmp/Dataset122_ISIC2017


/afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/dist_train.sh \
6 -c /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/configs/nat_tiny.yml \
$(pwd)/imagenet \
--output /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/output \
--experiment my_nat_tiny_702 --batch-size 170

/afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/dist_train.sh \
6 -c /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/configs/nat_tiny.yml \
$(pwd)/imagenet \
--output /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/output \
--experiment my_nat_tiny_1003 --batch-size 170