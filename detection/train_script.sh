/afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/detection/dist_train.sh /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/detection/configs/nat/mask_rcnn_nat_tiny_3x_coco.py 6 \
--cfg-options data.samples_per_gpu=3 data.workers_per_gpu=3 \
model.pretrained=/afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer/classification/output/imagenet_nat_tiny_85_keep/last.pth.tar model.backbone.use_checkpoint=True


/afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer_new/Neighborhood-Attention-Transformer/detection/dist_train.sh /afs/crc.nd.edu/user/y/ypeng4/Neighborhood-Attention-Transformer_new/Neighborhood-Attention-Transformer/detection/configs/nat/mask_rcnn_nat_tiny_3x_coco.py 4 --cfg-options data.samples_per_gpu=4 data.workers_per_gpu=4

