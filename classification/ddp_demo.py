from torch import nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import argparse
import torch
import os
from timm.models import create_model
from natt import *


import torch
from fvcore.nn import FlopCountAnalysis
from natten.flops import add_natten_handle


def get_gflops(model, img_size=224, disable_warnings=False, device='cpu'):
    flop_ctr = FlopCountAnalysis(model, torch.randn(1, 3, img_size, img_size).to(device))
    flop_ctr = add_natten_handle(flop_ctr)
    if disable_warnings:
        flop_ctr = flop_ctr.unsupported_ops_warnings(False)
    return flop_ctr.total() / 1e9


def get_mparams(model, **kwargs):
    return sum([m.numel() for m in model.parameters() if m.requires_grad]) / 1e6


def get_args_parser(parents=[], read_config=False):
    parser = argparse.ArgumentParser('NAT training script', parents=parents)

    # Misc
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')

    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')

    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--bn-tf', action='store_true', default=False,
                        help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')
    return parser

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv = nn.Conv2d(16, 64, 3, 1, 1, groups=16)
        self.num_classes = 1000
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(-1, 64)
        return self.fc(x)


def _parse_args(read_config=False):
    # setup_default_logging()
    parser = get_args_parser(read_config=read_config)
    if not read_config:
        return parser.parse_args(), parser
    parser, config_parser = parser
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args, parser


if __name__ == "__main__":
    args, _ = _parse_args(read_config=False)

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank

    args.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url)
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()

    # move model to GPU, enable channels last layout if set
    # model = Conv()

    model = create_model(
        # args.model,
        "nat_tiny",
        pretrained=False,
        num_classes=1000,
        # global_pool=args.gp,
        # bn_tf=args.bn_tf,
        # bn_momentum=args.bn_momentum,
        # bn_eps=args.bn_eps,
        # scriptable=args.torchscript,
        # checkpoint_path=args.initial_checkpoint
    )

    model.cuda()
    if args.local_rank == 0:
        with torch.cuda.device(args.local_rank):
            gmacs = get_gflops(model, img_size=224, device='cuda')
            mparams = get_mparams(model)

    if args.local_rank == 0 and not args.torchscript:
        # with torch.cuda.device(args.local_rank):
        print(f'{mparams:.3f}M Params and {gmacs:.3f}GFLOPs')

    model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)

    x = torch.rand((2, 3, 224, 224))

    for i in range(100):
        y = model(x)
        print(y.shape)

