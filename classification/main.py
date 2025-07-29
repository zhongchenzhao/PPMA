# modified from https://github.com/facebookresearch/deit
"""
torchrun --nproc_per_node=8 --master_port=29501 main.py --warmup_epochs 5 --model PPMAViT_T --data_path /mnt/nvme_data/zzc/datasets/ImageNet1K --num_workers 16 --batch_size 128 --drop_path 0.05 --epoch 300 --dist_eval --output_dir ./exp/PPMAViT_T_202503200000
"""
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
import utils
import os

# import models
from models.PPMA import PPMAViT_T, PPMAViT_S, PPMAViT_B
from models.PPMA_wo_RoPE import PPMAViT_wo_RoPE_T
from models.PPMA_only_ppmva import PPMVAViT_T
from models.RMT import RMT_T, RMT_S, RMT_B

from utils_common import save_codes, save_python_command
from utils_common import create_logger
import torch.distributed as torch_dist

archs = {
    'PPMAViT_T': PPMAViT_T,
    'PPMAViT_S': PPMAViT_S,
    'PPMAViT_B': PPMAViT_B,
    'RMT_T': RMT_T,
    'RMT_S': RMT_S,
    'RMT_B': RMT_B,
}


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--early_conv', action='store_true')
    parser.add_argument('--conv_pos', action='store_true')
    parser.add_argument('--use_ortho', action='store_true')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of models to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--num_classes', default=1000, type=int, help='num_classes')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--no_model_ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.99996, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated_aug', action='store_true')
    parser.add_argument('--no_repeated_aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher_model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher models to train (default: "regnety_160"')
    parser.add_argument('--teacher_path', type=str, default='')
    parser.add_argument('--distillation_type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation_alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation_tau', default=1.0, type=float, help="")
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat_category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--mesa', action='store_true', help='tricks to prevent overfitting, following MLLA')
    parser.add_argument('--mesa_value', type=float, default=1.0, help='tricks to prevent overfitting, following MLLA')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    logger = create_logger(output_dir=args.output_dir, dist_rank=torch_dist.get_rank(), name=f"{args.model}")
    if torch_dist.get_rank() == 0:
        path = os.path.join(args.output_dir, "config.json")
        with open(path, "w") as f:
            f.write(json.dumps(vars(args)))
        logger.info(f"Full config saved to {path}")
    save_codes(args.output_dir)
    save_python_command(args.output_dir)

    logger.info(json.dumps(vars(args)))

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.info(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size / 2),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                         label_smoothing=args.smoothing, num_classes=args.nb_classes)

    logger.info(f"Creating models: {args.model}")
    model = archs[args.model](args)

    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {str(n_parameters)}")


    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['models'], strict=True)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    model_ema = None
    if args.model_ema:  # True
        # Important to create EMA models after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema_decay = args.model_ema_decay ** (args.batch_size * utils.get_world_size() / 512.0)
        # args.model_ema_decay=0.99996
        model_ema = ModelEma(
            model,
            decay=model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        logger.info("Using EMA with decay = %.8f" % model_ema_decay)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params:{n_parameters}')

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # logger.info('linear_scaled_lr ', linear_scaled_lr)            # 0.001
    # utils.get_world_size() = 8
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    # lr_scheduler = build_scheduler(args, optimizer, len(data_loader_train))

    # criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        logger.info(f"Creating teacher models: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['models'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    max_accuracy = 0.0
    output_dir = Path(args.output_dir)
    # ipdb.set_trace()
    if args.resume == '':
        tmp = f"{args.output_dir}/checkpoint.pth"
        if os.path.exists(tmp):
            args.resume = tmp
    flag = os.path.exists(args.resume)
    if args.resume and flag:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['models'], strict=False)
        if args.model_ema:
            model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=True)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    if args.eval:
        # if args.resume == '':
        tmp = f"{args.output_dir}/downtarget.pth"
        if os.path.exists(tmp):
            args.resume = tmp
        flag = os.path.exists(args.resume)
        if args.resume and flag:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['models'], strict=False)
            if args.model_ema:
                model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=True)
        test_stats = evaluate(data_loader_val, model, device, logger=logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.3f}%")
        if model_ema is not None:
            test_stats_ema = evaluate(data_loader_val, model_ema.ema, device)
            logger.info(
                f"Accuracy of the network_ema on the {len(dataset_val)} test images: {test_stats_ema['acc1']:.3f}%")
        return

    if args.mesa:
        logger.info(
            f"Noting: using mesa following MLLA, mesa_value={args.mesa_value} when epoch >= {int(0.25 * args.epochs)} !!!")
    else:
        logger.info(f"Noting: without mesa !!!")

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            lr_scheduler, model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            logger=logger,
            mesa=args.mesa_value if (epoch >= int(0.25 * args.epochs) and args.mesa) else -1.0,
        )

        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                if args.model_ema:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'max_accuracy': max_accuracy,
                        'max_accuracy_ema': max_accuracy_ema,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'models': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'max_accuracy': max_accuracy,
                        'max_accuracy_ema': max_accuracy_ema,
                    }, checkpoint_path)
            if epoch % 20 == 0:
                if args.model_ema:
                    utils.save_on_master({
                        'models': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'max_accuracy': max_accuracy,
                        'max_accuracy_ema': max_accuracy_ema,
                    }, f"{args.output_dir}/backup{epoch}.pth")
                else:
                    utils.save_on_master({
                        'models': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'max_accuracy': max_accuracy,
                        'max_accuracy_ema': max_accuracy_ema,
                    }, f"{args.output_dir}/backup{epoch}.pth")

        test_stats = evaluate(data_loader_val, model, device, logger=logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.3f}%")
        if model_ema is not None:
            test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, logger=logger)
            logger.info(
                f"Accuracy of the network_ema on the {len(dataset_val)} test images: {test_stats_ema['acc1']:.3f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        if model_ema is not None:
            max_accuracy_ema = max(max_accuracy_ema, test_stats_ema['acc1'])

        logger.info('Max accuracy: {:.3f}%'.format(max_accuracy))
        logger.info('Max accuracy (ema): {:.3f}%'.format(max_accuracy_ema))
        if max_accuracy_ema == test_stats_ema['acc1']:
            if args.model_ema:
                utils.save_on_master({
                    'models': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'max_accuracy': max_accuracy,
                    'max_accuracy_ema': max_accuracy_ema,
                }, f"{args.output_dir}/best.pth")
            else:
                utils.save_on_master({
                    'models': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'max_accuracy': max_accuracy,
                    'max_accuracy_ema': max_accuracy_ema,
                }, f"{args.output_dir}/best.pth")
        train_f = {'train_{}'.format(k): v for k, v in train_stats.items()}
        test_f = {'test_{}'.format(k): v for k, v in test_stats.items()}
        if model_ema is not None:
            test_ema_f = {'test_ema_{}'.format(k): v for k, v in test_stats_ema.items()}
        log_stats = dict({'epoch': epoch,
                          'n_parameters': n_parameters}, **train_f, **test_f, **test_ema_f)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}, save path: {}.'.format(total_time_str, args.output_dir))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
