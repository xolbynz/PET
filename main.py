import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt   

# GPU 3번을 쓰기 위해 CUDA_VISIBLE_DEVICES 환경변수 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model



def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # training Parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="/raid/ai_solution/TRAIN/CrowdHumanDetection_dataset_251104", type=str)

    # misc parameters
    parser.add_argument('--output_dir', default='./ckpt_251111',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--gpu', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--syn_bn', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def plot_curves(output_dir, train_loss_list, eval_epoch_list, eval_mae_list, eval_mse_list, current_epoch):
    """
    loss_curve와 eval_curve를 하나의 이미지에 위-아래(subplot)로 붙여서 저장합니다.
    """
    # figure, 2행 1열 subplot 생성 (1, 위: loss_curve / 2, 아래: eval_curve)
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # loss_curve - 위쪽
    epochs = list(range(current_epoch + 1))
    loss_vals = train_loss_list[:len(epochs)]
    axs[0].plot(epochs, loss_vals, label='Train Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Training Loss')
    axs[0].set_title('Training Loss Curve')
    axs[0].legend()
    axs[0].grid()


    if len(eval_epoch_list) > 0:
        axs[1].plot(eval_epoch_list, eval_mae_list, label='MAE')
        axs[1].plot(eval_epoch_list, eval_mse_list, label='MSE')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Metric')
        axs[1].set_title('MAE & MSE Curve')
        axs[1].legend()
        axs[1].grid()
    else:
        axs[1].text(
            0.5, 0.5, "No evaluation data yet.",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
            transform=axs[1].transAxes
        )
        axs[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_curves.png'))
    plt.close(fig)

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)
    if args.syn_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # build optimizer
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs)

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # output directory and log 
    if utils.is_main_process:
        output_dir = os.path.join("./outputs", args.dataset_file, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        run_log_name = os.path.join(output_dir, 'run_log.txt')
        with open(run_log_name, "a") as log_file:
            log_file.write('Run Log %s\n' % time.strftime("%c"))
            log_file.write("{}".format(args))
            log_file.write("parameters: {}".format(n_parameters))
    else:
        output_dir = None   # non-main process placeholder to avoid NameError
        run_log_name = None

    # --- 곡선 저장 위한 리스트 선언 ---
    train_loss_list = []
    eval_epoch_list = []
    eval_mae_list = []
    eval_mse_list = []

    # 그래프 그릴지 말지 플래그 (resume이면 그리지 말자)
    draw_curves = True
    # resume
    best_mae, best_epoch = 1e8, 0
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint['best_mae']
            # best_epoch = checkpoint['best_epoch']
        # resume 이면 그래프 그리지 않는다.
        draw_curves = False

    # training
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        t1 = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        
        # ----- train loss 추출 및 저장 -----
        train_loss_epoch = None
        if "loss" in train_stats:
            train_loss_epoch = train_stats["loss"]
        elif "loss_total" in train_stats:
            train_loss_epoch = train_stats["loss_total"]
        else:
            # fallback: 가장 loss에 가까운 key 찾기
            for k in train_stats:
                if "loss" in k:
                    train_loss_epoch = train_stats[k]
                    break
        if train_loss_epoch is None:
            train_loss_epoch = 0
        train_loss_list.append(float(train_loss_epoch))

        if utils.is_main_process:
            with open(run_log_name, "a") as log_file:
                log_file.write('\n[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        lr_scheduler.step()

        # save checkpoint
        checkpoint_paths = [output_dir / 'checkpoint.pth'] if output_dir else []
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_mae': best_mae,
            }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # write log
        if utils.is_main_process():
            with open(run_log_name, "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # evaluation
        if epoch % args.eval_freq == 0 and epoch > 0:
            t1 = time.time()
            test_stats = evaluate(model, data_loader_val, device, epoch, None)
            t2 = time.time()

            mae, mse = test_stats['mae'], test_stats['mse']
            if mae < best_mae:
                best_epoch = epoch
                best_mae = mae
            print("\n==========================")
            print("\nepoch:", epoch, "mae:", mae, "mse:", mse, "\n\nbest mae:", best_mae, "best epoch:", best_epoch)
            print("==========================\n")

            # 그래프용 저장
            if utils.is_main_process:
                eval_epoch_list.append(epoch)
                eval_mae_list.append(float(mae))
                eval_mse_list.append(float(mse))

            if utils.is_main_process():
                with open(run_log_name, "a") as log_file:
                    log_file.write("\nepoch:{}, mae:{}, mse:{}, time{}, \n\nbest mae:{}, best epoch: {}\n\n".format(
                                                epoch, mae, mse, t2 - t1, best_mae, best_epoch))
            # save best checkpoint
            if mae == best_mae and utils.is_main_process():
                src_path = output_dir / 'checkpoint.pth'
                dst_path = output_dir / 'best_checkpoint.pth'
                shutil.copyfile(src_path, dst_path)


        if draw_curves and utils.is_main_process and output_dir is not None:
            plot_curves(str(output_dir), train_loss_list, eval_epoch_list, eval_mae_list, eval_mse_list, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if draw_curves and utils.is_main_process and output_dir is not None:
        plot_curves(str(output_dir), train_loss_list, eval_epoch_list, eval_mae_list, eval_mse_list, epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
