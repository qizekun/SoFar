import time
import math
import torch
import torch.nn as nn
from orientation.tools import builder
from orientation.utils.logger import *
from orientation.utils import dist_utils
from orientation.utils.AverageMeter import AverageMeter


class AccMetric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'AccMetric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def calculate_acc(pred_direction, direction, k=30):
    cos = torch.cosine_similarity(pred_direction, direction, dim=-1)
    acc = (cos > torch.cos(torch.tensor(k / 180 * math.pi))).sum() / cos.size(0) * 100.
    return acc


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)

    if args.local_rank == 0:
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                num_params = param.numel() / 1e6
                print(f"Layer: {name} | Number of parameters: {num_params:.3f} M")

        total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad) / 1e6
        print(f"Total number of parameters: {total_params:.3f} M")

    # parameter setting
    start_epoch = 0
    best_metrics = AccMetric(0.)
    metrics = AccMetric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = AccMetric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    npoints = config.npoints
    print_log("Downsample to %d points" % npoints, logger=logger)
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, (pcs, direction, instruction) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)

            if args.use_gpu:
                pcs = pcs.to(args.local_rank)
                direction = direction.to(args.local_rank)
            pred_direction = base_model(pcs, instruction).squeeze()  # B, 3

            if len(direction.shape) == 3:
                pred_expanded = pred_direction.unsqueeze(1).expand(-1, 10, -1)
                distances = torch.norm(direction - pred_expanded, dim=2)
                min_dist_indices = distances.argmin(dim=1)
                direction = torch.gather(direction, 1, min_dist_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 3))
                direction = direction.squeeze(1)

            acc = calculate_acc(pred_direction, direction, k=45)

            loss = 1 - torch.nn.functional.cosine_similarity(pred_direction, direction).mean()
            loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=logger)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Loss = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=logger)
            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
                print_log("----------------------------------------------------------------------", logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=None):
    base_model.eval()  # set model to eval mode

    pred_direction_list = []
    direction_list = []
    with torch.no_grad():
        for idx, (pcs, direction, instruction) in enumerate(test_dataloader):

            if args.use_gpu:
                pcs = pcs.to(args.local_rank)
                direction = direction.to(args.local_rank)

            pred_direction = base_model(pcs, instruction).squeeze()

            pred_direction_list.append(pred_direction)
            direction_list.append(direction)

        pred_direction = torch.cat(pred_direction_list, dim=0)
        direction = torch.cat(direction_list, dim=0)

        if args.distributed:
            pred_direction = dist_utils.gather_tensor(pred_direction, args)
            direction = dist_utils.gather_tensor(direction, args)

        if len(direction.shape) == 3:
            pred_expanded = pred_direction.unsqueeze(1).expand(-1, 10, -1)
            distances = torch.norm(direction - pred_expanded, dim=2)
            min_dist_indices = distances.argmin(dim=1)
            direction = torch.gather(direction, 1, min_dist_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 3))
            direction = direction.squeeze(1)

        acc_45 = calculate_acc(pred_direction, direction, k=45)
        acc_30 = calculate_acc(pred_direction, direction, k=30)
        acc_15 = calculate_acc(pred_direction, direction, k=15)
        acc_5 = calculate_acc(pred_direction, direction, k=5)

        metrics = AccMetric(acc_45)
        if metrics.better_than(best_metrics):
            best_metrics = metrics

        print_log('[Validation] EPOCH: %d  acc_45 = %.4f acc_30 = %.4f acc_15 = %.4f acc_5 = %.4f best_acc = %.4f' % (epoch, acc_45, acc_30, acc_15, acc_5, best_metrics.acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc_45, epoch)

    return metrics


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)  # for finetuned transformer
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    base_model.eval()  # set model to eval mode

    pred_direction_list = []
    direction_list = []
    with torch.no_grad():
        for idx, (pcs, direction, instruction) in enumerate(test_dataloader):

            if args.use_gpu:
                pcs = pcs.to(args.local_rank)
                direction = direction.to(args.local_rank)

            pred_direction = base_model(pcs, instruction).squeeze()

            pred_direction_list.append(pred_direction)
            direction_list.append(direction)

        pred_direction = torch.cat(pred_direction_list, dim=0)
        direction = torch.cat(direction_list, dim=0)

        if args.distributed:
            pred_direction = dist_utils.gather_tensor(pred_direction, args)
            direction = dist_utils.gather_tensor(direction, args)

        acc_45 = calculate_acc(pred_direction, direction, k=45)
        acc_30 = calculate_acc(pred_direction, direction, k=30)
        acc_15 = calculate_acc(pred_direction, direction, k=15)
        acc_5 = calculate_acc(pred_direction, direction, k=5)

        print_log('[Test] acc_45 = %.4f acc_30 = %.4f acc_15 = %.4f acc_5 = %.4f' % (acc_45, acc_30, acc_15, acc_5),
                  logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
