import torch
from torch import distributed as dist


def init_dist(local_rank, backend='nccl', **kwargs):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)
    print(f'init distributed in rank {local_rank}')


def reduce_tensor(tensor, args):
    '''
        for acc kind, get the mean in each gpu
    '''
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= args.world_size
    return rt


def gather_tensor(tensor, args):
    output_tensors = [tensor.clone() for _ in range(args.world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat


def set_batch_size(args, config):
    if args.distributed:
        assert config.total_bs % args.world_size == 0
        if config.dataset.get('train'):
            config.dataset.train.others.bs = config.total_bs // args.world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // args.world_size
        if config.dataset.get('val'):
            config.dataset.val.others.bs = config.total_bs // args.world_size
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // args.world_size
    else:
        if config.dataset.get('train'):
            config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs
        if config.dataset.get('extra_val'):
            config.dataset.extra_val.others.bs = config.total_bs
        if config.dataset.get('val'):
            config.dataset.val.others.bs = config.total_bs
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
