"""
utils for clip
"""
import os

import torch
import torch.distributed as dist
import torch.distributed.nn as distnn
from torch import nn
import numpy

def init_distributed_mode(args):
    """ init for distribute mode """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    '''
    This is commented due to the stupid icoding pylint checking.
    print('distributed init rank {}: {}'.format(args.rank, args.dist_url), flush=True)
    '''
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()


def ddp_all_reduce(*args):
    """ all reduce (op: sum) by ddp """
    t = torch.tensor([x for x in args], dtype=torch.float64, device='cuda')
    dist.barrier()
    dist.all_reduce(t)
    t = t.tolist()
    return t


def ddp_all_gather(*args):
    """ all gather by ddp, all gather don't have grad_fn by default """
    rets = []
    world_size = dist.get_world_size()
    for x in args:
        if type(x) is torch.Tensor:
            ret = [torch.zeros_like(x) for _ in range(world_size)]
            dist.barrier()
            dist.all_gather(ret, x)
        else:  # for any picklable object
            ret = [None for _ in range(world_size)]
            dist.barrier()
            dist.all_gather_object(ret, x)
        rets.append(ret)
    return rets if len(rets) > 1 else rets[0]




def gather_labels(labels):
    # We gather tensors from all gpus
    gathered_labels = ddp_all_gather(labels)
    all_labels = torch.cat(gathered_labels)
    return all_labels


def gen_label_cpu(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt


def gen_label(labels):
    num = len(labels)
    gt = torch.zeros(size=(num,num))
    labels_column = labels.reshape(-1,1).repeat(1,num)
    labels_row = labels.repeat(num,1)
    gt[labels_column == labels_row] = 1
    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    # print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()



def gather_features(
        image_features, text_features,
        local_loss=False, gather_with_grad=False, rank=0, world_size=1):

    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(distnn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(distnn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
    return all_image_features, all_text_features



def create_logits(image_features, text_features, logit_scale, local_loss=False):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    if dist.get_world_size() > 1:
        all_image_features, all_text_features = gather_features(
            image_features, text_features,
            local_loss=local_loss, gather_with_grad=False, 
            rank=dist.get_rank(), world_size=dist.get_world_size())
            
        # cosine similarity as logits
        if local_loss:
            logits_per_image = logit_scale * image_features @ all_text_features.T
            logits_per_text = logit_scale * text_features @ all_image_features.T
        else:
            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T   

    else:
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T                 

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text





def epoch_saving(epoch, model, video_head, optimizer, filename):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'fusion_model_state_dict': video_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, filename) #just change to your preferred folder/filename

def best_saving(working_dir, epoch, model, video_head, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': video_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


from torchnet import meter
def mean_average_precision(probs, labels):
    """Computes MAP for ActivityNet evaluation"""
    if not isinstance(probs, torch.Tensor):
        probs = torch.Tensor(probs).cuda()
    
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels).long().cuda()

    gt = torch.zeros_like(probs).int()
    acc_meter = meter.ClassErrorMeter(topk=[1, 3], accuracy=True)
    gt[torch.LongTensor(range(gt.size(0))), labels] = 1
    acc_meter.add(probs, labels)
    acc = acc_meter.value()

    probs = torch.nn.functional.softmax(probs, dim=1)
    
    map_meter = meter.mAPMeter()
    map_meter.add(probs, gt)
    ap = map_meter.value()
    ap = float(ap) * 100
    return [torch.tensor(acc[0]).cuda(), torch.tensor(ap).cuda()]


if __name__=='__main__':
    probs = torch.load('ANet_similarity_336.pth')        # similarity logits
    labels = torch.load('ANet_labels_336.pth')       # class ids

    mAP = mean_average_precision(probs, labels)
    print(mAP)

