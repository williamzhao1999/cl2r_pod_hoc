import torch
import torch.nn.functional as F

import time

from cl2r.utils import AverageMeter, log_epoch, l2_norm


def train(args, net, train_loader, optimizer, epoch, criterion_cls, previous_net, task_id):
    
    start = time.time()
    
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    net.train()
    for inputs, targets, t in train_loader:
        
        inputs, targets = inputs.cuda(args.device), targets.cuda(args.device)
        outputs = net(inputs)
        feature = outputs['features']
        output = outputs['output']
        intermediary_features = outputs['attention']

        loss = criterion_cls(output, targets)
        
        if previous_net is not None:
            with torch.no_grad():
                outputs = previous_net(inputs)
                feature_old = outputs['features']
                logits_old = outputs['output']
                old_intermediary_features = outputs['attention']

            if args.use_partial_memory == True:
                #print("use_partial_memory3")
                feat_old = feature_old[:args.batch_size//2] # only on memory samples
                feat_new = feature[:args.batch_size//2]     # only on memory samples
            else:
                #print("full_memory3")
                feat_old = feature_old
                feat_new = feature

            norm_feature_old, norm_feature_new = l2_norm(feat_old), l2_norm(feat_new)

            loss_fd = EmbeddingsSimilarity(norm_feature_new, norm_feature_old)
            loss = loss + args.criterion_weight * loss_fd

            if args.pod_loss == True:
                norm_old_intermediary_features = []
                norm_intermediary_features = []
                for old_int_f in old_intermediary_features:
                    if args.use_partial_memory == True:
                        #print("use_partial_memory")
                        norm_old_intermediary_features.append((old_int_f[:args.batch_size//2]))
                    else:
                        #print("full_memory")
                        norm_old_intermediary_features.append((old_int_f))
                for int_f in intermediary_features:
                    if args.use_partial_memory == True:
                        #print("use_partial_memory2")
                        norm_intermediary_features.append((int_f[:args.batch_size//2]))
                    else:
                        #print("full_memory2")
                        norm_intermediary_features.append((int_f))
                
                pod_spatial_loss = args.spatial_lambda_c * pod(
                        norm_old_intermediary_features,
                        norm_intermediary_features,
                )
                #print(f"pod_spatial_loss {pod_spatial_loss}, args.spatial_lambda_c {args.spatial_lambda_c}, \
                #      args.use_partial_memory {args.use_partial_memory}, args.criterion_weight {args.criterion_weight}")
                loss += pod_spatial_loss
            

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), inputs.size(0))

        acc_training = accuracy(output, targets, topk=(1,))
        acc_meter.update(acc_training[0].item(), inputs.size(0))

    end = time.time()
    log_epoch(args.epochs, loss_meter.avg, acc_meter.avg, epoch=epoch, task=task_id, time=end-start)


def EmbeddingsSimilarity(feature_a, feature_b):
    return F.cosine_embedding_loss(
        feature_a, feature_b,
        torch.ones(feature_a.shape[0]).to(feature_a.device)
    )

def pod(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="spatial",
    normalize=True,
    memory_flags=None,
    only_old=False,
    **kwargs
):
    """Pooled Output Distillation.

    Reference:
        * Douillard et al.
          Small Task Incremental Learning.
          arXiv 2020.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    assert len(list_attentions_a) == len(list_attentions_b)

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if only_old:
            a = a[memory_flags]
            b = b[memory_flags]
            if len(a) == 0:
                continue

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_a)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def classification(args, net, loader, criterion_cls):
    net.eval()
    classification_loss_meter = AverageMeter()
    classification_acc_meter = AverageMeter()
    with torch.no_grad():
        for inputs, targets, t in loader:

            inputs, targets = inputs.cuda(args.device), targets.cuda(args.device)
            outputs = net(inputs)
            feature = outputs['features']
            output = outputs['output']
            intermediary_features = outputs['attention']
            loss = criterion_cls(output, targets)

            classification_loss_meter.update(loss.item(), inputs.size(0))
            classification_acc = accuracy(output, targets, topk=(1,))
            classification_acc_meter.update(classification_acc[0].item(), inputs.size(0))

    log_epoch(loss=classification_loss_meter.avg, acc=classification_acc_meter.avg, classification=True)

    classification_acc = classification_acc_meter.avg

    return classification_acc