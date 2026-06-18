# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn

from misc.detr_utils import box_ops
from misc.detr_utils.misc import (accuracy, get_world_size,
                         is_dist_avail_and_initialized)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, focal_gamma=2, opt={}):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.opt = opt
        counter_class_rate = [0.00000000e+00, 0.00000000e+00, 1.93425917e-01, 4.12129084e-01,
       1.88929963e-01, 7.81296833e-02, 5.09541413e-02, 3.12718553e-02,
       1.84833650e-02, 8.39244680e-03, 6.59406534e-03, 4.49595364e-03,
       2.19802178e-03, 1.79838146e-03, 5.99460486e-04, 4.99550405e-04,
       4.99550405e-04, 1.99820162e-04, 2.99730243e-04, 3.99640324e-04,
       2.99730243e-04, 0.00000000e+00, 1.99820162e-04, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 9.99100809e-05, 9.99100809e-05]
        self.counter_class_rate = torch.tensor(counter_class_rate)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        indices, many2one_indices = indices
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        pred_count = outputs['pred_count']
        max_length = pred_count.shape[1] - 1
        counter_target = [len(target['boxes']) if len(target['boxes']) < max_length  else max_length for target in targets]
        counter_target = torch.tensor(counter_target, device=src_logits.device, dtype=torch.long)
        counter_target_onehot = torch.zeros_like(pred_count)
        counter_target_onehot.scatter_(1, counter_target.unsqueeze(-1), 1)
        weight = self.counter_class_rate[:max_length + 1].to(src_logits.device)

        counter_loss = cross_entropy_with_gaussian_mask(pred_count, counter_target_onehot, self.opt, weight)
        losses['loss_counter'] = counter_loss

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 2]
           The target boxes are expected in format (center, length), normalized by the image size.
        """
        indices, many2one_indices = indices
        N = len(indices[-1][0])
        assert 'pred_boxes' in outputs
        idx, idx2 = self._get_src_permutation_idx2(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cl_to_xy(src_boxes),
            box_ops.box_cl_to_xy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        self_iou = torch.triu(box_ops.box_iou(box_ops.box_cl_to_xy(src_boxes),
                                              box_ops.box_cl_to_xy(src_boxes))[0], diagonal=1)
        sizes = [len(v[0]) for v in indices]
        self_iou_split = 0
        for i, c in enumerate(self_iou.split(sizes, -1)):
            cc = c.split(sizes, -2)[i]
            self_iou_split += cc.sum() / (0.5 * (sizes[i]) * (sizes[i]-1))
        losses['loss_self_iou'] = self_iou_split

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_permutation_idx2(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        src_idx2 = torch.cat([src for (_, src) in indices])
        return (batch_idx, src_idx), src_idx2

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        last_indices = self.matcher(outputs_without_aux, targets)
        outputs['matched_indices'] = last_indices

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, last_indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            aux_indices = []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                aux_indices.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            return losses, last_indices, aux_indices
        return losses, last_indices

def cross_entropy_with_gaussian_mask(inputs, targets, opt, weight):
    gau_mask = opt.lloss_gau_mask
    beta = opt.lloss_beta

    N_, max_seq_len = targets.shape
    gassian_mu = torch.arange(max_seq_len, device=inputs.device).unsqueeze(0).expand(max_seq_len,
                                                                                     max_seq_len).float()
    x = gassian_mu.transpose(0, 1)
    gassian_sigma = 2
    mask_dict = torch.exp(-(x - gassian_mu) ** 2 / (2 * gassian_sigma ** 2))
    _, ind = targets.max(dim=1)
    mask = mask_dict[ind]

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight= 1 - weight)
    if gau_mask:
        coef = targets + ((1 - mask) ** beta) * (1 - targets)
    else:
        coef = targets + (1 - targets)
    loss = loss * coef
    loss = loss.mean(1)
    return loss.mean()

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def regression_loss(inputs, targets, opt, weight):
    inputs = F.relu(inputs) + 2
    max_id = torch.argmax(targets, dim=1)
    if opt.regression_loss_type == 'l1':
        loss = nn.L1Loss()(inputs[:, 0], max_id.float())
    elif opt.regression_loss_type == 'l2':
        loss = nn.MSELoss()(inputs[:, 0], max_id.float())
    return loss





class SetCriterion_cl(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, focal_gamma=2, opt={}):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.opt = opt
        counter_class_rate = [0.00000000e+00, 0.00000000e+00, 1.93425917e-01, 4.12129084e-01,
       1.88929963e-01, 7.81296833e-02, 5.09541413e-02, 3.12718553e-02,
       1.84833650e-02, 8.39244680e-03, 6.59406534e-03, 4.49595364e-03,
       2.19802178e-03, 1.79838146e-03, 5.99460486e-04, 4.99550405e-04,
       4.99550405e-04, 1.99820162e-04, 2.99730243e-04, 3.99640324e-04,
       2.99730243e-04, 0.00000000e+00, 1.99820162e-04, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 9.99100809e-05, 9.99100809e-05]
        self.counter_class_rate = torch.tensor(counter_class_rate)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        indices, many2one_indices = indices
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        pred_count = outputs['pred_count']
        max_length = pred_count.shape[1] - 1
        counter_target = [len(target['boxes']) if len(target['boxes']) < max_length  else max_length for target in targets]
        counter_target = torch.tensor(counter_target, device=src_logits.device, dtype=torch.long)
        counter_target_onehot = torch.zeros_like(pred_count)
        counter_target_onehot.scatter_(1, counter_target.unsqueeze(-1), 1)
        weight = self.counter_class_rate[:max_length + 1].to(src_logits.device)

        counter_loss = cross_entropy_with_gaussian_mask(pred_count, counter_target_onehot, self.opt, weight)
        losses['loss_counter'] = counter_loss

        if 'cap_cost_mat' in outputs:
            caption_loss = outputs['cap_cost_mat']
            cap_loss_list = []
            for i, (event_ids, cap_ids) in enumerate(indices):
                cap_loss = caption_loss[event_ids, cap_ids]
                cap_loss_list.append(cap_loss.mean())
            losses['loss_caption'] = sum(cap_loss_list) / len(cap_loss_list)
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 2]
           The target boxes are expected in format (center, length), normalized by the image size.
        """
        indices, many2one_indices = indices
        N = len(indices[-1][0])
        assert 'pred_boxes' in outputs
        idx, idx2 = self._get_src_permutation_idx2(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cl_to_xy(src_boxes),
            box_ops.box_cl_to_xy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        self_iou = torch.triu(box_ops.box_iou(box_ops.box_cl_to_xy(src_boxes),
                                              box_ops.box_cl_to_xy(src_boxes))[0], diagonal=1)
        sizes = [len(v[0]) for v in indices]
        self_iou_split = 0
        for i, c in enumerate(self_iou.split(sizes, -1)):
            cc = c.split(sizes, -2)[i]
            self_iou_split += cc.sum() / (0.5 * (sizes[i]) * (sizes[i]-1))
        losses['loss_self_iou'] = self_iou_split

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_permutation_idx2(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        src_idx2 = torch.cat([src for (_, src) in indices])
        return (batch_idx, src_idx), src_idx2

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        last_indices = self.matcher(outputs_without_aux, targets)
        outputs['matched_indices'] = last_indices

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, last_indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            aux_indices = []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                aux_indices.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            return losses, last_indices, aux_indices
        return losses, last_indices


class ContrastiveCriterion(nn.Module):
    '''
    Contrastive loss between event feature and caption feature
    '''

    def __init__(self, temperature=0.1, enable_cross_video_cl=False, enable_e2t_cl=False, enable_bg_for_cl=False):
        super().__init__()
        self.temperature = temperature
        self.enable_cross_video_cl = enable_cross_video_cl
        self.enable_e2t_cl = enable_e2t_cl
        self.enable_bg_for_cl = enable_bg_for_cl

    def forward_logits(self, text_embed, event_embed, bg_embed=None):

        if not isinstance(text_embed, torch.Tensor):
            text_embed = torch.cat(text_embed, dim=0) if isinstance(text_embed, list) else torch.tensor(text_embed)
        if not isinstance(event_embed, torch.Tensor):
            event_embed = torch.cat(event_embed, dim=0) if isinstance(event_embed, list) else torch.tensor(event_embed)

        normalized_text_emb = F.normalize(text_embed, p=2, dim=1).float()
        normalized_event_emb = F.normalize(event_embed, p=2, dim=1).float()
        logits = torch.mm(normalized_text_emb, normalized_event_emb.t())

        if bg_embed is not None and isinstance(bg_embed, torch.Tensor):
            bg_logits = torch.sum(normalized_event_emb * F.normalize(bg_embed, p=2), dim=1)
            logits = torch.cat((logits, bg_logits.unsqueeze(0)), dim=0)
        return logits

    def _preprocess(self, event_embed, text_embed, matching_indices):

        text_features = []
        event_features = []
        gt_labels = []
        gt_event_num = []

        for i, (event_idx, text_idx) in enumerate(matching_indices):
            if len(event_idx) == 0 or len(text_idx) == 0:
                gt_event_num.append(0)
                continue

            valid_event_idx = event_idx[event_idx >= 0]
            valid_event_idx = valid_event_idx[valid_event_idx < event_embed[i].shape[0]]

            valid_text_idx = text_idx[text_idx >= 0]

            min_len = min(len(valid_event_idx), len(valid_text_idx))
            if min_len == 0:
                gt_event_num.append(0)
                continue

            valid_event_idx = valid_event_idx[:min_len]
            valid_text_idx = valid_text_idx[:min_len]

            text_feat = text_embed[i][valid_text_idx] if isinstance(text_embed[i], torch.Tensor) else \
            torch.tensor(text_embed[i])[valid_text_idx]
            event_feat = event_embed[i][valid_event_idx]

            text_features.append(text_feat)
            event_features.append(event_feat)

            batch_labels = torch.arange(len(valid_event_idx), device=event_embed.device)
            batch_labels = torch.clamp(batch_labels, 0, self.n_classes - 1)
            gt_labels.append(batch_labels)

            gt_event_num.append(len(valid_event_idx))

        if len(text_features) == 0 or len(event_features) == 0 or len(gt_labels) == 0:
            return None, None, None, torch.tensor([0] * len(matching_indices), device=event_embed.device)

        text_features = torch.cat(text_features, dim=0) if text_features else torch.tensor([],
                                                                                           device=event_embed.device)
        event_features = torch.cat(event_features, dim=0) if event_features else torch.tensor([],
                                                                                              device=event_embed.device)
        gt_labels = torch.cat(gt_labels, dim=0) if gt_labels else torch.tensor([], device=event_embed.device)
        gt_event_num = torch.tensor(gt_event_num, device=event_embed.device)

        return event_features, text_features, gt_labels, gt_event_num

    def forward(self, text_embed, event_embed, matching_indices, return_logits=False, bg_embed=None):

        event_embed_proc, text_embed_proc, gt_labels, gt_event_num = self._preprocess(event_embed, text_embed,
                                                                                      matching_indices)

        if event_embed_proc is None or text_embed_proc is None or len(gt_labels) == 0:
            if return_logits:
                return torch.tensor(0.0, device=event_embed.device), None
            return torch.tensor(0.0, device=event_embed.device)

        raw_logits = self.forward_logits(text_embed_proc, event_embed_proc, bg_embed)
        logits = raw_logits / self.temperature

        if logits.shape[1] < gt_labels.max() + 1:
            pad_len = gt_labels.max() + 1 - logits.shape[1]
            logits = torch.cat([logits, torch.zeros(logits.shape[0], pad_len, device=logits.device)], dim=1)

        loss = torch.tensor(0.0, device=event_embed.device)
        batch_size = len(gt_event_num)

        if self.enable_cross_video_cl:
            t2e_loss = F.cross_entropy(logits, gt_labels) if len(gt_labels) > 0 else 0.0
            if self.enable_e2t_cl and bg_embed is not None:
                gt_label_matrix = torch.zeros(len(text_embed_proc) + 1, len(event_embed_proc), device=logits.device)
                gt_label_matrix[torch.arange(len(gt_labels)), gt_labels] = 1
                event_mask = gt_label_matrix.sum(dim=0) == 0
                e2t_gt_label = gt_label_matrix.max(dim=0)[1]

                bg_logits = torch.sum(F.normalize(event_embed_proc, p=2) * F.normalize(bg_embed, p=2), dim=1)
                e2t_logits = torch.cat((logits, bg_logits.unsqueeze(0) / self.temperature), dim=0)

                if self.enable_bg_for_cl:
                    e2t_loss = F.cross_entropy(e2t_logits.t(), e2t_gt_label) if len(e2t_gt_label) > 0 else 0.0
                else:
                    e2t_loss = F.cross_entropy(e2t_logits.t()[~event_mask], e2t_gt_label[~event_mask]) if sum(
                        ~event_mask) > 0 else 0.0
                loss = 0.5 * (t2e_loss + e2t_loss)
            else:
                loss = t2e_loss
        else:
            base = 0
            for i in range(batch_size):
                current_gt_event_num = gt_event_num[i]
                if current_gt_event_num == 0:
                    continue

                current_logits = logits[base:base + current_gt_event_num, i * self.n_classes:(i + 1) * self.n_classes]
                current_gt_labels = gt_labels[base:base + current_gt_event_num]

                t2e_loss = F.cross_entropy(current_logits, current_gt_labels) if len(current_gt_labels) > 0 else 0.0

                if self.enable_e2t_cl and bg_embed is not None:
                    gt_label_matrix = torch.zeros(current_gt_event_num + 1, self.n_classes, device=logits.device)
                    gt_label_matrix[torch.arange(len(current_gt_labels)), current_gt_labels] = 1
                    event_mask = gt_label_matrix.sum(dim=0) == 0
                    e2t_gt_label = gt_label_matrix.max(dim=0)[1]

                    bg_logits = torch.sum(F.normalize(event_embed_proc, p=2) * F.normalize(bg_embed, p=2), dim=1)
                    e2t_logits = torch.cat((current_logits, bg_logits.unsqueeze(0) / self.temperature), dim=0)

                    if self.enable_bg_for_cl:
                        e2t_loss = F.cross_entropy(e2t_logits.t(), e2t_gt_label) if len(e2t_gt_label) > 0 else 0.0
                    else:
                        e2t_loss = F.cross_entropy(e2t_logits.t()[~event_mask], e2t_gt_label[~event_mask]) if sum(
                            ~event_mask) > 0 else 0.0
                    loss += 0.5 * (t2e_loss + e2t_loss)
                else:
                    loss += t2e_loss
                base += current_gt_event_num
            loss = loss / batch_size if batch_size > 0 else loss

        if return_logits:
            return loss, raw_logits
        return loss

    def _preprocess(self, event_embed, text_embed, matching_indices):

        text_features = []
        event_features = []
        gt_labels = []
        gt_event_num = []

        for i, (event_idx, text_idx) in enumerate(matching_indices):
            valid_event_idx = event_idx[event_idx >= 0]
            valid_event_idx = valid_event_idx[valid_event_idx < event_embed[i].shape[0]]
            valid_text_idx = text_idx[text_idx >= 0]
            valid_text_idx = valid_text_idx[valid_text_idx < text_embed[i].shape[0]]

            min_len = min(len(valid_event_idx), len(valid_text_idx))
            if min_len == 0:
                gt_event_num.append(0)
                continue

            valid_event_idx = valid_event_idx[:min_len]
            valid_text_idx = valid_text_idx[:min_len]

            text_features.append(text_embed[i][valid_text_idx])
            event_features.append(event_embed[i][valid_event_idx])

            batch_labels = torch.arange(len(valid_event_idx), device=event_embed.device)
            batch_labels = torch.clamp(batch_labels, 0, event_embed[i].shape[0] - 1)
            gt_labels.append(batch_labels)

            gt_event_num.append(len(valid_event_idx))

        if len(text_features) == 0 or len(event_features) == 0 or len(gt_labels) == 0:
            return None, None, None, torch.tensor([0] * len(matching_indices), device=event_embed.device)

        text_features = torch.cat(text_features, dim=0)
        event_features = torch.cat(event_features, dim=0)
        gt_labels = torch.cat(gt_labels, dim=0)
        gt_event_num = torch.tensor(gt_event_num, device=event_embed.device)

        return event_features, text_features, gt_labels, gt_event_num

