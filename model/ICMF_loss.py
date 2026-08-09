import torch
from torch import nn
import torch.nn.functional as F

class CustomMultiLossLayer(nn.Module):


    def __init__(self, loss_num):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list, valid_mask=None):
        assert len(loss_list) == self.loss_num
        if valid_mask is None:
            valid_mask = [True] * self.loss_num
        assert len(valid_mask) == self.loss_num

        loss = self.log_vars.new_zeros(())
        for i in range(self.loss_num):
            if not valid_mask[i]:
                continue
            precision = torch.exp(-self.log_vars[i])
            loss = loss + precision * loss_list[i] + self.log_vars[i]
        return loss


class icl_loss(nn.Module):


    def __init__(self, tau=0.05, ab_weight=0.5, n_view=2, use_hnc=False, hnc_margin=0.2, hnc_topk=3):
        super(icl_loss, self).__init__()
        self.tau = tau
        self.weight = ab_weight
        self.n_view = n_view
        self.use_hnc = use_hnc
        self.hnc_margin = hnc_margin
        self.hnc_topk = hnc_topk

    def softXEnt(self, target, logits):
        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def _apply_hnc(self, logits, pos_mask):

        if (not self.use_hnc) or self.hnc_margin <= 0.0 or self.hnc_topk <= 0:
            return logits
        if logits.size(1) <= 1:
            return logits

        k = min(int(self.hnc_topk), max(0, logits.size(1) - 1))
        if k <= 0:
            return logits

        neg_logits = logits.masked_fill(pos_mask.bool(), float("-inf"))
        hard_idx = torch.topk(neg_logits, k=k, dim=1).indices
        hard_mask = torch.zeros_like(logits, dtype=torch.bool)
        hard_mask.scatter_(1, hard_idx, True)

        calibrated = logits.clone()
        margin = torch.full_like(logits, self.hnc_margin / self.tau)
        calibrated = calibrated + hard_mask.to(dtype=logits.dtype) * margin
        return calibrated

    def forward(self, emb, train_links, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
        zis = emb[train_links[:, 0]]
        zjs = emb[train_links[:, 1]]

        alpha = self.weight
        n_view = self.n_view
        hidden1, hidden2 = zis, zjs
        LARGE_NUM = 65500.0 if hidden1.dtype == torch.float16 else 1e9
        batch_size = hidden1.shape[0]
        hidden1_large = hidden1
        hidden2_large = hidden2

        num_classes = batch_size * n_view
        device = hidden1.device
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float().to(device)

        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float().to(device)
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / self.tau
        logits_aa = logits_aa - masks * LARGE_NUM

        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / self.tau
        logits_bb = logits_bb - masks * LARGE_NUM

        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / self.tau
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / self.tau


        logits_ab = self._apply_hnc(logits_ab, masks)
        logits_ba = self._apply_hnc(logits_ba, masks)
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        loss_a = self.softXEnt(labels, logits_a)
        loss_b = self.softXEnt(labels, logits_b)
        return alpha * loss_a + (1 - alpha) * loss_b
