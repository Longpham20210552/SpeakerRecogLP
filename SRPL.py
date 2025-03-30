import torch
import torch.nn as nn
import torch.nn.functional as F
from Dist import Dist

# Implementation of SRPL loss
class ARPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(ARPLoss, self).__init__()
        self.use_gpu = options['use_gpu']
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim'])
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def forward(self, x, y, labels=None):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot') #dot
        dist_l2_p = self.Dist(x, center=self.points)                #l2
        logits = dist_l2_p - dist_dot_p
        if labels is None:
            return logits, 0
        loss = F.cross_entropy(logits / self.temp, labels)

        # Calculate radius loss
        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size(), device=x.device)  # Ensure target is on the same device as x
        loss_r = self.margin_loss(self.radius.to(x.device), _dis_known, target)  # Ensure radius is on the same device

        loss = loss + self.weight_pl * loss_r
        return logits, loss

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()
        return loss
