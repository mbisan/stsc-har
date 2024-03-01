"""
BSD 2-Clause License

Copyright (c) 2020, Yonglong Tian
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class ContrastiveDist(nn.Module):

    def __init__(self, epsilon: float = 1e-6, m: float = 10.0) -> None:

        super().__init__()

        self.epsilon = epsilon
        self.m = m

    def forward(self, features, labels):
        '''
            features has shape (n, d)
            labels has shape (n)

            for each feature in the batch x_i, with its label y_i,  
            the elements of the same class are A(i) = {j | y_i = y_j and i!=j}   

            The elements of other classes are B(i) = {j | y_i != y_j and i!=j}

            Loss to minimize:

            Sum_i {
                // relu ensures that elements less than 0 don't propagate gradients
                ReLu(Mean_{j in A(i)} { dis(x_i, x_j) }
                - Mean_{j in B(i)} { dis(x_i, x_j)  + M)}
            }
        '''

        features = features[labels!=100, :]
        labels = labels[labels!=100]

        diff = features[:, None, :] - features[None, :, :] # shape (n, n, d)
        ed = diff.square().sum(dim=-1) # this matrix contains dis(x_i, x_j)

        equal = (labels.unsqueeze(0) == labels.unsqueeze(1))
        counts = equal.sum(1, keepdim=True) - 1
        w = equal/(counts + self.epsilon) - ~equal/(labels.shape[0]-counts-1 + self.epsilon)
        w[torch.arange(w.shape[0]), torch.arange(w.shape[0])] = 0

        loss = (w*ed).sum(1) + self.m

        loss_valid = loss[counts.squeeze()!=0]

        return torch.nn.functional.relu(loss_valid).mean()

class TripletLoss(nn.Module):

    def __init__(self, epsilon: float = 1e-6, m: float = 2.0) -> None:

        super().__init__()

        self.epsilon = epsilon
        self.m = m

    def forward(self, features, _):
        '''
            features has shape (n, 3, d)
            labels has shape (n) (not used)

            Loss to minimize:

            Mean_i {
                // relu ensures that elements less than 0 don't propagate gradients
                ReLu( dis(x_i_0, x_i_1)
                - dis(x_i_0, x_i_2)  + M)
            }
        '''

        if len(features.shape) == 2:
            return 0

        maximize = (features[:, 0, :] - features[:, 1, :]).square().sum(dim=-1) # shape n
        minimize = (features[:, 0, :] - features[:, 2, :]).square().sum(dim=-1) # shape n

        loss_parts = (maximize + self.epsilon).sqrt() - (minimize + self.epsilon).sqrt() + self.m

        return torch.nn.functional.relu(loss_parts).mean()
