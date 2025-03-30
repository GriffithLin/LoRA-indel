#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/26 21:04
# @Author : fhh
# @FileName: model.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.util import *
from typing import Optional, Sequence, Any, Tuple
from typing import List, Dict
from torch.autograd import Function
from models.FFN import *
from models.ban import *

torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式
def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix
def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

class Theta(nn.Module):
    """
    maximize loss respect to :math:`\theta`
    minimize loss respect to features
    """
    def __init__(self, dim: int):
        super(Theta, self).__init__()
        self.grl1 = GradientReverseLayer()
        self.grl2 = GradientReverseLayer()
        self.layer1 = nn.Linear(dim, dim)
        nn.init.eye_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.grl1(features)
        return self.grl2(self.layer1(features))


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class DomainAdversarialLoss(nn.Module):
    r"""
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    Examples::

        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None, sigmoid=True):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        if self.sigmoid:
            d_s, d_t = d.chunk(2, dim=0)
            d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
            self.domain_discriminator_accuracy = 0.5 * (
                        binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
            )
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)

class DomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True, sigmoid=True):
        if sigmoid:
            final_layer = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            final_layer = nn.Linear(hidden_size, 2)
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                final_layer
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                final_layer
            )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]



class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks (ICML 2015) <https://arxiv.org/pdf/1502.02791>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as

    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},

    :math:`k` is a kernel function in the function space

    .. math::
        \mathcal{K} \triangleq \{ k=\sum_{u=1}^{m}\beta_{u} k_{u} \}

    where :math:`k_{u}` is a single kernel.

    Using kernel trick, MK-MMD can be computed as

    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s})\\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t})\\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}).\\

    Args:
        kernels (tuple(torch.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`

    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels.

    Examples::

        >>> from tllib.modules.kernels import GaussianKernel
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(z_s, z_t)
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)


        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss



class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Joint Multiple Kernel Maximum Mean Discrepancy (JMMD) used in
    `Deep Transfer Learning with Joint Adaptation Networks (ICML 2017) <https://arxiv.org/abs/1605.06636>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations in layers :math:`\mathcal{L}` as :math:`\{(z_i^{s1}, ..., z_i^{s|\mathcal{L}|})\}_{i=1}^{n_s}` and
    :math:`\{(z_i^{t1}, ..., z_i^{t|\mathcal{L}|})\}_{i=1}^{n_t}`. The empirical estimate of
    :math:`\hat{D}_{\mathcal{L}}(P, Q)` is computed as the squared distance between the empirical kernel mean
    embeddings as

    .. math::
        \hat{D}_{\mathcal{L}}(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{sl}) \\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{tl}, z_j^{tl}) \\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{tl}). \\

    Args:
        kernels (tuple(tuple(torch.nn.Module))): kernel functions, where `kernels[r]` corresponds to kernel :math:`k^{\mathcal{L}[r]}`.
        linear (bool): whether use the linear version of JAN. Default: False
        thetas (list(Theta): use adversarial version JAN if not None. Default: None

    Inputs:
        - z_s (tuple(tensor)): multiple layers' activations from the source domain, :math:`z^s`
        - z_t (tuple(tensor)): multiple layers' activations from the target domain, :math:`z^t`

    Shape:
        - :math:`z^{sl}` and :math:`z^{tl}`: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{sl}` and :math:`z^{tl}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels for a certain layer.

    Examples::

        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> layer1_kernels = (GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.))
        >>> layer2_kernels = (GaussianKernel(1.), )
        >>> loss = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, layer2_kernels))
        >>> # layer1 features from source domain and target domain
        >>> z1_s, z1_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # layer2 features from source domain and target domain
        >>> z2_s, z2_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss((z1_s, z2_s), (z1_t, z2_t))
    """

    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: Optional[bool] = True, thetas: Sequence[nn.Module] = None):
        super(JointMultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        if thetas:
            self.thetas = thetas
        else:
            self.thetas = [nn.Identity() for _ in kernels]

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s[0].device)

        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            kernel_matrix *= sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)




class TextCNN(nn.Module):
    def __init__(self, in_channels, n_filters, filter_sizes, dropout, pool_num = 10):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_channels,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])
        # self.fc = nn.Sequential(nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters), nn.Mish(),
        #                         nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2), nn.Mish(),
        #                         nn.Linear(len(filter_sizes) * n_filters // 2, output_dim)
        #                         )
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.fan = FAN_encode(dropout, len(filter_sizes) * n_filters * pool_num)
        self.pool_num = pool_num

    def forward(self, data, length=None, encode='sequence'):
        # 在最后一维进行卷积的，进行维度变换
        # [bs, fea_size, sequen_length]
        embedded = data.permute(0, 2, 1)

        # 多分枝卷积
        # [bs, n_filters , sequen_length] *  filter_sizes
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        # 多分枝最大池化
        # [bs, 1, self.pool_num  * n_filters]* filter_sizes
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // self.pool_num)) for conv in conved]
        # 多分枝线性展开
        # [bs, self.pool_num  * n_filters]* filter_sizes
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        # 将各分支连接在一起
        # [bs, self.pool_num* filter_sizes * n_filters]
        cat = self.dropout(torch.cat(flatten, dim=1))
        # 输入全连接层，进行回归输出
        return self.fan(cat)

class TextCNN_CLS(nn.Module):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout, in_channels = 768):
        super(TextCNN_CLS, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_channels,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])

        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.fan = FAN_encode(dropout, len(filter_sizes) * n_filters * 10)
        self.FNN = nn.Sequential(
            nn.Linear(len(filter_sizes) * n_filters * 10, 1024), self.dropout, nn.Mish(),
            nn.Linear(1024, 512), self.dropout, nn.Mish(),
            nn.Linear(512, 256), self.dropout, nn.Mish(),
            nn.Linear(256, 64), self.dropout, nn.Mish()
        )
        self.out = nn.Linear(64, output_dim)


    def forward(self, data, length=None, encode='sequence'):
        # 在最后一维进行卷积的，进行维度变换
        # [bs, fea_size, sequen_length]
        embedded = data.permute(0, 2, 1)

        # 多分枝卷积
        # [bs, n_filters , sequen_length] *  filter_sizes
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        # 多分枝最大池化
        # [bs, 1, 10  * n_filters]* filter_sizes
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]
        # 多分枝线性展开
        # [bs, 10  * n_filters]* filter_sizes
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        # 将各分支连接在一起
        # [bs, 10* filter_sizes * n_filters]
        cat = self.dropout(torch.cat(flatten, dim=1))
        # 输入全连接层，进行回归输出
        return self.FNN(self.fan(cat))

    def cls(self, data):
        output = self.forward(data)
        result = self.out(output)
        return result
class TextCNN_noFan(nn.Module):
    def __init__(self, in_channels, n_filters, filter_sizes, dropout, pool_num = 10):
        super(TextCNN_noFan, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_channels,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.pool_num = pool_num


    def forward(self, data, length=None, encode='sequence'):
        # 在最后一维进行卷积的，进行维度变换
        # [bs, fea_size, sequen_length]
        embedded = data.permute(0, 2, 1)

        # 多分枝卷积
        # [bs, n_filters , sequen_length] *  filter_sizes
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        # 多分枝最大池化
        # [bs, 1, 10  * n_filters]* filter_sizes
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // self.pool_num )) for conv in conved]
        # 多分枝线性展开
        # [bs, 10  * n_filters]* filter_sizes
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        # 将各分支连接在一起
        # [bs, 10* filter_sizes * n_filters]
        cat = self.dropout(torch.cat(flatten, dim=1))
        # 输入全连接层，进行回归输出
        return cat

class TextCNN_confusion_noFan(nn.Module):
    def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280):
        super(TextCNN_confusion_noFan, self).__init__()
        self.dna_textCNN = TextCNN_noFan(in_channel_dna, n_filters, filter_sizes_dna, dropout)
        self.protein_textCNN = TextCNN_noFan(in_channel_protein, n_filters, filter_sizes_protein, dropout)
        # self.fc = nn.Sequential(nn.Linear(len(filter_sizes) * n_filters * 10 * 2, len(filter_sizes) * n_filters * 10), nn.Mish(),
        #                         nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters), nn.Mish(),
        #                         nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2), nn.Mish(),
        #
        #                         )
        # self.out = nn.Linear(len(filter_sizes) * n_filters // 2, output_dim)


        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.FNN = nn.Sequential(
            # TODO
            nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * 10 , 1024), self.dropout, nn.Mish(),
            nn.Linear(1024, 512), self.dropout, nn.Mish(),
            nn.Linear(512, 256), self.dropout, nn.Mish(),
            nn.Linear(256, 64), self.dropout, nn.Mish()
        )
        self.out = nn.Sequential(nn.Linear(64, output_dim))


    def forward(self, dna_data, protein_data):
        dna_fea = self.dna_textCNN(dna_data)
        protein_fea = self.protein_textCNN(protein_data)
        return self.FNN(torch.cat([dna_fea, protein_fea], dim = 1))

    def cls(self, dna_data, protein_data):
        output = self.forward(dna_data, protein_data)
        result = self.out(output)
        return result

class TextCNN_confusion_Embedding0(nn.Module):
    def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280):
        super(TextCNN_confusion_Embedding0, self).__init__()
        k_mer = 5
        kmer_length = (450 - k_mer + 1)

        self.embedding_dna = nn.Embedding(4**5+1, in_channel_dna)
        self.embedding_protein = nn.Embedding(21, in_channel_protein)
        self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes_dna, dropout)
        self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes_protein, dropout)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.FNN = nn.Sequential(
            nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * 10 , 1024), self.dropout, nn.Mish(),
            nn.Linear(1024, 512), self.dropout, nn.Mish(),
            nn.Linear(512, 256), self.dropout, nn.Mish(),
            nn.Linear(256, 64), self.dropout, nn.Mish()
        )
        self.out = nn.Sequential(nn.Linear(64, output_dim))

    # encode_output = self.addNorm(Multi_encode, self.FFN(Multi_encode))

    def forward(self, dna_data, protein_data):
        dna_embed = self.embedding_dna(dna_data)
        # dna_embed = self.positionEncoding_dna(dna_embed) + dna_embed
        # dna_embed = self.AttentionEncode_dna(dna_embed)
        dna_embed = self.dna_textCNN(dna_embed)

        protein_embed = self.embedding_protein(protein_data)
        # protein_embed = self.positionEncoding_protein(protein_embed) + protein_embed
        # protein_embed = self.AttentionEncode_protein(protein_embed)
        protein_embed = self.protein_textCNN(protein_embed)

        return self.FNN(torch.cat([dna_embed, protein_embed], dim = 1))

    def cls(self, dna_data, protein_data):
        output = self.forward(dna_data, protein_data)
        result = self.out(output)
        return result

class TextCNN_confusion_Embedding_unstudent(nn.Module):
    def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280):
        super(TextCNN_confusion_Embedding_unstudent, self).__init__()
        k_mer = 5
        kmer_length = (450 - k_mer + 1)

        self.embedding_dna = nn.Embedding(4**5+1, in_channel_dna)
        self.embedding_protein = nn.Embedding(21, in_channel_protein)
        self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes_dna, dropout)
        self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes_protein, dropout)
        self.positionEncoding_dna = PositionalEncoding(in_channel_dna, dropout)
        self.positionEncoding_protein = PositionalEncoding(in_channel_protein, dropout)
        self.AttentionEncode_dna = AttentionEncode(dropout, in_channel_dna, 8, kmer_length)
        self.AttentionEncode_protein = AttentionEncode(dropout, in_channel_protein, 8, 1022)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.FNN = nn.Sequential(
            # TODO
            nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * 10 , 1024), self.dropout, nn.Mish(),
            nn.Linear(1024, 512), self.dropout, nn.Mish(),
            nn.Linear(512, 256), self.dropout, nn.Mish(),
            nn.Linear(256, 64), self.dropout, nn.Mish()
        )
        self.out = nn.Sequential(nn.Linear(64, output_dim))

    # encode_output = self.addNorm(Multi_encode, self.FFN(Multi_encode))

    def forward(self, dna_data, protein_data):
        dna_embed = self.embedding_dna(dna_data)
        dna_embed = self.positionEncoding_dna(dna_embed) + dna_embed
        dna_embed = self.AttentionEncode_dna(dna_embed)
        dna_embed = self.dna_textCNN(dna_embed)

        protein_embed = self.embedding_protein(protein_data)
        protein_embed = self.positionEncoding_protein(protein_embed) + protein_embed
        protein_embed = self.AttentionEncode_protein(protein_embed)
        protein_embed = self.protein_textCNN(protein_embed)

        return self.FNN(torch.cat([dna_embed, protein_embed], dim = 1))

    def cls(self, dna_data, protein_data):
        f_64 = self.forward(dna_data, protein_data)
        predictions = self.out(f_64)
        if self.training:
            return f_64, predictions
        else:
            return predictions


class TextCNN_confusion_Embedding(nn.Module):
    def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280):
        super(TextCNN_confusion_Embedding, self).__init__()
        k_mer = 5
        kmer_length = (450 - k_mer + 1)

        self.embedding_dna = nn.Embedding(4**5+1, in_channel_dna)
        self.embedding_protein = nn.Embedding(21, in_channel_protein)
        self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes_dna, dropout)
        self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes_protein, dropout)
        self.positionEncoding_dna = PositionalEncoding(in_channel_dna, dropout)
        self.positionEncoding_protein = PositionalEncoding(in_channel_protein, dropout)
        self.AttentionEncode_dna = AttentionEncode(dropout, in_channel_dna, 8, kmer_length)
        self.AttentionEncode_protein = AttentionEncode(dropout, in_channel_protein, 8, 1022)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.FNN = nn.Sequential(
            # TODO
            nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * 10 , 1024), self.dropout, nn.Mish(),
            nn.Linear(1024, 512), self.dropout, nn.Mish(),
            nn.Linear(512, 256), self.dropout, nn.Mish()
        )
        self.tmp = nn.Sequential(nn.Linear(256, 64), self.dropout, nn.Mish())
        self.out = nn.Sequential(nn.Linear(64, output_dim))

    # encode_output = self.addNorm(Multi_encode, self.FFN(Multi_encode))

    def forward(self, dna_data, protein_data):
        dna_embed = self.embedding_dna(dna_data)
        dna_embed = self.positionEncoding_dna(dna_embed) + dna_embed
        dna_embed = self.AttentionEncode_dna(dna_embed)
        dna_embed = self.dna_textCNN(dna_embed)

        protein_embed = self.embedding_protein(protein_data)
        protein_embed = self.positionEncoding_protein(protein_embed) + protein_embed
        protein_embed = self.AttentionEncode_protein(protein_embed)
        protein_embed = self.protein_textCNN(protein_embed)

        return self.FNN(torch.cat([dna_embed, protein_embed], dim = 1))

    def cls(self, dna_data, protein_data):
        f_256 = self.forward(dna_data, protein_data)
        f_64 = self.tmp(f_256)
        predictions = self.out(f_64)
        if self.training:
            return f_64, predictions
        else:
            return predictions

class TextCNN_confusion_Embedding2(nn.Module):
    def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280):
        super(TextCNN_confusion_Embedding2, self).__init__()
        k_mer = 5
        kmer_length = (450 - k_mer + 1)

        self.embedding_dna = nn.Embedding(4**5+1, in_channel_dna)
        self.embedding_protein = nn.Embedding(21, in_channel_protein)
        self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes_dna, dropout)
        self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes_protein, dropout)
        self.positionEncoding_dna = PositionalEncoding(in_channel_dna, dropout)
        self.positionEncoding_protein = PositionalEncoding(in_channel_protein, dropout)
        self.AttentionEncode_dna = AttentionEncode(dropout, in_channel_dna, 8, kmer_length)
        self.AttentionEncode_protein = AttentionEncode(dropout, in_channel_protein, 8, 1022)
        self.AttentionEncode_dna2 = AttentionEncode(dropout, in_channel_dna, 8, kmer_length)
        self.AttentionEncode_protein2 = AttentionEncode(dropout, in_channel_protein, 8, 1022)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.FNN = nn.Sequential(
            # TODO
            nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * 10 , 1024), self.dropout, nn.Mish(),
            nn.Linear(1024, 512), self.dropout, nn.Mish(),
            nn.Linear(512, 256), self.dropout, nn.Mish(),
            nn.Linear(256, 64), self.dropout, nn.Mish()
        )
        self.out = nn.Sequential(nn.Linear(64, output_dim))

    # encode_output = self.addNorm(Multi_encode, self.FFN(Multi_encode))

    def forward(self, dna_data, protein_data):
        dna_embed = self.embedding_dna(dna_data)
        dna_embed = self.positionEncoding_dna(dna_embed) + dna_embed
        dna_embed = self.AttentionEncode_dna(dna_embed)
        dna_embed = self.AttentionEncode_dna2(dna_embed)
        dna_embed = self.dna_textCNN(dna_embed)

        protein_embed = self.embedding_protein(protein_data)
        protein_embed = self.positionEncoding_protein(protein_embed) + protein_embed
        protein_embed = self.AttentionEncode_protein(protein_embed)
        protein_embed = self.AttentionEncode_protein2(protein_embed)
        protein_embed = self.protein_textCNN(protein_embed)

        return self.FNN(torch.cat([dna_embed, protein_embed], dim = 1))

    def cls(self, dna_data, protein_data):
        output = self.forward(dna_data, protein_data)
        result = self.out(output)
        return result


class TextCNN_confusion_Embedding3(nn.Module):
    def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280):
        super(TextCNN_confusion_Embedding3, self).__init__()
        k_mer = 5
        kmer_length = (450 - k_mer + 1)

        self.embedding_dna = nn.Embedding(4**5+1, in_channel_dna)
        self.embedding_protein = nn.Embedding(21, in_channel_protein)
        self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes_dna, dropout)
        self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes_protein, dropout)
        self.positionEncoding_dna = PositionalEncoding(in_channel_dna, dropout)
        self.positionEncoding_protein = PositionalEncoding(in_channel_protein, dropout)
        self.AttentionEncode_dna = AttentionEncode(dropout, in_channel_dna, 8, kmer_length)
        self.AttentionEncode_protein = AttentionEncode(dropout, in_channel_protein, 8, 1022)
        self.AttentionEncode_dna2 = AttentionEncode(dropout, in_channel_dna, 8, kmer_length)
        self.AttentionEncode_protein2 = AttentionEncode(dropout, in_channel_protein, 8, 1022)
        self.AttentionEncode_dna3 = AttentionEncode(dropout, in_channel_dna, 8, kmer_length)
        self.AttentionEncode_protein3 = AttentionEncode(dropout, in_channel_protein, 8, 1022)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.FNN = nn.Sequential(
            # TODO
            nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * 10 , 1024), self.dropout, nn.Mish(),
            nn.Linear(1024, 512), self.dropout, nn.Mish(),
            nn.Linear(512, 256), self.dropout, nn.Mish(),
            nn.Linear(256, 64), self.dropout, nn.Mish()
        )
        self.out = nn.Sequential(nn.Linear(64, output_dim))

    # encode_output = self.addNorm(Multi_encode, self.FFN(Multi_encode))

    def forward(self, dna_data, protein_data):
        dna_embed = self.embedding_dna(dna_data)
        dna_embed = self.positionEncoding_dna(dna_embed) + dna_embed
        dna_embed = self.AttentionEncode_dna(dna_embed)
        dna_embed = self.AttentionEncode_dna2(dna_embed)
        dna_embed = self.AttentionEncode_dna3(dna_embed)
        dna_embed = self.dna_textCNN(dna_embed)

        protein_embed = self.embedding_protein(protein_data)
        protein_embed = self.positionEncoding_protein(protein_embed) + protein_embed
        protein_embed = self.AttentionEncode_protein(protein_embed)
        protein_embed = self.AttentionEncode_protein2(protein_embed)
        protein_embed = self.AttentionEncode_protein3(protein_embed)
        protein_embed = self.protein_textCNN(protein_embed)

        return self.FNN(torch.cat([dna_embed, protein_embed], dim = 1))

    def cls(self, dna_data, protein_data):
        output = self.forward(dna_data, protein_data)
        result = self.out(output)
        return result


# class TextCNN_confusion(nn.Module):
#     def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280):
#         super(TextCNN_confusion, self).__init__()
#         self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes_dna, dropout)
#         self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes_protein, dropout)
#         self.dropout = nn.Dropout(dropout)
#         self.Mish = nn.Mish()
#         feature_num = (len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * 10
#         if feature_num < 1024:
#             self.FNN = nn.Sequential(
#                 # TODO
#                 nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * 10 , 512), self.dropout, nn.Mish(),
#                 nn.Linear(512, 256), self.dropout, nn.Mish(),
#                 nn.Linear(256, 64), self.dropout, nn.Mish()
#             )
#         else:
#             self.FNN = nn.Sequential(
#                 # TODO
#                 nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * 10 , 1024), self.dropout, nn.Mish(),
#                 nn.Linear(1024, 512), self.dropout, nn.Mish(),
#                 nn.Linear(512, 256), self.dropout, nn.Mish(),
#                 nn.Linear(256, 64), self.dropout, nn.Mish()
#             )
#         self.out = nn.Sequential(nn.Linear(64, output_dim))


#     def forward(self, dna_data, protein_data):
#         dna_fea = self.dna_textCNN(dna_data)
#         protein_fea = self.protein_textCNN(protein_data)
#         f = self.FNN(torch.cat([dna_fea, protein_fea], dim = 1))
#         predictions = self.out(f)
#         if self.training:
#             return f, predictions
#         else:
#             return predictions

#     def cls(self, dna_data, protein_data):
#         if self.training:
#             _, result = self.forward(dna_data, protein_data)
#             # result = self.out(output)
#         else:
#             result = self.forward(dna_data, protein_data)
#         return result

# 测试特征融合
class TextCNN_confusion(nn.Module):
    def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280, fusion_mode = "cat", pool_num = 10):
        super(TextCNN_confusion, self).__init__()
        if "new" in fusion_mode:
            self.dna_textCNN = TextCNN_noFan(in_channel_dna, n_filters, filter_sizes_dna, dropout, pool_num = pool_num)
            self.protein_textCNN = TextCNN_noFan(in_channel_protein, n_filters, filter_sizes_protein, dropout, pool_num = pool_num)
        else:
            self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes_dna, dropout, pool_num = pool_num)
            self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes_protein, dropout, pool_num = pool_num)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.n_filters = n_filters
        self.filter_sizes_protein = filter_sizes_protein
        self.filter_sizes_dna = filter_sizes_dna
        self.out = nn.Sequential(nn.Linear(64, output_dim))
        self.pool_num = pool_num
        hout = 2
        feature_num = (len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * pool_num
        attention_dim = int(feature_num / 2)
        self.attention_dim = attention_dim
        self.fusion_mode = fusion_mode
        # print(fusion_mode)


        if self.fusion_mode == "ban":
            self.hout = hout
            self.ban = BANLayer(attention_dim, attention_dim, attention_dim, self.hout, 0.5, 6)  # Bilinear Attention Networks
        if self.fusion_mode == "new_ban":
            self.hout = hout
            self.ban = BANLayer(n_filters * pool_num, n_filters * pool_num, n_filters * pool_num, self.hout, 0.5, 6)
        if self.fusion_mode == "ffn":
            self.ffn = FeatureFusionNetwork(d_model = attention_dim, dropout=dropout, dim_feedforward = attention_dim * 2) 
        if self.fusion_mode == "new_ffn":
            self.ffn = FeatureFusionNetwork(d_model = n_filters * pool_num, dropout=dropout, dim_feedforward = n_filters * pool_num * 2) 
        if self.fusion_mode =="new_cross_attention" or self.fusion_mode == "new_self_attention" :
            # 把不同filter 视为不同的token
            self.attention1 = nn.MultiheadAttention(n_filters * pool_num, 8) #64是特征维度，8是头数
            self.attention2 = nn.MultiheadAttention(n_filters * pool_num, 8) #64是特征维度，8是头数
        if self.fusion_mode == "cross_attention":
            self.attention1 = nn.MultiheadAttention(attention_dim, 8) #64是特征维度，8是头数
            self.attention2 = nn.MultiheadAttention(attention_dim, 8) #64是特征维度，8是头数           
        if  self.fusion_mode == "ffn" or  self.fusion_mode == "cat" or self.fusion_mode =="new_cross_attention" or self.fusion_mode == "new_self_attention" or self.fusion_mode == "cross_attention" or self.fusion_mode == "new_ffn":
            
            self.FNN = nn.Sequential(
                #new attention
                nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * pool_num , 1024), self.dropout, nn.Mish(),
                # concat
                nn.Linear(1024, 256), self.dropout, nn.Mish(),
                nn.Linear(256, 64), self.dropout, nn.Mish()
            )
        if self.fusion_mode == "new_ban" or self.fusion_mode == "add" or self.fusion_mode == "ban":
            # or self.fusion_mode =="new_cross_attentionadd" 
            if self.fusion_mode == "new_ban" :
                in_dim = n_filters * pool_num
            else:
                in_dim = attention_dim
            self.FNN = nn.Sequential(
                # add
                nn.Linear(in_dim, 256), self.dropout, nn.Mish(),
                # 
                nn.Linear(256, 64), self.dropout, nn.Mish()
            )
            # self.fnn_dna = nn.Linear(len(filter_sizes_dna) * n_filters * pool_num, 512)
            # self.fnn_protein = nn.Linear(len(filter_sizes_protein) * n_filters * pool_num, 512)


    def forward(self, dna_data, protein_data):
        dna_fea = self.dna_textCNN(dna_data)
        protein_fea = self.protein_textCNN(protein_data)
        # if self.mode512:
        #     dna_fea = self.fnn_dna(dna_fea)
        #     protein_fea =  self.fnn_protein(protein_fea)
        # CAT
        if self.fusion_mode == "cat":
            fea = torch.cat([dna_fea, protein_fea], dim = 1)
        # add
        if self.fusion_mode == "add":
            fea = dna_fea + protein_fea
        
        # if self.fusion_mode == "mul":
        #     fea = dna_fea + protein_fea
        # ban
        if self.fusion_mode == "ban":
            fea, att_weight = self.ban(dna_fea.unsqueeze(1), protein_fea.unsqueeze(1))
            # print(fea.shape)
        
        if self.fusion_mode == "new_ban":
            n_filters_per_slice = self.pool_num * self.n_filters
            new_shape = (-1, len(self.filter_sizes_dna), n_filters_per_slice)
            dna_fea = dna_fea.reshape(new_shape)#变为[batch, len(filter_sizes_dna),  feature]
            new_shape2 = (-1, len(self.filter_sizes_protein), n_filters_per_slice)
            protein_fea = protein_fea.reshape(new_shape2)#变为[batch, len(filter_sizes_protein),  feature]
            fea, att_weight = self.ban(dna_fea, protein_fea)
            # print(fea.shape)
            #  # (batch_size, h_dim)
            
        if self.fusion_mode == "cross_attention":
            # 基于交叉注意力融合两类特征  分别concat 和add 两种
            dna_fea = dna_fea.unsqueeze(0)  #变为[len(filter_sizes_dna), batch, feature]
            protein_fea = protein_fea.unsqueeze(0)
            dna_att, attention_weight = self.attention1(dna_fea, protein_fea, protein_fea)
            protein_att, attention_weight = self.attention2(protein_fea, dna_fea, dna_fea)
            ###加
            # fusConv = dna_att * 0.5 + protein_att * 0.5
            # fea = fusConv.squeeze(0)      
        ##        ###残差连接，注意力变换特征加上原始特征
            dnaConv = dna_fea * 0.5 + dna_att * 0.5
            proteinConv = protein_fea * 0.5 + protein_att * 0.5
            fea = torch.cat([dnaConv.squeeze(0), proteinConv.squeeze(0)], dim=1)
            
        
        # change fusion 
        if self.fusion_mode == "new_cross_attention":
            n_filters_per_slice = self.pool_num * self.n_filters
            new_shape = (-1, len(self.filter_sizes_dna), n_filters_per_slice)
            dna_fea = dna_fea.reshape(new_shape).permute(1, 0, 2) #变为[len(filter_sizes_dna), batch, feature]
            new_shape2 = (-1, len(self.filter_sizes_protein), n_filters_per_slice)
            protein_fea = protein_fea.reshape(new_shape2).permute(1, 0, 2) #变为[len(filter_sizes_protein), batch, feature]
            dna_att, attention_weight = self.attention1(dna_fea, protein_fea, protein_fea)
            protein_att, attention_weight = self.attention2(protein_fea, dna_fea, dna_fea)
            dnaConv = dna_fea * 0.5 + dna_att * 0.5
            proteinConv = protein_fea * 0.5 + protein_att * 0.5
            fea = torch.cat([dnaConv.permute(1, 0, 2).reshape(-1, len(self.filter_sizes_dna) * n_filters_per_slice),
                            proteinConv.permute(1, 0, 2).reshape(-1, len(self.filter_sizes_protein) * n_filters_per_slice)], dim=1)

        if self.fusion_mode == "new_self_attention":
            n_filters_per_slice = self.pool_num * self.n_filters
            new_shape = (-1, len(self.filter_sizes_dna), n_filters_per_slice)
            dna_fea = dna_fea.reshape(new_shape).permute(1, 0, 2) #变为[len(filter_sizes_dna), batch, feature]
            new_shape2 = (-1, len(self.filter_sizes_protein), n_filters_per_slice)
            protein_fea = protein_fea.reshape(new_shape2).permute(1, 0, 2) #变为[len(filter_sizes_protein), batch, feature]
            dna_att, attention_weight = self.attention1(dna_fea, dna_fea, dna_fea)
            protein_att, attention_weight = self.attention2(protein_fea, protein_fea, protein_fea)
            dnaConv = dna_fea * 0.5 + dna_att * 0.5
            proteinConv = protein_fea * 0.5 + protein_att * 0.5
            fea = torch.cat([dnaConv.permute(1, 0, 2).reshape(-1, len(self.filter_sizes_dna) * n_filters_per_slice),
                            proteinConv.permute(1, 0, 2).reshape(-1, len(self.filter_sizes_protein) * n_filters_per_slice)], dim=1)

        if self.fusion_mode == "ffn":
            # 基于自注意力+交叉注意力的FFN模型融合两类特征
            dna_fea = dna_fea.unsqueeze(0)  #变为[seq_length, batch,  feature]
            protein_fea = protein_fea.unsqueeze(0)
            ###不使用FFN中最后一个交叉注意力模块(若使用，则去FFN.py代码中进行修改)
            dna_fea, protein_fea = self.ffn(dna_fea, protein_fea)
            fea = torch.cat([dna_fea.squeeze(0), protein_fea.squeeze(0)], dim=1)
        #        ###使用FFN中最后一个交叉注意力模块
        #        fea_fusion_squeeze = self.ffn(text_fea, bio_fea)
        #        fea = fea_fusion_squeeze.squeeze(0) 
        if self.fusion_mode == "new_ffn":
            n_filters_per_slice = self.pool_num * self.n_filters
            new_shape = (-1, len(self.filter_sizes_dna), n_filters_per_slice)
            dna_fea = dna_fea.reshape(new_shape).permute(1, 0, 2) #变为[len(filter_sizes_dna), batch, feature]
            new_shape2 = (-1, len(self.filter_sizes_protein), n_filters_per_slice)
            protein_fea = protein_fea.reshape(new_shape2).permute(1, 0, 2) #变为[len(filter_sizes_protein), batch, feature]
            dna_fea, protein_fea = self.ffn(dna_fea, protein_fea)
            fea = torch.cat([dna_fea.permute(1, 0, 2).reshape(-1, len(self.filter_sizes_dna) * n_filters_per_slice),
            protein_fea.permute(1, 0, 2).reshape(-1, len(self.filter_sizes_protein) * n_filters_per_slice)], dim=1)

        f = self.FNN(fea)
        predictions = self.out(f)
        if self.training:
            return f, predictions
        else:
            return predictions

    def cls(self, dna_data, protein_data):
        if self.training:
            _, result = self.forward(dna_data, protein_data)
            # result = self.out(output)
        else:
            result = self.forward(dna_data, protein_data)
        return result

# 是否是因为  特征维度太大而数据样本少  导致的不同融合方法差距小。
class TextCNN_confusion512(nn.Module):
    def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, feature_dim = 512, in_channel_dna = 768, in_channel_protein = 1280, fusion_mode = "cat", pool_num = 10):
        super(TextCNN_confusion512, self).__init__()
        if "new" in fusion_mode:
            self.dna_textCNN = TextCNN_noFan(in_channel_dna, n_filters, filter_sizes_dna, dropout, pool_num = pool_num)
            self.protein_textCNN = TextCNN_noFan(in_channel_protein, n_filters, filter_sizes_protein, dropout, pool_num = pool_num)
        else:
            self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes_dna, dropout, pool_num = pool_num)
            self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes_protein, dropout, pool_num = pool_num)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.n_filters = n_filters
        self.filter_sizes_protein = filter_sizes_protein
        self.filter_sizes_dna = filter_sizes_dna
        self.out = nn.Sequential(nn.Linear(64, output_dim))
        self.pool_num = pool_num
        hout = 2
        feature_num = (len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * pool_num
        attention_dim = int(feature_num / 2)
        self.attention_dim = attention_dim
        self.fusion_mode = fusion_mode
        self.feature_dim = feature_dim
        self.fnn_dna = nn.Linear(len(filter_sizes_dna) * n_filters * pool_num, self.feature_dim)
        self.fnn_protein = nn.Linear(len(filter_sizes_protein) * n_filters * pool_num, self.feature_dim)



        if self.fusion_mode == "ban" or self.fusion_mode == "new_ban":
            self.hout = hout
            self.ban = BANLayer(feature_dim, feature_dim, feature_dim, self.hout, 0.5, 6)  # Bilinear Attention Networks
        if self.fusion_mode == "ffn":
            self.ffn = FeatureFusionNetwork(d_model = feature_dim, dropout=dropout, dim_feedforward = feature_dim * 2) 
        if self.fusion_mode == "cross_attention":
            self.attention1 = nn.MultiheadAttention(feature_dim, 8) #64是特征维度，8是头数
            self.attention2 = nn.MultiheadAttention(feature_dim, 8) #64是特征维度，8是头数           
        if self.fusion_mode == "cat" or  self.fusion_mode == "cross_attention":
            self.FNN = nn.Sequential(
                # concat
                nn.Linear(feature_dim * 2, 256), self.dropout, nn.Mish(),
                nn.Linear(256, 64), self.dropout, nn.Mish()
            )
        if self.fusion_mode == "ffn" or self.fusion_mode == "add" or self.fusion_mode == "ban":
            # or self.fusion_mode =="new_cross_attentionadd" 
            in_dim = feature_dim
            self.FNN = nn.Sequential(
                # add
                nn.Linear(in_dim, 256), self.dropout, nn.Mish(),
                # 
                nn.Linear(256, 64), self.dropout, nn.Mish()
            )


    def forward(self, dna_data, protein_data):
        dna_fea = self.dna_textCNN(dna_data)
        protein_fea = self.protein_textCNN(protein_data)
        dna_fea = self.fnn_dna(dna_fea)
        protein_fea =  self.fnn_protein(protein_fea)
        # CAT
        if self.fusion_mode == "cat":
            fea = torch.cat([dna_fea, protein_fea], dim = 1)
        # add
        if self.fusion_mode == "add":
            fea = dna_fea + protein_fea
        
        # if self.fusion_mode == "mul":
        #     fea = dna_fea + protein_fea
        # ban
        if self.fusion_mode == "ban":
            fea, att_weight = self.ban(dna_fea.unsqueeze(1), protein_fea.unsqueeze(1))
            # print(fea.shape)
            
        if self.fusion_mode == "cross_attention":
            # 基于交叉注意力融合两类特征  分别concat 和add 两种
            dna_fea = dna_fea.unsqueeze(0)  #变为[len(filter_sizes_dna), batch, feature]
            protein_fea = protein_fea.unsqueeze(0)
            dna_att, attention_weight = self.attention1(dna_fea, protein_fea, protein_fea)
            protein_att, attention_weight = self.attention2(protein_fea, dna_fea, dna_fea)
            ###加
            # fusConv = dna_att * 0.5 + protein_att * 0.5
            # fea = fusConv.squeeze(0)      
        ##        ###残差连接，注意力变换特征加上原始特征
            dnaConv = dna_fea * 0.5 + dna_att * 0.5
            proteinConv = protein_fea * 0.5 + protein_att * 0.5
            fea = torch.cat([dnaConv.squeeze(0), proteinConv.squeeze(0)], dim=1)
            

        if self.fusion_mode == "ffn":
            # 基于自注意力+交叉注意力的FFN模型融合两类特征
            dna_fea = dna_fea.unsqueeze(0)  #变为[seq_length, batch,  feature]
            protein_fea = protein_fea.unsqueeze(0)
            ###不使用FFN中最后一个交叉注意力模块(若使用，则去FFN.py代码中进行修改)
            # dna_fea, protein_fea = self.ffn(dna_fea, protein_fea)
            # fea = torch.cat([dna_fea.squeeze(0), protein_fea.squeeze(0)], dim=1)
        #        ###使用FFN中最后一个交叉注意力模块
            fea_fusion_squeeze = self.ffn(dna_fea, protein_fea)
            fea = fea_fusion_squeeze.squeeze(0) 

        f = self.FNN(fea)
        predictions = self.out(f)
        if self.training:
            return f, predictions
        else:
            return predictions

    def cls(self, dna_data, protein_data):
        if self.training:
            _, result = self.forward(dna_data, protein_data)
            # result = self.out(output)
        else:
            result = self.forward(dna_data, protein_data)
        return result
    
class TextCNN_confusion_DA(nn.Module):
    def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280):
        super(TextCNN_confusion_DA, self).__init__()
        self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes_dna, dropout)
        self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes_protein, dropout)
        # self.fc = nn.Sequential(nn.Linear(len(filter_sizes) * n_filters * 10 * 2, len(filter_sizes) * n_filters * 10), nn.Mish(),
        #                         nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters), nn.Mish(),
        #                         nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2), nn.Mish(),
        #
        #                         )
        # self.out = nn.Linear(len(filter_sizes) * n_filters // 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.FNN = nn.Sequential(
            # TODO
            nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein) ) * n_filters * 10 , 1024), self.dropout, nn.Mish(),
            nn.Linear(1024, 512), self.dropout, nn.Mish(),
            nn.Linear(512, 256), self.dropout, nn.Mish()

        )
        self.tmp = nn.Sequential(nn.Linear(256, 64), self.dropout, nn.Mish())
        self.out = nn.Sequential(nn.Linear(64, output_dim))

    def forward(self, dna_data, protein_data):
        dna_fea = self.dna_textCNN(dna_data)
        protein_fea = self.protein_textCNN(protein_data)
        f_256 = self.FNN(torch.cat([dna_fea, protein_fea], dim = 1))
        f = self.tmp(f_256)
        predictions = self.out(f)
        if self.training:
            return f, predictions
        else:
            return predictions

    def cls(self, dna_data, protein_data):
        if self.training:
            _, result = self.forward(dna_data, protein_data)
        else:
            result = self.forward(dna_data, protein_data)
        return result


class TextCNN_confusion_cap(nn.Module):
    def __init__(self, n_filters, filter_sizes_dna, filter_sizes_protein, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280):
        super(TextCNN_confusion_cap, self).__init__()
        self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes_dna, dropout)
        self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes_protein, dropout)
        # self.fc = nn.Sequential(nn.Linear(len(filter_sizes) * n_filters * 10 * 2, len(filter_sizes) * n_filters * 10), nn.Mish(),
        #                         nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters), nn.Mish(),
        #                         nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2), nn.Mish(),
        #
        #                         )
        # self.out = nn.Linear(len(filter_sizes) * n_filters // 2, output_dim)


        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.FNN = nn.Sequential(
            # TODO
            nn.Linear((len(filter_sizes_dna) + len(filter_sizes_protein)) * n_filters * 10, 1024), self.dropout, nn.Mish(),
            nn.Linear(1024, 512), self.dropout, nn.Mish(),
            nn.Linear(512, 256), self.dropout, nn.Mish(),
            nn.Linear(256, 64), self.dropout, nn.Mish()
        )
        self.out = nn.Sequential(nn.Linear(64 + 26, output_dim))


    def forward(self, dna_data, protein_data):
        dna_fea = self.dna_textCNN(dna_data)
        protein_fea = self.protein_textCNN(protein_data)
        return self.FNN(torch.cat([dna_fea, protein_fea], dim = 1))

    def cls(self, dna_data, protein_data, cap_fea):
        output = self.forward(dna_data, protein_data)
        result = self.out(torch.cat([output, cap_fea], dim = 1))
        return result



class TextCNN_confusion_batch_pad(nn.Module):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout, in_channel_dna = 768, in_channel_protein = 1280):
        super(TextCNN_confusion_batch_pad, self).__init__()
        self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes, dropout)
        self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes, dropout)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.FNN = nn.Sequential(
            # TODO
            nn.Linear(len(filter_sizes) * n_filters * 10 * 2, 1024), self.dropout, nn.BatchNorm1d(1024), nn.Mish(),
            nn.Linear(1024, 512), self.dropout, nn.BatchNorm1d(512), nn.Mish(),
            nn.Linear(512, 256), self.dropout, nn.BatchNorm1d(256), nn.Mish(),
            nn.Linear(256, 64), self.dropout, nn.BatchNorm1d(64), nn.Mish()
        )
        self.out = nn.Sequential(nn.Linear(64, output_dim))


    def forward(self, dna_data, protein_data, protein_len_list):
        max_len = torch.max(protein_len_list)
        dna_fea = self.dna_textCNN(dna_data)
        protein_fea = self.protein_textCNN(protein_data[:,:max_len,:])
        return self.FNN(torch.cat([dna_fea, protein_fea], dim = 1))

    def cls(self, dna_data, protein_data, protein_len_list):
        output = self.forward(dna_data, protein_data, protein_len_list)
        result = self.out(output)
        return result


class TextCNN_confusion_Contrastive(nn.Module):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout, in_channel_dna=768, in_channel_protein=1280):
        super(TextCNN_confusion_Contrastive, self).__init__()
        self.dna_textCNN = TextCNN(in_channel_dna, n_filters, filter_sizes, dropout)
        self.protein_textCNN = TextCNN(in_channel_protein, n_filters, filter_sizes, dropout)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.FNN = nn.Sequential(
            # TODO
            nn.Linear(len(filter_sizes) * n_filters * 10 * 2, 1024), self.dropout, nn.Mish(),
            nn.Linear(1024, 512), self.dropout,  nn.BatchNorm1d(512) , nn.Mish(),
            nn.Linear(512, 256), self.dropout
        )
        # self.FNN = nn.Sequential(
        #     nn.Linear(len(filter_sizes) * n_filters * 10 * 2, 1024), nn.Mish(),
        #     nn.Linear(1024, 512), nn.Mish(),
        #     nn.Linear(512, 256), nn.Mish(),
        #     nn.Linear(256, 64), nn.BatchNorm1d(64), nn.Mish()
        #     )

        self.out =nn.Sequential( nn.Mish(),  nn.Linear(256, 64),
                                  self.dropout, nn.Mish(), nn.Linear(64, output_dim))

    def forward(self, dna_data, protein_data):
        dna_fea = self.dna_textCNN(dna_data)
        protein_fea = self.protein_textCNN(protein_data)
        return self.FNN(torch.cat([dna_fea, protein_fea], dim=1))

    def cls(self, dna_data, protein_data):
        output = self.forward(dna_data, protein_data)
        return self.out(output)


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # print('label.shape', label.shape)
        cos_distance = D(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 2))

        return loss_contrastive



class TextCNN_WithAttentionEncode_noEmbeding(nn.Module):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout, max_length):
        super(TextCNN_WithAttentionEncode_noEmbeding, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=768,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])
        self.positionEncoding = models.util.PositionalEncoding(768, 0)
        self.AttentionEncode = models.util.AttentionEncode(dropout, 768, 8, max_length)

        self.fc = nn.Sequential(nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters // 2, output_dim)
                                )
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self, data, length=None, encode='sequence'):
        embedded = data
        position = self.positionEncoding(embedded)
        embedded += position
        embedded = self.AttentionEncode(embedded)

        # 进行维度变换
        embedded = embedded.permute(0, 2, 1)
        # 多分枝卷积
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        # 多分枝最大池化
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]
        # 多分枝线性展开
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        # 将各分支连接在一起
        cat = self.dropout(torch.cat(flatten, dim=1))
        # 输入全连接层，进行回归输出
        return self.fc(cat)


class TextCNN_WithAttentionEncode(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, max_length):
        super(TextCNN_WithAttentionEncode, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])
        self.positionEncoding = models.util.PositionalEncoding(embedding_dim, 0)
        self.AttentionEncode = models.util.AttentionEncode(dropout, embedding_dim, 8, max_length)

        self.fc = nn.Sequential(nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters // 2, output_dim)
                                )
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self, data, length=None, encode='sequence'):
        # 对输入数据进行词向量映射
        # 分开再拼接
        embedded1 = self.embedding(data[:,:200])
        embedded2 = self.embedding(data[:,200:])
        embedded = torch.cat([embedded1, embedded2], dim=1)

        #一起再拼接
        # embedded = self.embedding(data)
        # embedded = torch.cat([embedded[:,:71], embedded[:,71:]], dim=2)

        #
        # embedded = self.embedding(data)
        position = self.positionEncoding(embedded)
        embedded += position
        embedded = self.AttentionEncode(embedded)

        # 进行维度变换
        embedded = embedded.permute(0, 2, 1)
        # 多分枝卷积
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        # 多分枝最大池化
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]
        # 多分枝线性展开
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        # 将各分支连接在一起
        cat = self.dropout(torch.cat(flatten, dim=1))
        # 输入全连接层，进行回归输出
        return self.fc(cat)
