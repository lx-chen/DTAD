import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normal_fl_weighting(logps, gamma=0.5, alpha=11.7, normalizer=10):
    """
    Normal focal weighting.
    Args:
        logps: og-likelihoods, shape (N, ).
        gamma: gamma hyperparameter for normal focal weighting.
        alpha: alpha hyperparameter for abnormal focal weighting.
    """
    logps = logps / normalizer
    mask_larger = logps > -0.2
    mask_lower = logps <= -0.2
    probs = torch.exp(logps)
    fl_weights = alpha * (1 - probs).pow(gamma) * torch.abs(logps)
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_larger] = 1.0 
    weights[mask_lower] = fl_weights[mask_lower]

    return weights


def abnormal_fl_weighting(logps, gamma=2, alpha=0.53, normalizer=10):
    """
    Abnormal focal weighting.
    Args:
        logps: og-likelihoods, shape (N, ).
        gamma: gamma hyperparameter for normal focal weighting.
        alpha: alpha hyperparameter for abnormal focal weighting.
    """
    logps = logps / normalizer
    mask_larger = logps > -1.0
    mask_lower = logps <= -1.0
    probs = torch.exp(logps)
    fl_weights = alpha * (1 + probs).pow(gamma) * (1 / torch.abs(logps))
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_lower] = 1.0 
    weights[mask_larger] = fl_weights[mask_larger]

    return weights

 
def get_logp_boundary(logps, lable, pos_beta=0.05, margin_tau=0.3, normalizer=10):  # pos_beta=0.05, margin_tau=0.1, normalizer=10
    """
    Find the equivalent log-likelihood decision boundaries from normal log-likelihood distribution.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, )
        pos_beta: position hyperparameter: beta
        margin_tau: margin hyperparameter: tau
    """ 
    # print("logps.shape", logps.shape) 
    normal_logps = logps[lable == 0.0].detach()  # 从 logps 中提取正常样本的对数似然值
    # print("normal_logps", normal_logps)
    n_idx = int(((lable == 0.0).sum() * pos_beta).item())  # 计算正常样本中位数的索引 n_idx
    # print("n_idx", n_idx)
    sorted_indices = torch.sort(normal_logps)[1]  # 对正常样本的对数似然值按升序进行排序
    
    n_idx = sorted_indices[n_idx]
    b_n = normal_logps[n_idx]  # normal boundary
    b_n = b_n / normalizer

    b_a = b_n - margin_tau  # abnormal boundary   计算异常边界 b_a

    return b_n, b_a


def calculate_bg_spp_loss(logps, lable, boundaries, normalizer=10, weights=None):
    """
    Calculate boudary guided semi-push-pull contrastive loss.
    Args:
        logps: log-likelihoods, shape (N, )
        lable: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: normal boundary and abnormal boundary
        正样本在负样本的右侧，两者有一定交叉
    """
    logps = logps / normalizer
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logps[lable == 0.0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]  # 分别在正常样本中筛选出位于边界值左侧的对数似然值。
    loss_n = b_n - normal_logps_inter  # 半拉损失是边界值与正常样本对数似然值之间的差
    # 随着训练的进行，loss_n 越来越小，表示模型对正常样本的概率估计逐渐向正常边界靠拢，
    # 也就是模型对正常样本的区分能力逐渐增强。这意味着模型在训练过程中更加自信地认为某些样本是正常的，
    # 而不会将正常样本过度分类为异常样本 


    b_a = boundaries[1]   # anomaly boundaries
    anomaly_logps = logps[lable == 1.0]    
    anomaly_logps_inter = anomaly_logps[anomaly_logps >= b_a]  # 分别在异常样本中筛选出位于边界值右侧的对数似然值。
    loss_a = anomaly_logps_inter - b_a  # 半推损失是异常样本对数似然值与边界值之间的差

    if weights is not None:
        nor_weights = weights[lable == 0.0][normal_logps <= b_n]
        loss_n = loss_n * nor_weights
        ano_weights = weights[lable == 1.0][anomaly_logps >= b_a]
        loss_a = loss_a * ano_weights
    
    loss_n = torch.mean(loss_n)
    loss_a = torch.mean(loss_a)

    return loss_n, loss_a


def calculate_bg_spp_loss_normal(logps, mask, boundaries, normalizer=10, weights=None):
    """
    Calculate boudary guided semi-push-pull contrastive loss.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: normal boundary and abnormal boundary
    """
    logps = logps / normalizer
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logps[mask == 0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]
    loss_n = b_n - normal_logps_inter

    if weights is not None:
        nor_weights = weights[mask == 0][normal_logps <= b_n]
        loss_n = loss_n * nor_weights
    
    loss_n = torch.mean(loss_n)

    return loss_n