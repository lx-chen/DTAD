import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class DeviationLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        confidence_margin = 5.0
        ref = torch.normal(mean=0., std=torch.full([5000], 1.)).cuda()  # 满足以 0.0 为均值、1.0 为标准差的正态分布
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)   # 与标准正态分布的差异
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        # print("y_pred.shape", y_pred.shape)
        # print("y_pred", y_pred)
        # print("inlier_loss.shape", inlier_loss.shape)
        # print("inlier_loss", inlier_loss)
        # print("outlier_loss.shape", outlier_loss.shape)
        # print("outlier_loss", outlier_loss)

        dev_loss = (1 - y_true) * inlier_loss + y_true * outlier_loss  # 一个分类损失，会让正样本的inlier_loss增大，outlier_loss减小，而让负本的inlier_loss增小，outlier_loss减小，
        return torch.mean(dev_loss)


# # 这个是Arcface的loss计算
# class DeviationLoss(nn.Module):
#     def __init__(self, num_classes = 1, in_features = 4, s=30.0, m=0.5):
#         super(DeviationLoss, self).__init__()
#         self.num_classes = num_classes
#         self.in_features = in_features
#         self.s = s  # 尺度参数
#         self.m = m  # 角度间隔（margin）
#
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
#         nn.init.xavier_uniform_(self.weight)  # 初始化权重
#
#     def forward(self, features, targets):
#         print("features", features)
#         print("targets", targets)
#         features = features.view(-1, 1)
#         # 特征向量归一化
#         # features = F.normalize(features, dim=1)
#         print("features = F.normalize", features.shape)
#         weights = F.normalize(self.weight.t(), dim=1)
#         print("weights", weights.shape)
#         # 计算特征向量与权重向量的夹角
#         cos_theta = F.linear(features, weights)
#         theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))  # 夹角范围限制在 [0, pi]
#
#         # 获取目标类别的权重向量
#         one_hot = torch.zeros_like(cos_theta)
#         one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
#         output = cos_theta * 1.0  # 复制 cos_theta 的值作为输出
#         output += one_hot * (torch.cos(theta + self.m) - cos_theta) * self.s  # 更新输出
#
#         return output


# class DeviationLoss(nn.Module):
#     r"""Implement of large margin arc distance: :
#         Args:
#           in_features: size of each input sample
#           out_features: size of each output sample
#           s: norm of input feature
#           m: additive angular margin
#           cos(theta + m)
#       """
#     def __init__(self, in_features=512, out_features=1, s=30.0, m=0.50, easy_margin=False):
#         super(DeviationLoss, self).__init__()
#
#         self.in_features = in_features  # 特征输入通道数
#         self.out_features = out_features  # 特征输出通道数
#         self.s = s  # 输入特征范数 ||x_i||
#         self.m = m  # 加性角度边距 m (additive angular margin)
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))  # FC 权重
#         nn.init.xavier_uniform_(self.weight)  # Xavier 初始化 FC 权重
#
#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m
#
#
#     def forward(self, input, label):
#         print("input", input.shape)
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         # 分别归一化输入特征 xi 和 FC 权重 W, 二者点乘得到 cosθ, 即预测值 Logit
#         # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         cosine = F.linear(input, self.weight)
#         # 由 cosθ 计算相应的 sinθ
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         # 展开计算 cos(θ+m) = cosθ*cosm - sinθ*sinm, 其中包含了 Target Logit (cos(θyi+ m)) (由于输入特征 xi 的非真实类也参与了计算, 最后计算新 Logit 时需使用 One-Hot 区别)
#         phi = cosine * self.cos_m - sine * self.sin_m
#         # 是否松弛约束??
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#
#         # --------------------------- convert label to one-hot ---------------------------
#         # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
#         # 将 labels 转换为独热编码, 用于区分是否为输入特征 xi 对应的真实类别 yi
#         one_hot = torch.zeros(cosine.size(), device='cuda')
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         # 计算新 Logit
#         #  - 只有输入特征 xi 对应的真实类别 yi (one_hot=1) 采用新 Target Logit cos(θ_yi + m)
#         #  - 其余并不对应输入特征 xi 的真实类别的类 (one_hot=0) 则仍保持原 Logit cosθ_j
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # can use torch.where if torch.__version__  > 0.4
#         # 使用 s rescale 放缩新 Logit, 以馈入传统 Softmax Loss 计算
#         output *= self.s
#
#         return output
