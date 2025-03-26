import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.networks.backbone import build_feature_extractor, NET_OUT_DIM
import timm



# 注意力网络结构，用于给输入的patch赋权重值，输入是512*2*2，学习注意力权重值
class AttentionNet(nn.Module):
    # patch是7*7的大小
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=200, out_channels=200, kernel_size=2, padding=0)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = nn.Conv2d(in_channels=200, out_channels=512, kernel_size=2, padding=0)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 1)
        self.softmax = nn.Softmax()

    # # 尝试一下patch是2*2的大小
    # def __init__(self):
    #     super(AttentionNet, self).__init__()
    #     self.Conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0)
    #     self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.Conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0)
    #     self.relu = nn.ReLU()
    #     self.fc1 = nn.Linear(4096, 256)
    #     self.fc2 = nn.Linear(256, 1)
    #     self.softmax = nn.Softmax()

    def forward(self, x):
        # print("x1", x.shape)
        x = self.Conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.Conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        # print("x2", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return float(x)

# 用于将特征提取后的feature map划分为512*2*2的patch,
# 并将注意力权重值乘以patch的feature map，最后再组合导一起送到网络中计算各类head
def AttentionModule(input_features):
    # print("the size of input_features in AttentionModule", input_features.shape)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    input_features = input_features.to(device)

    # 输入特征图的大小
    batch_size, num_channels, height, width = input_features.size()

    # 每个通道 patch 的大小
    # patch_size = (7, 7)
    patch_size = (7, 7)
    # print("input_features.shape", input_features.shape)

    # 划分的行数和列数
    num_rows = height // patch_size[0]
    num_cols = width // patch_size[1]

    # 划分后的 patch 数量
    num_patches = num_rows * num_cols

    # 创建一个空的张量来存储划分后的 patch
    patches = input_features.view(batch_size, num_channels, num_rows, patch_size[0], num_cols, patch_size[1])
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(batch_size, num_patches, num_channels, *patch_size)

    # 输出划分后的 patch 张量,这个张量patches.shape torch.Size([batch_size, num_patches, num_channels, patch_height, patch_width])
    # print("patches.shape", patches.shape)
    # print("切分patch结束")

    # 创建一个空的张量来存储注意力权重乘积后的 patch
    weighted_patches = torch.empty_like(patches)

    attention_net = AttentionNet().to(device)

    # 乘以权重_注意力网络得出

    for batch in range(batch_size):
        for i in range(num_patches):
            patch = patches[batch, i]
            # 获取权重值
            attention_output = attention_net(patch.unsqueeze(0))
            weighted_patch = patch * attention_output
            weighted_patches[batch, i] = weighted_patch

    # 重新组合 patch
    output_features = weighted_patches.view(batch_size, num_rows, num_cols, num_channels, *patch_size)
    output_features = output_features.permute(0, 3, 1, 4, 2, 5).contiguous()
    output_features = output_features.view(batch_size, num_channels, height, width)

    return output_features


class DRA(nn.Module):
    def __init__(self, args, backbone="resnet18"):

        super(DRA, self).__init__()
        self.args = args    # 保存模型的配置信息
        self.feature_extractor = build_feature_extractor(backbone, args)  # 首先使用resnet18的卷积部分做特征提取，其输出dim为512
        self.in_c = NET_OUT_DIM[backbone]    # 输入特征图的通道数512
        print("in_c", self.in_c)

    def forward(self, image, ref_image, label):  # 这边图像输入的大小是448*448的

        # 图片特征提取，参考图片特征提取
        feature = self.feature_extractor(image)
        ref_feature = self.feature_extractor(ref_image)
        # 图片特征提取后经过注意力网络，参考图片特征提取后经过注意力网络
        feature = AttentionModule(feature)
        ref_feature = AttentionModule(ref_feature)

        comparison_scores = self.composite_head(feature, ref_feature)


        return comparison_scores

