import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：MobileNet的核心组件"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 深度卷积：每个输入通道单独卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        # 点卷积：1×1卷积融合通道信息
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        # return self.relu(x)
        return self.gelu(x)


class TransformerEncoder(nn.Module):
    """Transformer编码器：包含多头注意力和前馈网络"""

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        # 多头自注意力
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 注意力模块
        attn_output, _ = self.attn(x, x, x)  # q, k, v
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # 前馈网络
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.norm2(x)
        return x


class ClassAwareAttention(nn.Module):
    """类别感知注意力：为不同类别特征分配权重"""

    def __init__(self, dim, num_classes):
        super().__init__()
        self.class_weights = nn.Parameter(torch.randn(num_classes, dim))  # 类别权重矩阵

    def forward(self, x):
        # x: (seq_len, batch_size, dim)
        # 计算每个token与类别权重的相似度
        class_sim = torch.matmul(x, self.class_weights.t())  # (seq_len, batch_size, num_classes)
        class_attn = F.softmax(class_sim, dim=0)  # 沿序列长度维度归一化
        # 加权求和：为每个类别特征分配权重
        weighted_x = torch.sum(class_attn.unsqueeze(-1) * x.unsqueeze(2), dim=0)  # (batch_size, num_classes, dim)
        return weighted_x


class MobileNetViT(nn.Module):
    """改进的MobileNetViT模型：适配交通标志检测"""

    def __init__(self, num_classes=5, dim=64, num_heads=4, mlp_dim=128, num_transformer_layers=2):
        super().__init__()
        self.num_classes = num_classes

        # 1. 卷积特征提取模块
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            # nn.ReLU()
            nn.GELU()
        )
        # 深度可分离卷积层
        self.conv_layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 64, stride=1),
            # 改进：新增1×1卷积增强特征表达
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            # nn.ReLU()
            nn.GELU()
        )

        # 2. Transformer全局特征模块
        self.patch_size = 8  # 将32×32特征图分割为4×4个8×8的patch
        self.num_patches = (32 // self.patch_size) ** 2  # 4×4=16个patch
        # Patch嵌入：将8×8×64的patch转换为64维向量
        self.patch_embed = nn.Conv2d(64, dim, kernel_size=self.patch_size, stride=self.patch_size)
        # 类别令牌：用于最终分类
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        # Transformer编码器
        self.transformer = nn.Sequential(
            *[TransformerEncoder(dim, num_heads, mlp_dim) for _ in range(num_transformer_layers)]
        )
        # 改进：类别感知注意力
        self.class_aware_attn = ClassAwareAttention(dim, num_classes)

        # 3. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(dim, 32),
            # nn.ReLU(),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # 输入：(batch_size, 3, 64, 64)
        batch_size = x.shape[0]

        # 1. 卷积特征提取
        x = self.conv_stem(x)  # (batch_size, 32, 32, 32)
        x = self.conv_layers(x)  # (batch_size, 64, 32, 32)

        # 2. Patch嵌入
        x = self.patch_embed(x)  # (batch_size, 64, 4, 4)
        x = x.flatten(2)  # (batch_size, 64, 16)
        x = x.transpose(1, 2)  # (batch_size, 16, 64)

        # 添加类别令牌
        class_tokens = self.class_token.expand(batch_size, -1, -1)  # (batch_size, 1, 64)
        x = torch.cat([class_tokens, x], dim=1)  # (batch_size, 17, 64)

        # 添加位置编码
        x = x + self.pos_embed  # (batch_size, 17, 64)

        # Transformer处理（需转换为(seq_len, batch_size, dim)）
        x = x.transpose(0, 1)  # (17, batch_size, 64)
        x = self.transformer(x)  # (17, batch_size, 64)

        # 类别感知注意力
        x = self.class_aware_attn(x)  # (batch_size, num_classes, 64)

        # 3. 分类
        x = x.mean(dim=1)  # (batch_size, 64)：平均每个类别的特征
        logits = self.classifier(x)  # (batch_size, num_classes)
        return logits


# 对比模型：VGG16
class VGG16(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = models.vgg16(pretrained=False).features
        # 适配64×64输入
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            # nn.ReLU(),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            # nn.ReLU(),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# # 对比模型：ResNet18
# class ResNet18(nn.Module):
#     def __init__(self, num_classes=5):
#         super().__init__()
#         self.resnet = models.resnet18(pretrained=False)
#         # 替换最后一层以适配5分类
#         self.resnet.fc = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         return self.resnet(x)

from torchvision.models.resnet import ResNet, BasicBlock
# class GELUResNet18(ResNet):
#     def __init__(self, num_classes=5, **kwargs):
#         # 继承ResNet基类，并指定使用BasicBlock（ResNet18的基础块）
#         super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
#
#         # 遍历模型的所有子模块，将ReLU替换为GELU
#         for module in self.modules():
#             if isinstance(module, nn.ReLU):
#                 # 注意：inplace=True的ReLU不能直接替换，需要先处理
#                 # 但ResNet中的ReLU默认inplace=False，所以可以直接替换
#                 parent_module = None
#                 # 找到父模块以便替换
#                 for name, m in self.named_modules():
#                     if m == module:
#                         # 获取父模块的引用（这里简化处理，实际中可能需要更复杂的查找）
#                         # 我们通过遍历所有模块来找到并替换
#                         break
#
#                 # 更直接和安全的方式是遍历并替换
#         # 一个更简洁的递归替换方法
#         self._replace_relu_with_gelu(self)
#
#     def _replace_relu_with_gelu(self, module):
#         for name, child_module in module.named_children():
#             if isinstance(child_module, nn.ReLU):
#                 # 将ReLU替换为GELU
#                 setattr(module, name, nn.GELU())
#             else:
#                 # 递归处理子模块
#                 self._replace_relu_with_gelu(child_module)
#
#
# 现在，你的对比模型可以更新为这个新版本
# class ResNet18(nn.Module):
#     def __init__(self, num_classes=5):
#         super().__init__()
#         # 使用我们自定义的GELU版ResNet18
#         self.resnet = ResNet18(num_classes=num_classes)
#
#     def forward(self, x):
#         return self.resnet(x)


from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn

# 1. GELU版ResNet18（真正的实现，无递归）
class GELUResNet18(ResNet):
    def __init__(self, num_classes=5, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
        self._replace_relu_with_gelu(self)

    def _replace_relu_with_gelu(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.GELU())
            else:
                self._replace_relu_with_gelu(child)

# 2. 封装ResNet18（调用GELUResNet18，而非自身）
class ResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 核心修复：实例化GELUResNet18，停止递归
        self.resnet = GELUResNet18(num_classes=num_classes)
        # self.resnet = ResNet18(num_classes=num_classes)

    def forward(self, x):
        return self.resnet(x)

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models
#
#
# class DepthwiseSeparableConv(nn.Module):
#     """深度可分离卷积：MobileNet的核心组件"""
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         # 深度卷积：每个输入通道单独卷积
#         self.depthwise = nn.Conv2d(
#             in_channels, in_channels, kernel_size=3,
#             stride=stride, padding=1, groups=in_channels, bias=False
#         )
#         # 点卷积：1×1卷积融合通道信息
#         self.pointwise = nn.Conv2d(
#             in_channels, out_channels, kernel_size=1,
#             stride=1, padding=0, bias=False
#         )
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.relu(x)
#
#
# class TransformerEncoder(nn.Module):
#     """Transformer编码器：包含多头注意力和前馈网络"""
#
#     def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
#         super().__init__()
#         # 多头自注意力
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
#         # 前馈网络
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_dim),
#             # nn.RELU(),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_dim, dim),
#             nn.Dropout(dropout)
#         )
#         # 层归一化
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # 注意力模块
#         attn_output, _ = self.attn(x, x, x)  # q, k, v
#         x = x + self.dropout(attn_output)
#         x = self.norm1(x)
#         # 前馈网络
#         mlp_output = self.mlp(x)
#         x = x + self.dropout(mlp_output)
#         x = self.norm2(x)
#         return x
#
#
# class ClassAwareAttention(nn.Module):
#     """类别感知注意力：为不同类别特征分配权重"""
#
#     def __init__(self, dim, num_classes):
#         super().__init__()
#         self.class_weights = nn.Parameter(torch.randn(num_classes, dim))  # 类别权重矩阵
#
#     def forward(self, x):
#         # x: (seq_len, batch_size, dim)
#         # 计算每个token与类别权重的相似度
#         class_sim = torch.matmul(x, self.class_weights.t())  # (seq_len, batch_size, num_classes)
#         class_attn = F.softmax(class_sim, dim=0)  # 沿序列长度维度归一化
#         # 加权求和：为每个类别特征分配权重
#         weighted_x = torch.sum(class_attn.unsqueeze(-1) * x.unsqueeze(2), dim=0)  # (batch_size, num_classes, dim)
#         return weighted_x
#
#
# class MobileNetViT(nn.Module):
#     """改进的MobileNetViT模型：适配交通标志检测"""
#
#     def __init__(self, num_classes=5, dim=64, num_heads=4, mlp_dim=128, num_transformer_layers=2):
#         super().__init__()
#         self.num_classes = num_classes
#
#         # 1. 卷积特征提取模块
#         self.conv_stem = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
#         # 深度可分离卷积层
#         self.conv_layers = nn.Sequential(
#             DepthwiseSeparableConv(32, 64, stride=1),
#             DepthwiseSeparableConv(64, 64, stride=1),
#             # 改进：新增1×1卷积增强特征表达
#             nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#
#         # 2. Transformer全局特征模块
#         self.patch_size = 8  # 将32×32特征图分割为4×4个8×8的patch
#         self.num_patches = (32 // self.patch_size) ** 2  # 4×4=16个patch
#         # Patch嵌入：将8×8×64的patch转换为64维向量
#         self.patch_embed = nn.Conv2d(64, dim, kernel_size=self.patch_size, stride=self.patch_size)
#         # 类别令牌：用于最终分类
#         self.class_token = nn.Parameter(torch.randn(1, 1, dim))
#         # 位置编码
#         self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
#         # Transformer编码器
#         self.transformer = nn.Sequential(
#             *[TransformerEncoder(dim, num_heads, mlp_dim) for _ in range(num_transformer_layers)]
#         )
#         # 改进：类别感知注意力
#         self.class_aware_attn = ClassAwareAttention(dim, num_classes)
#
#         # 3. 分类头
#         self.classifier = nn.Sequential(
#             nn.Linear(dim, 32),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(32, num_classes)
#         )
#
#     def forward(self, x):
#         # 输入：(batch_size, 3, 64, 64)
#         batch_size = x.shape[0]
#
#         # 1. 卷积特征提取
#         x = self.conv_stem(x)  # (batch_size, 32, 32, 32)
#         x = self.conv_layers(x)  # (batch_size, 64, 32, 32)
#
#         # 2. Patch嵌入
#         x = self.patch_embed(x)  # (batch_size, 64, 4, 4)
#         x = x.flatten(2)  # (batch_size, 64, 16)
#         x = x.transpose(1, 2)  # (batch_size, 16, 64)
#
#         # 添加类别令牌
#         class_tokens = self.class_token.expand(batch_size, -1, -1)  # (batch_size, 1, 64)
#         x = torch.cat([class_tokens, x], dim=1)  # (batch_size, 17, 64)
#
#         # 添加位置编码
#         x = x + self.pos_embed  # (batch_size, 17, 64)
#
#         # Transformer处理（需转换为(seq_len, batch_size, dim)）
#         x = x.transpose(0, 1)  # (17, batch_size, 64)
#         x = self.transformer(x)  # (17, batch_size, 64)
#
#         # 类别感知注意力
#         x = self.class_aware_attn(x)  # (batch_size, num_classes, 64)
#
#         # 3. 分类
#         x = x.mean(dim=1)  # (batch_size, 64)：平均每个类别的特征，平均池化
#         logits = self.classifier(x)  # (batch_size, num_classes)
#         return logits
#
#
# # 对比模型：VGG16
# class VGG16(nn.Module):
#     def __init__(self, num_classes=5):
#         super().__init__()
#         self.features = models.vgg16(pretrained=False).features
#         # 适配64×64输入
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 2 * 2, 4096),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
#
#
# # 对比模型：ResNet18
# class ResNet18(nn.Module):
#     def __init__(self, num_classes=5):
#         super().__init__()
#         self.resnet = models.resnet18(pretrained=False)
#         # 替换最后一层以适配5分类
#         self.resnet.fc = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         return self.resnet(x)

