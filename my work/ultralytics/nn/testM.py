
# 验证一下 CBAM通道注意力机制改为SE
# 还是task里的解析函数改好就行

import torch
from torch import nn

# 定义SEAttention模块
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 定义SpatialAttention模块（不变）
class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

# 修改后的CBAM模块，使用SEAttention替换掉Channel Attention
class testM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, reduction=16, kernel_size=7):
        """Initialize CBAM with given input channel (c1), reduction ratio, and kernel size."""
        super().__init__()
        self.channel_attention = SEAttention(channel=c1, reduction=reduction)  # 用SEAttention替换
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))

# 测试代码
if __name__ == '__main__':
    input_tensor = torch.randn(50, 512, 7, 7)  # 示例输入
    cbam = testM(c1=512, reduction=16)
    output = cbam(input_tensor)
    print(output.shape)
