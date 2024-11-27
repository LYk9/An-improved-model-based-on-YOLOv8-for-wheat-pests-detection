###################### PolarizedSelfAttention     ####     start    ###############################
import torch
import torch.nn as nn
from torch.nn import functional as F


class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM,self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        # print(x.size())
        b, c, h, w = x.size()

        n = w * h - 1
        # 计算每个元素与其平均值之间的平方差
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  # .pow(2)对张量中每个参数进行平方运算
        # 接下来是计算注意力权重。
        # x_minus_mu_square.sum(dim=[2, 3], keepdim=True) 计算每个通道中所有像素的平方差之和
        # 除以 (4 * (平方差之和 / n + self.e_lambda))，这是一个归一化过程，避免除零错误和调整权重
        # 最后加上 0.5
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # 将输入张量 x 与激活后的权重 y 相乘，返回加权后的结果。
        return x * self.activaton(y)

###################### PolarizedSelfAttention    ####     end    ###############################