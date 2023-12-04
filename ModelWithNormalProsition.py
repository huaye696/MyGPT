from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    n_embadding: int = 360  # 嵌入维度
    # 注意力相关参数
    n_heads: int = 4  # 注意力头
    head_dim: int = n_embadding // n_heads  # 每个注意力头的维度
    vocab_size: int = -1  # 词表大小
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    batch_size: int = 32  # 一个批量大小
    block_size: int = 64  # 一个批量中包含的字符数
    dropout: int = 0.4
    device : str = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    max_iter: int = 10


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Head(nn.Module):
    """ 单个注意力头 """
    def __init__(self, arg: ModelArgs):
        super(Head, self).__init__()
        self.key = nn.Linear(arg.n_embadding, arg.head_dim, bias=False)  # 当前这个词是一个什么
        self.value = nn.Linear(arg.n_embadding, arg.head_dim, bias=False)  # 当前这个值需要与其他值交互的一个内容
        self.query = nn.Linear(arg.n_embadding, arg.head_dim, bias=False)  # 当前词正在寻找什么，比如“我是一个元音，我在找一个辅音”
        # 生成下三角矩阵，命名为tril，将其存放于torch的缓存区中，用于长期存储，但是不会反向传播，不受参数优化
        self.register_buffer('tril', torch.tril(torch.ones(arg.block_size, arg.block_size)))
        self.dropout = nn.Dropout(arg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1) * C**-0.5  # 得到权重，但是GPT不想看见未出现的字符权重
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  #  把上三角的等于0的位置，全部替换为-inf
        weight = F.softmax(weight, dim=-1)
        v = self.value(x)
        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    """ 多头注意力 """
    def __init__(self, arg: ModelArgs):
        super().__init__()
        self.head = nn.ModuleList(Head(arg) for _ in range(arg.n_heads))
        self.linear = nn.Linear(arg.n_embadding, arg.n_embadding)
        self.dropout = nn.Dropout(arg.dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.head], dim=-1)
        output = self.dropout(self.linear(output))
        return output

class FeedBlock(nn.Module):
    def __init__(self, arg: ModelArgs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(arg.n_embadding, 4 * arg.n_embadding),
            nn.SiLU(),
            nn.Linear(4 * arg.n_embadding, arg.n_embadding),
            nn.Dropout(arg.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, arg: ModelArgs):
        super().__init__()
        self.heads = MultiHeadAttention(arg)
        self.fb = FeedBlock(arg)
        self.l1 = RMSNorm(arg.n_embadding)
        self.l2 = RMSNorm(arg.n_embadding)
        self.dropout = nn.Dropout(arg.dropout)

    def forward(self, x):
        x = self.l1(x + self.heads(x))
        x = self.l2(x + self.fb(x))
        return x

class GPT(nn.Module):
    def __init__(self, arg: ModelArgs):
        super().__init__()
        self.token_embedding_table = nn.Embedding(arg.vocab_size, arg.n_embadding)
        self.token_position_table = nn.Embedding(arg.block_size, arg.n_embadding)
        # 可学习的位置编码
        self.heads = nn.Sequential(
            Block(arg),
            Block(arg),
            Block(arg),
            nn.Dropout(arg.dropout)
        )
        self.device = arg.device
        self.l1 = nn.Linear(arg.n_embadding, 14)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embadding)
        pos_emb = self.token_position_table(torch.arange(T, device=self.device))
        temp = tok_emb + pos_emb # 自动广播
        temp = self.heads(temp)
        temp = self.heads(temp)[:,-1,:].reshape(B, -1)
        logits = self.l1(temp)
        return logits