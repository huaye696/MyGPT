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
    block_size: int = 256  # 一个批量中包含的字符数
    dropout: int = 0.4
    device : str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
        # 生成旋转辐角矩阵
        self.freqs_cis = precompute_freqs_cis(arg.head_dim, arg.block_size).to(arg.device)
        self.batch_size = arg.batch_size
        self.block_size = arg.block_size
        self.head_dim = arg.head_dim

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        xk = k.view(self.batch_size, self.block_size, 1, self.head_dim)
        xq = q.view(self.batch_size, self.block_size, 1, self.head_dim)
        q, k = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis)
        q = q.flatten(2)
        k = k.flatten(2)
        weight = q @ k.transpose(-2, -1) * C**-0.5  # 得到权重，但是GPT不想看见未出现的字符权重
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  #  把上三角的等于0的位置，全部替换为-inf
        weight = F.softmax(weight, dim=-1)
        v = self.value(x)
        out = weight @ v
        return out


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):  # 初始化获得旋转辐角
    # 将每个注意力头的维度，两两分组，得到分组后需要旋转的辐角
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # t 与 freqs 做外积，也就是t中的每一个元素都和freqs中的每一个元素相乘，
    # 第一个维度是t的维度，第二个维度是freqs的维度
    # 这一步得到旋转角 m * theta ； size ——》 （block_size, 分组数）
    freqs = torch.outer(t, freqs).float()
    # torch.ones_like(freqs)创建了一个和freqs大小相同、但所有元素为 1 的张量。
    # torch.polar 创建复数形式，第一个是复数的模，第二个是辐角
    # 由于不能改变原向量的大小，只旋转，所以创建一个模为1的复数，并且辐角为旋转角
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim  # 获取x的总维度数
    assert 0 <= 1 < ndim
    # freqs_cis的第二维要是m，就是序列的长度，并且与传入的x保持一致，
    # 最后一维要是分组数，与传入的x保持一致，因为这两个维度不能动，需要在其他地方增加维度，以便进行广播
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:  # 传入 Q K ,返回一个元组
    # view_as_complex 把传入的向量转化到复数域
    # *xq.shape[:-1] 是指xq去掉最后一个维度，取前面的batch_size、block_size、n_heads
    # xq.float().reshape(*xq.shape[:-1], -1, 2) 后的维度为：
    # batch_size、block_size、n_heads、n_heads // 2、2
    # 最后一个维度一定要是2，因为是两两分组，分别作为转化后的虚数的实部和虚部
    # 做上述的转化后，xq_的形状变成了：
    # batch_size、block_size、n_heads、n_heads // 2
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(ModelArgs.device)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(ModelArgs.device)
    # 将freqs_cis做一个重塑，从（block_size, 分组数） -> (1, block_size, 1, 分组数)
    # 其实就是加了一个batch_size的维度，和分组前的头的维度，以便进行多头和多batch之间的广播
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # xq_ * freqs_cis 就是复数域的乘法，freqs_cis的模是1，相乘就是只旋转，不改变大小
    # 最后再由view_as_real将虚数张量当作实张量返回，也就是拆分实部和虚部
    # ——》size：（batch_size, block_size, n_heads, 分组数，2）；
    # 最后展平第三个维度，也就是变成了最开始的形状：（batch_size, block_size, n_heads, n_dims）
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # type_as代表将xq_out转换为与xq相同的数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)

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
            nn.ReLU(),
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
        # 可学习的位置编码
        # self.position_embedding_table = nn.Embedding(arg.block_size, arg.n_embadding)
        self.heads = nn.Sequential(
            Block(arg),
            Block(arg),
            Block(arg),
            nn.Dropout(arg.dropout)
        )
        self.l1 = nn.Linear(arg.n_embadding, 16)
        self.decice = arg.device

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embadding)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        # temp = tok_emb + pos_emb  # 自动广播
        temp = tok_emb  # 自动广播
        temp = self.heads(temp)[:,-1,:].reshape(B, -1)
        # (b_size, block_size, embadding_size)
        logits = self.l1(temp)
        return logits