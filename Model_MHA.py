from dataclasses import dataclass
from typing import Tuple
import math
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    n_embadding: int = 128  # 嵌入维度
    # 注意力相关参数
    n_heads: int = 4  # 注意力头
    head_dim: int = n_embadding // n_heads  # 每个注意力头的维度
    vocab_size: int = -1  # 词表大小
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    batch_size: int = 32  # 一个批量大小
    block_size: int = 256  # 一个批量中包含的字符数
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

class MultiHeadAttention(nn.Module):
    def __init__(self, arg: ModelArgs):
        super().__init__()
        self.heads = arg.n_heads
        self.head_dim = arg.head_dim
        self.Wq = nn.Linear(arg.n_embadding, arg.n_embadding)
        self.Wk = nn.Linear(arg.n_embadding, arg.n_embadding)
        self.Wv = nn.Linear(arg.n_embadding, arg.n_embadding)
        self.Wo = nn.Linear(arg.n_embadding, arg.n_embadding)
        self.freqs_cis = precompute_freqs_cis(arg.head_dim, arg.block_size).to(arg.device)
        self.dropout = nn.Dropout(arg.dropout)

    def forward(self, x, effective_len):
        bsz, bls, _ = x.shape
        _queries = self.Wq(x)  # (bsz, bls, n_embadding)
        _keys = self.Wk(x)
        _values = self.Wv(x)

        queries = _queries.view(bsz, bls, self.heads, self.head_dim)  # (bsz, bls, n_heads, head_dim)
        keys = _keys.view(bsz, bls, self.heads, self.head_dim)
        values = _values.view(bsz, bls, self.heads, self.head_dim)

        queries, keys = apply_rotary_emb(queries, keys, freqs_cis=self.freqs_cis)  # 应用旋转位置编码

        queries = queries.transpose(1, 2)  # (bsz, head_size, bls, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)  # (bsz, head_size, bls, bls)

        mask_look_ahead = torch.full((1, 1, bls, bls), float("-inf"), device=x.device)  # (1,1,bls,bls)的全-inf矩阵
        mask_look_ahead = torch.triu(mask_look_ahead, diagonal=1).type_as(x)  # 对角线以上的元素保持不变，以下清零
        scores = scores + mask_look_ahead

        if effective_len is not None:
            mask_pad = torch.full(scores.shape, float(0), device=x.device)
            for batch, batch_effective_len in enumerate(effective_len):
                mask_pad[batch, :, :, bls - batch_effective_len:] = float("-inf")
            scores = scores + mask_pad

        scores = F.softmax(scores.float(), dim=-1).type_as(x)  # (bsz, head_size, bls, bls)
        output = torch.matmul(scores, values)  # (bsz, head_size, bls, head_dim)
        output = output.transpose(1, 2)  # (bsz, bls, head_size, head_dim)
        output = output.contiguous().view(bsz, bls, -1)
        return self.dropout(self.Wo(output))

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
        x = self.l1(x + self.heads(x, None))
        x = self.l2(x + self.fb(x))
        return x

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

class GPT(nn.Module):
    def __init__(self, arg: ModelArgs):
        super().__init__()
        self.token_embedding_table = nn.Embedding(arg.vocab_size, arg.n_embadding)
        # 可学习的位置编码
        self.heads = nn.Sequential(
            Block(arg),
            Block(arg),
            Block(arg),
            nn.Dropout(arg.dropout)
        )
        self.l1 = nn.Linear(arg.n_embadding, 14)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embadding)
        temp = self.heads(tok_emb)  # (B, T, n_embadding)
        token_last_feature = temp[:,-1,:].reshape(B, -1)
        logits = self.l1(token_last_feature)
        return logits