from model import ModelArgs as arg
from model import GPT
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from data import getDataLoader, getVocabSize
import matplotlib.pyplot as plt
import logging
from NewsDataLoader import getABatch, getVocabSize


# 创建logger对象
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# 创建FileHandler并设置日志格式、保存路径等参数
file_handler = logging.FileHandler('log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加FileHandler到logger对象
logger.addHandler(file_handler)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = []
    for x, y in getABatch('val', arg.batch_size, arg.block_size):
        x = x.to(arg.device)
        y = y.to(arg.device)
        logits = model(x)
        # 将targets展平,取出所有的字符
        y = y.view(-1)
        # 交叉熵损失函数会先计算logits的softmax,确定类别,再进行标签比较
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
    out = np.mean(losses)
    model.train()
    return out

a = arg()
a.vocab_size = getVocabSize()
logger.info(f"词表大小{a.vocab_size}")
m = GPT(a)
m.to(arg.device)
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
trainLosses = []
val_losses = []
count = 0
logger.info("start training")
for step in range(arg.max_iter):
    trainLoss = []
    logger.info(f"The step is {step}")
    for X, Y in getABatch('train', arg.batch_size, arg.block_size):
        X, Y = X.to(arg.device),Y.to(arg.device)
        logits = m(X)
        # 将targets展平,取出所有的字符
        Y = Y.view(-1)
        # 交叉熵损失函数会先计算logits的softmax,确定类别,再进行标签比较
        loss = F.cross_entropy(logits, Y)
        trainLoss.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if step != -1:
        val_loss = estimate_loss(m)
        t_loss = np.mean(trainLoss)
        trainLosses.append(t_loss)
        val_losses.append(val_loss)
        count += 1
        logger.info(f"step{step}: train loss {t_loss}, val loss {val_loss}")


torch.save(m,'NewsModel.pth')
plt.plot(trainLosses,label='train_loss')
plt.plot(val_losses,label='val_loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.title('train and val loss')
plt.legend()

# 显示图形
plt.show()
plt.savefig("多分类问题训练损失.png")