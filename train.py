from ModelWithNormalProsition import ModelArgs as arg
from ModelWithNormalProsition import GPT
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
from NewsDataLoader import getABatch, getVocabSize
import time


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
def estimate_loss_acc(model, p):
    model.eval()
    losses = []
    # 验证集中计算损失
    for x, y in getABatch('val', p.batch_size, p.block_size):
        x = x.to(p.device)
        y = y.to(p.device)
        logits = model(x)
        # 将targets展平,取出所有的字符
        y = y.view(-1)
        # 交叉熵损失函数会先计算logits的softmax,确定类别,再进行标签比较
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
    val_loss = np.mean(losses)
    # 测试集中计算准确度
    test_acc = 0
    testCount = 0
    for x, y in getABatch('test', p.batch_size, p.block_size):
        x = x.to(p.device)
        y = y.to(p.device)
        testCount += p.batch_size   # 计算有多少测试集
        logits = model(x)
        y = y.view(-1)
        probabilities = F.softmax(logits, dim=-1)
        _, y_hat = probabilities.max(dim=1)
        acc = y == y_hat
        acc = acc.sum().item()
        test_acc += acc
    acc_score = test_acc / testCount
    model.train()
    return [val_loss, acc_score]

parameter = arg()
parameter.vocab_size = getVocabSize()
logger.info(f"词表大小{parameter.vocab_size}")
m = GPT(parameter)
m.to(parameter.device)
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
trainLosses = []
val_losses = []
test_accuracies = []
logger.info("start training")
print("start training")
for step in range(parameter.max_iter):
    trainLoss = []
    logger.info(f"step {step}")
    start_time = time.time()
    for X, Y in getABatch('train', parameter.batch_size, parameter.block_size):
        X, Y = X.to(parameter.device),Y.to(parameter.device)
        logits = m(X)
        # 将targets展平,取出所有的字符
        Y = Y.view(-1)
        # 交叉熵损失函数会先计算logits的softmax,确定类别,再进行标签比较
        loss = F.cross_entropy(logits, Y)
        trainLoss.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # 计算训练时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"step {step} complete! spend time is {elapsed_time} s")
    print(f"step {step} complete! spend time is {elapsed_time} s")
    # 训练一轮后,计算验证损失和测试准确度
    out = estimate_loss_acc(m, parameter)
    val_loss = out[0]
    test_acc = out[1]
    train_loss = np.mean(trainLoss)  # 每次迭代一轮,平均每一个批量的loss
    trainLosses.append(train_loss)  # 加入最终的训练loss表示
    val_losses.append(val_loss)  # 加入最终的验证loss表示
    test_accuracies.append(test_acc)  # 加入最终的测试准确度表示
    logger.info(f"step {step}: train loss {train_loss}, val loss {val_loss}, test acc {test_acc}")
    print(f"step {step}: train loss {train_loss}, val loss {val_loss}, test acc {test_acc}")


torch.save(m,'NewsModel.pth')
plt.plot(trainLosses,label='train_loss')
plt.plot(val_losses,label='val_loss')
plt.plot(test_accuracies,label='test_accuracy')
plt.xlabel('step')
plt.ylabel('loss')
plt.title('train loss & val loss & test accuracy')
plt.legend()

# 显示图形
plt.show()
plt.savefig("多分类问题训练结果.png")