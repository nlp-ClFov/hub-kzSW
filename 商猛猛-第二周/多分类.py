import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y


def build_dataset(size):
    X = []
    Y = []
    for i in range(size):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


class TorchModel(torch.nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        # 5维输入，5维输出
        self.linear = nn.Linear(5, 5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def main():
    # 配置参数
    epoch_num = 100
    batch_size = 32
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001
    log = []
    # 建立模型
    model = TorchModel()
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建训练集,5000条
    tran_x, tran_y = build_dataset(train_sample)
    # 训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = tran_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = tran_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def evaluate(model):
    model.eval()
    test_sample = 100
    test_x, test_y = build_dataset(test_sample)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(test_x)
        y_pred = torch.argmax(y_pred, dim=1)
        for y_p, y_t in zip(y_pred, test_y):
            if y_p.item() == y_t.item():
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = TorchModel()
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        probs = torch.softmax(result, dim=1)
    for vec, prob in zip(input_vec, probs):
        pred_class = torch.argmax(prob).item()
        pred_prob = prob[pred_class].item()
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred_class, pred_prob))  # 打印结果


if __name__ == '__main__':
    # main()
    test_vec = [[0.88889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.99349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]

    predict("model.bin", test_vec)
