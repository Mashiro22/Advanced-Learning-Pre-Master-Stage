#%% md
# # COVID-19 Cases Prediction
#%% md
# ## Step0 初始化 SwanLab
#%%
import swanlab

swanlab.init(project="COVID-19_Cases_Prediction")

#%% md
# ## Step1 数据预处理
#%%
import pandas as pd
import numpy as np

# 加载数据集
COVID_train = pd.read_csv("covid.train.csv")
COVID_test = pd.read_csv("covid.test.csv")

COVID_test
#%%
COVID_test.iloc[:, :-1].describe()
#%%
# 去除 ID
COVID_train_drop = COVID_train.drop(columns=["id"])
COVID_test_drop = COVID_test.drop(columns=["id"])

# 分离特征和标签
x_train = COVID_train_drop.iloc[:, 40:-1]
y_train = COVID_train_drop.iloc[:, -1]
x_test = COVID_test_drop.iloc[:, 40:]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 归一化特征
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 划分训练集与验证集
# 90% 训练集，10% 验证集
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.1,
    random_state=42,
)


# 三分类标签化
# tested_positive:(0,20)为 0
# tested_positive:(20,50)为 1
# tested_positive:(50,100)为 2
def one_hot(labels, num_classes=10):
    labels_split = np.where(
        labels < 5, 0,
        np.where(
            labels < 8, 1,
            np.where(
                labels < 11, 2,
                np.where(
                    labels < 13, 3,
                    np.where(
                        labels < 15, 4,
                        np.where(
                            labels < 18, 5,
                            np.where(
                                labels < 20, 6,
                                np.where(
                                    labels < 22, 7,
                                    np.where(
                                        labels < 25, 8, 9
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return np.eye(num_classes)[labels_split.astype(int).ravel()]


# 应用
# x_train_split["tested_positive"] = one_hot(x_train_split["tested_positive"], 3)
# x_train_split["tested_positive.1"] = one_hot(x_train_split["tested_positive.1"], 3)
y_train_split = one_hot(y_train_split)
# x_val["tested_positive"] = one_hot(x_val["tested_positive"], 3)
# x_val["tested_positive.1"] = one_hot(x_val["tested_positive.1"], 3)
y_val = one_hot(y_val)

y_val
#%% md
# ## Step2 定义 Sigmoid 函数和 RMSE 函数 (线性回归预测新冠率)
#%%
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)  # sigmoid 的导数


def rmse_loss(y_true, y_pred):
    """均方误差损失（适用于回归任务）"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
#%% md
# ## Step2 定义 Softmax 函数、 Cross Entropy 函数和 Dropout 函数 (多分类预测的病率范围)
#%%
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)  # 防止数值爆炸（稳定性技巧）
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12  # 防止 log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def dropout(x, p=0.2, dropout_act=True):
    if not dropout_act:
        return x  # 验证/测试时不做dropout
    mask = (np.random.rand(*x.shape) > p).astype(float)
    return x * mask / (1.0 - p)  # 反向缩放，保持期望一致

#%% md
# ## Step3 评估函数
#%%
# def predict(x, w, b):
#     probs = np.dot(x, w) + b
#     return probs

def predict(logits):
    probs = softmax(logits)
    return np.argmax(probs, axis=1)


def evaluate_accuracy(x, y, w, b, tolerance=1):
    y_pred = predict(x, w, b)
    return np.mean(abs(y_pred - y) <= tolerance)
#%% md
# ## Step4 训练函数 (线性回归)
#%%
def train_linear_regression(x, y, x_val=None, y_val=None, lr=0.01, epochs=1000):
    w = np.zeros(x.shape[1])
    b = 0

    for epoch in range(epochs + 1):
        y_pred = np.dot(x, w) + b
        # y_pred = sigmoid(z)

        # 损失
        loss = rmse_loss(y, y_pred)

        # 梯度
        dz = y_pred - y
        dw = np.dot(x.T, dz) / len(x)
        db = np.mean(dz)

        # 参数更新
        w -= lr * dw
        b -= lr * db

        if epoch % 5 == 0:
            train_acc = evaluate_accuracy(x, y, w, b)
            val_acc = evaluate_accuracy(x_val, y_val, w, b) if x_val is not None else None

        # ✅ Swanlab 记录
        swanlab.log({
            "train_loss": loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "epoch": epoch
        })

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w, b
#%% md
# ## Step4 训练函数 （三层 BP 神经网络）
#%%
def init_bp_network(input_dim, hidden_dim, output_dim):
    # 使用正态分布初始化参数
    w1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))

    w2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros((1, output_dim))

    return w1, b1, w2, b2


def forward(x, w1, b1, w2, b2, dropout_act):
    z1 = np.dot(x, w1) + b1  # (n, h)
    a1 = sigmoid(z1)  # (n, h)
    a1 = dropout(a1, p=0.035, dropout_act=dropout_act)
    z2 = np.dot(a1, w2) + b2  # (n, c)
    a2 = softmax(z2)  # (n, c)
    return z1, a1, z2, a2


def backward(x, y_true, z1, a1, z2, a2, w2):
    """
    x: 输入样本 (n, d)
    y_true: one-hot 标签 (n, c)
    z1, a1, z2, a2: 前向传播中得到的中间值
    w2: 第二层权重，用于反向传播
    """
    n = x.shape[0]

    # 输出层误差（softmax + cross-entropy）
    dz2 = (a2 - y_true) / n  # (n, c)
    dw2 = np.dot(a1.T, dz2)  # (h, c)
    db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, c)

    # 隐藏层误差（sigmoid导数 + 链式法则）
    da1 = np.dot(dz2, w2.T)  # (n, h)
    dz1 = da1 * sigmoid_derivative(a1)  # (n, h)
    dw1 = np.dot(x.T, dz1)  # (d, h)
    db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, h)

    return dw1, db1, dw2, db2


def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    return w1, b1, w2, b2


def train_bp_network(
        x_train, y_train,
        x_val=None, y_val=None,
        input_dim=53, hidden_dim=64, output_dim=3,
        lr=0.05, epochs=100, batch_size=64, verbose=True
):
    # 1. 初始化
    w1, b1, w2, b2 = init_bp_network(input_dim, hidden_dim, output_dim)
    history = {"train_acc": [], "val_acc": [], "loss": []}
    n = x_train.shape[0]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    for epoch in range(epochs + 1):
        indices = np.arange(n)
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]
        for i in range(0, n, batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            z1, a1, z2, a2 = forward(x_batch, w1, b1, w2, b2, dropout_act=True)
            dw1, db1, dw2, db2 = backward(x_batch, y_batch, z1, a1, z2, a2, w2)
            w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)

        if epoch % 50 == 0 or epoch == epochs - 1 or epoch == 1:
            _, _, z2_train, a2_train = forward(x_train, w1, b1, w2, b2, dropout_act=False)
            train_preds = predict(z2_train)
            train_labels = np.argmax(y_train, axis=1)
            train_acc = np.mean(train_preds == train_labels)

            if x_val is not None and y_val is not None:
                _, _, z2_val, _ = forward(x_val, w1, b1, w2, b2, dropout_act=False)
                val_preds = predict(z2_val)
                val_labels = np.argmax(y_val, axis=1)
                val_acc = np.mean(val_preds == val_labels)
            else:
                val_acc = None

        loss = cross_entropy_loss(y_train, a2_train)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["loss"].append(loss)

        # ✅ Swanlab 记录
        swanlab.log({
            "train_loss": loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "epoch": epoch,
        })

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {loss:.4f}")

    return w1, b1, w2, b2, history
#%% md
# ## Step5 训练模型
#%%
w1, b1, w2, b2, history = train_bp_network(
    x_train_split, y_train_split,
    x_val, y_val,
    input_dim=53,
    hidden_dim=64,
    output_dim=10,
    lr=0.13,
    epochs=8000,
    batch_size=64,
)
#%% md
# ## Step6 模型推理
#%%
def save_submission(x_test, w1, b1, w2, b2):
    _, _, z2_pred, a2_pred = forward(x_train, w1, b1, w2, b2, dropout_act=False)
    test_preds = predict(z2_pred)

    # test_df 中的 PassengerId 是测试集乘客的 ID
    submission = pd.DataFrame({
        "tested_positive.2": test_preds,
    })

    submission.to_csv("submission.csv", index=False)
    print("保存完成：submission.csv")


save_submission(x_test, w1, b1, w2, b2)