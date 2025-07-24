#%% md
# # Titanic - Machine Learning from Disaster
#%% md
# ## Step0 初始化 SwanLab
#%%
import swanlab

swanlab.init(project="Titanic-Machine_Learning_from_Disaster")

#%% md
# ## Step1 数据预处理
#%%
import pandas as pd
import numpy as np

# 读取训练集和测试集
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')
#%%
titanic_train.describe()
#%%
# 训练集中 Age 与 Embarked 无效项的填充
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())
titanic_train["Fare"] = titanic_train["Fare"].fillna(titanic_train["Fare"].median())
titanic_train["Embarked"] = titanic_train["Embarked"].fillna(titanic_train["Embarked"].mode()[0])

# 测试集中 Age 与 Embarked 无效项的填充
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test["Embarked"] = titanic_test["Embarked"].fillna(titanic_test["Embarked"].mode()[0])

titanic_train.describe()
#%%
titanic_train
#%%
# 替换性别为编码：male=0,female=1
titanic_train["Sex"] = titanic_train["Sex"].map({"male": 0, "female": 1})
titanic_test["Sex"] = titanic_test["Sex"].map({"male": 0, "female": 1})

# 替换上船港口为编码
embarked_map = {"S": 0, "C": 1, "Q": 2}
titanic_train["Embarked"] = titanic_train["Embarked"].map(embarked_map)
titanic_test["Embarked"] = titanic_test["Embarked"].map(embarked_map)

titanic_test
#%%
selected_features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]

titanic_train["FamilySize"] = titanic_train["SibSp"] + titanic_train["Parch"] + 1
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"] + 1

x_train = titanic_train[selected_features]
y_train = titanic_train["Survived"].values

x_test = titanic_test[selected_features]

# 划分训练集与验证集
from sklearn.model_selection import train_test_split

x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

# 特征标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_split)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

x_train
#%% md
# ## Step2 实现 sigmoid 和 loss 函数
#%%
def sigmoid(z):
    z = np.clip(z, -500, 500)  # 限制输入范围，避免溢出
    return 1 / (1 + np.exp(-z))


def compute_loss(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)  # 避免log(0)
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
#%% md
# ## Step3 模型验证
#%%
def predict(x, weights, b):
    probs = sigmoid(np.dot(x, weights) + b)
    return (probs >= 0.7).astype(int)


def accuracy(x, y, weights, b):
    y_pred = predict(x, weights, b)
    accuracy = np.mean(y_pred == y)
    return accuracy
#%% md
# ## Step4 训练函数
#%%
def train_logistic_regression(x, y, lr=0.001, epochs=100):
    m, n = x.shape
    weights = np.zeros(n)
    b = 0

    for epoch in range(epochs):
        # 前向传播
        z = np.dot(x, weights) + b
        y_pred = sigmoid(z)

        # 损失
        loss = compute_loss(y, y_pred)

        # 梯度
        dz = y_pred - y
        dw = np.dot(x.T, (dz)) / m
        db = np.mean(dz)

        # 更新参数
        weights -= lr * dw
        b -= lr * db

        if epoch % 1000 == 0 or epoch == epochs - 1:
            train_accuracy = accuracy(x_train_scaled, y_train_split, weights, b)
            val_accuracy = accuracy(x_val_scaled, y_val, weights, b)
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")

        # ✅ Swanlab 记录
        swanlab.log({
            "train_loss": loss,
            "train_acc": train_accuracy,
            "val_acc": val_accuracy,
            "epoch": epoch
        })

        # 打印部分训练过程与验证
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.4f}\n")

    return weights, b
#%% md
# ## Step5 训练模型
#%%
weights, b = train_logistic_regression(x_train_scaled, y_train_split, lr=0.01, epochs=12000)
#%% md
# ## Step6 模型推理
#%%
test_preds = predict(x_test_scaled, weights, b)

# test_df 中的 PassengerId 是测试集乘客的 ID
submission = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": test_preds
})

submission.to_csv("submission.csv", index=False)
print("保存完成：submission.csv")