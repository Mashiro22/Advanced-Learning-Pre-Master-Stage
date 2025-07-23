#%% md
# ## Titanic - Machine Learning from Disaster
#%% md
# ## Step1 数据预处理
#%%
import pandas as pd

# 读取训练集和测试集
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')

train
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
selected_features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]

x_train = titanic_train[selected_features]
y_train = titanic_train["Survived"].values

x_test = titanic_test[selected_features]

x_train
#%% md
# ## Step2 基础准备
#%%
import numpy as np


# 加一个 bias 列（全1列）到特征中
def add_bias(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


X_train_b = add_bias(x_train)  # shape = (891, 6)
X_test_b = add_bias(x_test)  # shape = (418, 6)

#%% md
# ## Step3 实现 sigmoid 和 loss 函数
#%%
def sigmoid(z):
    z = np.clip(z, -500, 500)  # 限制输入范围，避免溢出
    return 1 / (1 + np.exp(-z))


def compute_loss(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)  # 避免log(0)
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
#%% md
# ## Step4 训练函数
#%%
def train_logistic_regression(x, y, lr=0.001, epochs=100):
    m, n = x.shape
    weights = np.zeros(n)

    for epoch in range(epochs):
        # 前向传播
        z = np.dot(x, weights)
        y_pred = sigmoid(z)

        # 损失
        loss = compute_loss(y, y_pred)

        # 梯度
        gradient = np.dot(x.T, (y_pred - y)) / m

        # 更新参数
        weights -= lr * gradient

        # 打印部分训练过程
        if epoch % 5000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights
#%% md
# ## Step5 训练模型
#%%
weights = train_logistic_regression(X_train_b, y_train, lr=0.005, epochs=100000)
#%% md
# ## Step6 模型推理
#%%
def predict(X, weights):
    probs = sigmoid(np.dot(X, weights))
    return (probs >= 0.5).astype(int)


y_train_pred = predict(X_train_b, weights)
accuracy = np.mean(y_train_pred == y_train)
print(f"Train Accuracy: {accuracy:.4f}")
#%%
test_preds = predict(X_test_b, weights)

# test_df 中的 PassengerId 是测试集乘客的 ID
submission = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": test_preds
})

submission.to_csv("submission.csv", index=False)
print("保存完成：submission.csv")