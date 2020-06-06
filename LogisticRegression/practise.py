import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# 导入和拆分数据集
data = pd.read_csv('/home/jinghuan/Documents/PythonFiles/Practise/Practise13/iris.csv', dtype=str, delimiter=',')
x = data.iloc[:, 0:4]
y = data.iloc[:, 4:]
x_train, x_test, y_train, y_test = train_test_split(x, np.ravel(y), test_size=0.25, random_state=0, stratify=y)

# 构建Logistics回归模型
iris_logistic = LogisticRegression()
iris_logistic.fit(x_train, y_train)
a = iris_logistic.intercept_  # 截距
b = iris_logistic.coef_  # 斜率
print('模型截距项为：\n', a)
print('各自变量的系数为：\n', b)

# 模型预测
iris_predict = iris_logistic.predict(x_test)
cm = metrics.confusion_matrix(y_test, iris_predict, labels=['setosa', 'versicolor', 'virginica'])
print('混淆矩阵为：\n', cm)
# 绘制热力图
sns.heatmap(cm, annot=True, fmt='.2e',cmap='GnBu')
plt.show()
