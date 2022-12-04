from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------第一题第一小问--------

df = pd.read_excel('D:/pytorch/2022c建模/汇总信息表.xlsx', sheet_name='铅钡')
X, y = df.iloc[:, 9:23].to_numpy(), df.iloc[:, 2].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
pred = clf.predict(X_test)
res = 0
for i in range(len(pred)):
    if pred[i] == y_test[i]:
        res += 1

print(res / len(pred))

print((clf.coef_, clf.intercept_))

# ----------第一题第三小问--------

def qianBa(x):
    return np.array([
            2.017*x[0] + 0.075, 0.388*x[1] + 1.588,
            0.029*x[2] + 0.215, 0.421*x[3] + 0.152,
            0.299*x[4] + 0.435, 1.302*x[5] + 0.42,
            0.339*x[6] + 0.513, 0.536*x[7] + 0.241,
            0.505*x[8] + 0.012, 0.852*x[9] + 0.081,
            0.211*x[10] + 0.044, 0.088*x[11] + 0.235,
            0.004*x[12] + 0.046, 0.03*x[13] + 0.153
    ])

data = pd.read_excel('D:/pytorch/2022c建模/未风化-风化文物信息表.xlsx', sheet_name='高钾风化')
data1 = data.iloc[:, 7:21].to_numpy()
new_data = []
for i in range(len(data1)):
    new_data.append(qianBa(data1[i]))
new_data = np.asarray(new_data)

pred1 = clf.predict(new_data)
print(pred1)

# ----------第三题--------

df_new = pd.read_excel('D:/pytorch/2022c建模/第三题结果.xlsx', sheet_name='铅钡')
x_new = df_new.iloc[:, 4:].values
x_new = scaler.fit_transform(x_new)

print(clf.predict(x_new))


