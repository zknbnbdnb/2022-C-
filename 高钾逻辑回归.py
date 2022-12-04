from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------第一题第一小问--------

df = pd.read_excel('D:/pytorch/2022c建模/汇总信息表.xlsx', sheet_name='高钾')
X, y = df.iloc[:, 7:21].to_numpy(), df.iloc[:, 2].to_numpy()
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

def gaoK(x):
    return np.array([
            0.723*x[0] + 0.008,  0*x[1] + 0.695,
            3.914*x[2] + 7.204, 2.641*x[3] + 3.035,
            0.204*x[4] + 1.039, 2.704*x[5] + 1.401,
            0.478*x[6] + 1.805, 1.114*x[7] + 0.713,
            0*x[8] + 0.412, 0*x[9] + 0.598,
            0.364*x[10] + 1.301, 0*x[11] + 0.042,
            0*x[12] + 0.197, 0*x[13] + 0.102
    ])

data = pd.read_excel('D:/pytorch/2022c建模/未风化-风化文物信息表.xlsx', sheet_name='铅钡风化')
data1 = data.iloc[:, 7:21].to_numpy()
new_data = []
for i in range(len(data1)):
    new_data.append(gaoK(data1[i]))
new_data = np.asarray(new_data)

pred1 = clf.predict(new_data)
print(pred1)

# ----------第三题--------

df_new = pd.read_excel('D:/pytorch/2022c建模/第三题结果.xlsx', sheet_name='高钾')
x_new = df_new.iloc[:, 4:].values

print(clf.predict(x_new))