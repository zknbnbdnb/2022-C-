from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# 第3问确定类型

df = pd.read_excel('D:/pytorch/2022c建模/kmeans结果.xlsx')

X, y = df.iloc[:, 9:23].values, df.iloc[:, 4].values
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022)

c_list = [1e2, 1e1, 1, 1e-1, 1e-2, 1e-3]
method_list = ['rbf', 'linear', 'poly']
res = []
x_label = []

for i in range(len(c_list)):
    for j in range(len(method_list)):
        x_label.append(str(c_list[i]) + ',' + method_list[j])
        clf_tmp = svm.SVC(C=c_list[i], kernel=method_list[j])
        clf_tmp.fit(X_train, y_train)
        pred_tmp = clf_tmp.predict(X_test)
        scoure = 0
        for k in range(len(pred_tmp)):
            if pred_tmp[k] == y_test[k]:
                scoure += 1
        res.append(scoure / len(pred_tmp))

plt.figure(figsize=(14,10))
plt.plot(x_label, res, 'o-')
plt.axhline(1, color='r')
plt.xticks(rotation=90)
plt.savefig('svm.png')
plt.show()


clf = svm.SVC(kernel='linear').fit(X_train, y_train)
pred1 = clf.predict(X_test)
print(classification_report(pred1, y_test))

print(pred1)