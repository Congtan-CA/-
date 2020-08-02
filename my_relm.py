import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from numpy import*
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# pandas导入数据
df = pd.read_excel('Airpak_data.xlsx')
a = np.array(df)

# numpy.around(保留三位小数点)
X = np.around(a,decimals=3)
x = X[:,[0,2,4,5]]
y = X[:,[1]]

# 随机划分数据集
X_train,X_test,Y_train,Y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# 归一化
m1 = MinMaxScaler()
x_train = m1.fit_transform(X_train)
x_test = m1.fit_transform(X_test)
m2 = MinMaxScaler()
y_train = m2.fit_transform(Y_train)

# 创建elm类模型
class elm:
    def __init__(self,x,hidden_num,c):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState()
        # 随机权重
        self.w_1 = rnd.uniform(0,1,(hidden_num,columns))
        # 随机阈值
        self.b = np.zeros([hidden_num,row],dtype=float)
        for i in range(hidden_num):
            rand_b = rnd.uniform(0,1)
            for j in range(columns):
                self.b[i,j] = rand_b
        x_matrix = np.matrix(x).H
        self.h0 = np.dot(self.w_1,x_matrix)+self.b
        self.h = self.sigmoid(self.h0).H
        # L2正则化处理
        self.c = c
        self.mh = np.dot(self.h.H, self.h)
        self.p1 = np.linalg.pinv(self.mh+c)
        self.p2 = np.dot(self.p1,self.h.H)
    @staticmethod
    def sigmoid(x):
        return  1/(1 + np.exp(-x))

    # 回归问题 训练
    def regressor_train(self, y):
        self.beta =np.dot(self.p2,y)
        return self.beta

    def regressor_test(self,test_x):
        test_x_matrix = np.matrix(test_x)
        b_row = test_x_matrix.shape[0]
        self.test_b = self.b[:,:b_row]
        self.test_h = np.matrix(self.sigmoid(np.dot(self.w_1,test_x_matrix.H) + self.test_b)).H
        result = np.dot(self.test_h, self.beta)
        return result

# 网络训练与仿真
my_Elm = elm(x_train,500,1e8)
my_Elm.regressor_train(y_train)
elm_test = my_Elm.regressor_test(x_test)
# 数据还原
yy_test = m2.inverse_transform(elm_test)

# 重组结果矩阵
result_elm = np.hstack((X_test[:,[0]],Y_test[:,[0]]))
result_elm = np.hstack((result_elm,yy_test[:,[0]]))
a = result_elm[np.lexsort(result_elm[:,::-1].T)]

plt.plot(a[:,0], a[:,1], '.g')
plt.plot(a[:,0], a[:,2], '-r')
plt.title('elm_PMV')
plt.xlabel('temperature')
plt.ylabel('pmv')
plt.legend([['original'], ['prediction']], loc='upper left')
plt.show()
