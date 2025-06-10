import re
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials
from xgboost import XGBClassifier
import xgboost
import matplotlib.pyplot as plt
from model.xgb import XGBoostOptimizer
from model.svm import PCASVMClassifier
from model.knn import StandardizedKNNClassifier
import seaborn as sns

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 加载数据并对数据进行初步分析
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
# 合并训练、测试集数据
alldata = pd.concat([train_data, test_data], axis = 0, ignore_index = True)

# print(train_data.head())

print(f"训练集有{len(train_data)}个样本数据，测试集有{len(test_data)}样本数据")
print(f"总共有{len(alldata)}个样本数据")

# 查看每列缺失值数量
print(alldata.isnull().sum())

# 可视化缺失值分布
plt.figure(figsize=(10, 6))
sns.heatmap(alldata.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# 对样本数值数据进行更详细观察，得出其分布特征
# round(train_data.describe(percentiles=[.5, .6, .7, .8, .9,]), 2)

# 对train_data中所有对象类型的列进行描述性统计分析
train_data.describe(include=['O'])

# 对一些类别特征进行进一步分析，观察其与是否生存之间的相关程度
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)

sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Survival Rate by Pclass')
plt.show()

train_data[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)

sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Survival Rate by Sex')
plt.show()

train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)

sns.barplot(x='SibSp', y='Survived', data=train_data)
plt.title('Survival Rate by SibSp')
plt.show()

train_data[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)

sns.barplot(x='Parch', y='Survived', data=train_data)
plt.title('Survival Rate by Parch')
plt.show()

train_data[['Embarked', 'Survived']].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False)

sns.barplot(x='Embarked', y='Survived', data=train_data)
plt.title('Survival Rate by Embarked')
plt.show()

# 对一些类别数据进行简单的可视化
plt.scatter(x=train_data["Age"], y=train_data["Fare"], c=train_data["Survived"])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.colorbar(label='Survived')
plt.show()

# 数据预处理
# 提取乘客称谓到title中
alldata['title'] = alldata.Name.apply(lambda x: re.search(r',\s(.+?)\.', x).group(1))

# 统计title称谓计数
alldata.title.value_counts()

fig = px.scatter(alldata, x="title", y="Age", color="Sex",
                 title="年龄、姓氏、性别分布图")

# 按照不同性别分组绘制散点图
for sex, group in alldata.groupby('Sex'):
    plt.scatter(group['title'], group['Age'], label=sex)
plt.xlabel("Title")
plt.ylabel("Age")
plt.title("年龄、姓氏、性别分布图")
plt.legend(title='Sex')
plt.show()

# 整合称谓信息
alldata.loc[alldata.title.isin(['Ms', 'Mlle']), 'title'] = 'Miss'
alldata.loc[alldata.title.isin(['Mme']), 'title'] = 'Mrs'
rare = ['Major', 'Lady', 'Sir', 'Don', 'Capt', 'the Countess', 'Jonkheer', 'Dona', 'Dr', 'Rev', 'Col']
alldata.loc[alldata.title.isin(rare), 'title'] = 'rare'
alldata.title.value_counts()

# 删除Name数据
alldata.drop(['Name'], axis=1, inplace=True)

# 提取新特征
alldata['family_size'] = alldata.SibSp + alldata.Parch + 1
alldata['ticket_group_count'] = alldata.groupby('Ticket')['Ticket'].transform('count')
alldata.drop('Ticket', axis=1, inplace=True)

alldata['group_size'] = alldata[['family_size', 'ticket_group_count']].max(axis=1)
alldata['is_alone'] = alldata.group_size.apply(lambda x: 1 if x == 1 else 0)

# 按照 Pclass 分组绘制散点图
for pclass, group in alldata.groupby('Pclass'):
    plt.scatter(group['Pclass'], group['Fare'], label=f'Pclass {pclass}')
# 添加标签和标题
plt.xlabel("Pclass")
plt.ylabel("Fare")
plt.title("票价，等级分布图")
plt.legend(title='Pclass')
plt.show()

# 每位乘客实际支付的票价（总票价除以同票人数）
alldata['fare_p'] = alldata.Fare / alldata.ticket_group_count
# 使用对应舱位等级的中位数票价进行填充
alldata.loc[alldata[alldata.fare_p.isna()].index, 'fare_p'] = alldata.groupby('Pclass')['fare_p'].median()[3]

alldata.drop('Fare', axis=1, inplace=True)

# 处理港口缺失数据 找出 Embarked 列中缺失值的位置，定位到对应的行，将这些缺失值填充为 'S'
alldata.loc[alldata.Embarked.isnull(), 'Embarked'] = 'S'

# 独热编码
preprocessing_dummies = pd.get_dummies(alldata[['Pclass', 'Sex', 'Embarked', 'title']],
               columns = ['Pclass', 'Sex', 'Embarked', 'title'],
               prefix = ['pclass', 'sex', 'embarked', 'title'],
               drop_first= False
              )
alldata = pd.concat([alldata, preprocessing_dummies], axis=1)
alldata.drop(['Pclass', 'Sex', 'Embarked', 'title'], axis=1, inplace=True)


# 处理Age数据 K近邻算法的缺失值插补器填充数据集中Age字段的缺失值
imputer = KNNImputer(n_neighbors=4)
# 选择用于插补的特征
features = ['SibSp', 'Parch', 'Age',
       'family_size', 'ticket_group_count', 'group_size', 'is_alone',
       'fare_p', 'pclass_1', 'pclass_2', 'pclass_3',
       'sex_female', 'sex_male', 'embarked_C', 'embarked_Q', 'embarked_S',
       'title_Master', 'title_Miss', 'title_Mr', 'title_Mrs', 'title_rare']
all_data_filled = pd.DataFrame(imputer.fit_transform(alldata[features]), columns=features)
alldata['Age'] = all_data_filled['Age']


# k均值聚类 把年龄划分成多个年龄段
kmeans = KMeans(n_clusters=4, random_state=41)
labels_pred = kmeans.fit_predict(alldata[['Age']])

# 获取聚类中心，返回每段年龄平均值。
kmeans.cluster_centers_.flatten()
# 返回聚类中心从小到大的索引顺序
np.argsort(kmeans.cluster_centers_.flatten())
# 将原始聚类标签映射为按年龄升序排列的新标签
label_dict = {label: v for v, label in enumerate(np.argsort(kmeans.cluster_centers_.flatten()))}
# 重映射聚类标签
labels = [label_dict[label] for label in labels_pred]
alldata['Age_category'] = labels


# 按照 Age_category 分组绘制散点图
for category, group in alldata.groupby('Age_category'):
    plt.scatter(group['Age_category'], group['Age'], label=category)

# 添加标签和标题
plt.xlabel("Age Category")
plt.ylabel("Age")
plt.title("年龄分布图")
plt.legend(title='Age Category')
plt.show()

alldata.drop(['PassengerId', 'Cabin', 'Age'], axis = 1, inplace = True)



# 查看处理后数据信息
alldata.info()

# 对整体数据再处理，划分数据集
train_clean = alldata.loc[alldata.Survived.notnull()].copy()
test_clean = alldata.loc[alldata.Survived.isnull()].drop('Survived', axis = 1).copy()

X = train_clean.drop('Survived', axis = 1)
y = train_clean.Survived

features = ['family_size', 'group_size', 'is_alone', 'Age_category','ticket_group_count',
       'pclass_1', 'pclass_2', 'pclass_3', 'sex_female', 'sex_male',
       'embarked_C', 'embarked_Q', 'embarked_S', 'title_Master', 'title_Miss',
       'title_Mr', 'title_Mrs', 'title_rare', 'fare_p']
X = X[features].copy()

# 数据准备
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)

# # xgboost
# # 创建DMatrix
# dtrain = xgboost.DMatrix(X_train, label=y_train)
# dval = xgboost.DMatrix(X_val, label=y_val)
#
# optimizer = XGBoostOptimizer(X_train, y_train, dtrain, dval, y_val)
#
# # 搜索最佳参数
# best_params = optimizer.search_best_params()
# # 训练模型
# bst = optimizer.train_best_model()
# # 评估模型性能
# metrics = optimizer.evaluate_model()
# # 获取模型或参数
# model = optimizer.get_best_model()
# print("Best Parameters:", optimizer.get_best_params())


# svm
# 创建模型对象
pca_svm = PCASVMClassifier(n_components=0.95)
# 训练模型
pca_svm.fit(X_train, y_train)
# 评估模型
pca_svm.evaluate(X_val, y_val)
# 获取模型或 PCA
model = pca_svm.get_model()
pca = pca_svm.get_pca()

# # knn
# # 创建并训练模型
# knn_clf = StandardizedKNNClassifier(n_neighbors=5)
# knn_clf.fit(X_train, y_train)
# # 评估模型
# knn_clf.evaluate(X_val, y_val)
# # 获取模型
# model = knn_clf.get_model()