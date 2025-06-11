import re
import numpy as np
import pandas as pd
import seaborn as sns
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
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据并对数据进行初步分析
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
# 合并训练、测试集数据
alldata = pd.concat([train_data, test_data], axis = 0, ignore_index = True)

# print(f"训练集有{len(train_data)}个样本数据，测试集有{len(test_data)}样本数据")
# print(f"总共有{len(alldata)}个样本数据")

# 对train_data中所有对象类型的列进行描述性统计分析
train_describe = train_data.describe(include=['O'])
print(train_describe)

# 对一些类别数据进行简单的可视化
plt.scatter(x=train_data["Age"], y=train_data["Fare"], c=train_data["Survived"])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.colorbar(label='Survived')
plt.show()

# 提取乘客称谓到title中
alldata['title'] = alldata.Name.apply(lambda x: re.search(r',\s(.+?)\.', x).group(1))
# 统计title称谓计数
print(alldata.title.value_counts())

# 整合称谓信息
alldata.loc[alldata.title.isin(['Ms', 'Mlle']), 'title'] = 'Miss'
alldata.loc[alldata.title.isin(['Mme']), 'title'] = 'Mrs'
rare = ['Major', 'Lady', 'Sir', 'Don', 'Capt', 'the Countess', 'Jonkheer', 'Dona', 'Dr', 'Rev', 'Col']
alldata.loc[alldata.title.isin(rare), 'title'] = 'rare'
alldata.title.value_counts()

# 按照不同性别分组绘制散点图
for sex, group in alldata.groupby('Sex'):
    plt.scatter(group['title'], group['Age'], label=sex)
plt.xlabel("Title")
plt.ylabel("Age")
plt.title("年龄、姓氏、性别分布图")
plt.legend(title='Sex')
plt.show()

# 删除Name数据
alldata.drop(['Name'], axis=1, inplace=True)

# 提取新特征
alldata['family_size'] = alldata.SibSp + alldata.Parch + 1
alldata['ticket_group_count'] = alldata.groupby('Ticket')['Ticket'].transform('count')
alldata.drop('Ticket', axis=1, inplace=True)
alldata['group_size'] = alldata[['family_size', 'ticket_group_count']].max(axis=1)
alldata['is_alone'] = alldata.group_size.apply(lambda x: 1 if x == 1 else 0)

# 按照Pclass分组绘制散点图
for pclass, group in alldata.groupby('Pclass'):
    plt.scatter(group['Pclass'], group['Fare'], label=f'Pclass {pclass}')
# 添加标签和标题
plt.xlabel("Pclass")
plt.ylabel("Fare")
plt.title("票价，等级分布图")
plt.legend(title='Pclass')
plt.show()


# 每位乘客实际支付的票价 总票价除以同票人数
alldata['fare_p'] = alldata.Fare / alldata.ticket_group_count
# 使用对应舱位等级的中位数票价进行填充
alldata.loc[alldata[alldata.fare_p.isna()].index, 'fare_p'] = alldata.groupby('Pclass')['fare_p'].median()[3]
alldata.drop('Fare', axis=1, inplace=True)

# 处理港口缺失数据 找出Embarked列中缺失值的位置，定位到对应的行，将这些缺失值填充为S
alldata.loc[alldata.Embarked.isnull(), 'Embarked'] = 'S'

# 独热编码
preprocessing_dummies = pd.get_dummies(alldata[['Pclass', 'Sex', 'Embarked', 'title']],
               columns = ['Pclass', 'Sex', 'Embarked', 'title'],
               prefix = ['pclass', 'sex', 'embarked', 'title'],
               drop_first= False
              )
alldata = pd.concat([alldata, preprocessing_dummies], axis=1)
alldata.drop(['Pclass', 'Sex', 'Embarked', 'title'], axis=1, inplace=True)


# 处理Age数据 KNN算法填充数据集中Age字段的缺失值
imputer = KNNImputer(n_neighbors=4)
# 用于插补的特征
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
# 获取聚类中心，返回每段年龄平均值
kmeans.cluster_centers_.flatten()
# 返回聚类中心从小到大的索引顺序
np.argsort(kmeans.cluster_centers_.flatten())
# 将无序的簇标签映射为按年龄升序的从0开始的新标签
label_dict = {label: v for v, label in enumerate(np.argsort(kmeans.cluster_centers_.flatten()))}
labels = [label_dict[label] for label in labels_pred]
alldata['Age_category'] = labels

# 按照 Age_category 分组绘制散点图
for category, group in alldata.groupby('Age_category'):
    plt.scatter(group['Age_category'], group['Age'], label=category, s=10)
plt.xlabel("Age Category")
plt.ylabel("Age")
plt.title("年龄分布图")
plt.legend(title='Age Category')
plt.show()

alldata.drop(['PassengerId', 'Cabin', 'Age'], axis=1, inplace=True)

# 查看处理后数据信息
print(alldata.info())

# 对整体数据再处理，划分数据集
train_clean = alldata.loc[alldata.Survived.notnull()].copy()
test_clean = alldata.loc[alldata.Survived.isnull()].drop('Survived', axis=1).copy()

X = train_clean.drop('Survived', axis=1)
y = train_clean.Survived

features = ['family_size', 'group_size', 'is_alone', 'Age_category','ticket_group_count',
       'pclass_1', 'pclass_2', 'pclass_3', 'sex_female', 'sex_male',
       'embarked_C', 'embarked_Q', 'embarked_S', 'title_Master', 'title_Miss',
       'title_Mr', 'title_Mrs', 'title_rare', 'fare_p']
X = X[features].copy()

# 数据准备
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)

# 存储loss
com_results = {}
com_time = {}
'''
xgboost
'''
# 创建DMatrix
dtrain = xgboost.DMatrix(X_train, label=y_train)
dval = xgboost.DMatrix(X_val, label=y_val)

xgb_model = XGBoostOptimizer(X_train, y_train, dtrain, dval, y_val)
# 搜索最佳超参数
# best_params = xgb_model.search_best_params()
best_params = {'colsample_bytree': 0.8776679846181519, 'gamma': 2.4503929556523807, 'learning_rate': 0.5721294623723316, 'max_depth': 5, 'min_child_weight': 1.0, 'subsample': 0.90421468196085}
# 存储loss
evals_result = {}
# 训练模型
bst, evals_result = xgb_model.train_best_model(evals_result, best_params)
# 最佳迭代次数
best_iteration = bst.best_iteration
train_loss = evals_result['train']['logloss']
eval_loss = evals_result['eval']['logloss']
# 绘制训练 loss 和验证 loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss', color='blue')
plt.plot(eval_loss, label='Validation Loss', color='orange')
plt.axvline(x=best_iteration, color='red', linestyle='--', label='Best Iteration')
plt.xlabel('Boosting Rounds')
plt.ylabel('Log Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 评估模型性能
start_time = time.time()
metrics = xgb_model.evaluate_model()
inference_time_xbg = time.time() - start_time
com_results['XGBoost'] = {'accuracy': metrics['accuracy'], 'inference_time': inference_time_xbg}
print("Best Parameters:", xgb_model.get_best_params())


'''
SVM
'''
# 创建模型
pca_svm = PCASVMClassifier(n_components=0.95)
# 训练模型
pca_svm.fit(X_train, y_train)
# 评估模型
start_time = time.time()
y_pred_svm = pca_svm.evaluate(X_val, y_val)
inference_time_svm = time.time() - start_time
pca = pca_svm.get_pca()
acc_svm = accuracy_score(y_val, y_pred_svm)

com_results['SVM'] = {'accuracy': acc_svm, 'inference_time': inference_time_svm}


'''
KNN
'''
# 创建并训练模型
knn_clf = StandardizedKNNClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)
# 评估模型
start_time = time.time()
y_pred_knn = knn_clf.evaluate(X_val, y_val)
inference_time_knn = time.time() - start_time
# 获取模型
knn_model = knn_clf.get_model()
acc_knn = accuracy_score(y_val, y_pred_knn)

com_results['KNN'] = {'accuracy': acc_knn, 'inference_time': inference_time_knn}






# 绘制acc和time的对比图
results_df = pd.DataFrame.from_dict(com_results, orient='index').reset_index()
results_df.columns = ['Model', 'accuracy', 'inference_time']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('模型准确率与推理时间对比', fontsize=16)

# 左图：准确率
sns.barplot(x='Model', y='accuracy', data=results_df, palette="Blues", ax=ax1)
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.5, 1.0)
ax1.set_title('模型准确率')
for i, v in enumerate(results_df['accuracy']):
    ax1.text(i, v + 0.01, f"{v:.2%}", ha='center', fontsize=10)

# 右图：推理时间
sns.barplot(x='Model', y='inference_time', data=results_df, palette="Greens", ax=ax2)
ax2.set_ylabel('Inference Time (seconds)')
ax2.set_ylim(0, 0.08)
ax2.set_title('模型推理时间')
for i, v in enumerate(results_df['inference_time']):
    ax2.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)

plt.show()
