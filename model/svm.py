from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


class PCASVMClassifier:
    def __init__(self, n_components=0.95, C=1.0, kernel='rbf', gamma='scale'):
        """
        初始化 PCA + SVM 分类器
        :param n_components: PCA 保留的主成分数量（可以是比例或具体数值）
        :param C: SVM 正则化参数
        :param kernel: 核函数类型
        :param gamma: 核函数系数
        """
        self.n_components = n_components
        self.pca = None
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=C, kernel=kernel, gamma=gamma))
        ])

    def apply_pca(self, X_train, X_val):
        """
        应用 PCA 到训练集和验证集
        :param X_train: 训练集特征
        :param X_val: 验证集特征
        :return: 降维后的训练集、验证集、PCA 模型
        """
        pca = PCA(n_components=self.n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        self.pca = pca
        return X_train_pca, X_val_pca

    def build_model(self, C=1.0, kernel='rbf', gamma='scale'):
        """
        构建包含标准化和 SVM 的流水线模型
        :param C: SVM 正则化参数
        :param kernel: 核函数类型
        :param gamma: 核函数系数
        """
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=C, kernel=kernel, gamma=gamma))
        ])

    def fit(self, X_train, y_train):
        """
        使用 PCA 处理数据并训练模型
        :param X_train: 原始训练集特征
        :param y_train: 训练集标签
        """
        X_train_pca, _ = self.apply_pca(X_train, X_train)  # 只在训练集上 fit PCA
        self.model.fit(X_train_pca, y_train)

    def predict(self, X_val):
        """
        使用训练好的模型对验证集进行预测
        :param X_val: 原始验证集特征
        :return: 预测结果
        """
        if self.pca is None:
            raise ValueError("请先调用 fit 方法训练模型")
        X_val_pca = self.pca.transform(X_val)
        return self.model.predict(X_val_pca)

    def evaluate(self, X_val, y_val):
        """
        评估模型性能
        :param X_val: 原始验证集特征
        :param y_val: 验证集标签
        """
        y_pred = self.predict(X_val)
        print("Accuracy:", accuracy_score(y_val, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        return y_pred

    def get_pca(self):
        """获取训练好的 PCA 模型"""
        return self.pca

    def get_model(self):
        """获取训练好的模型"""
        return self.model