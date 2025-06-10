from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


class StandardizedKNNClassifier:
    def __init__(self, n_neighbors=5):
        """
        初始化标准化 + KNN 分类器
        :param n_neighbors: KNN 中的邻居数
        """
        self.n_neighbors = n_neighbors
        self.model = None

    def build_model(self):
        """
        构建包含标准化和 KNN 的流水线模型
        """
        self.model = Pipeline([
            ('scaler', StandardScaler()),  # 标准化
            ('knn', KNeighborsClassifier(n_neighbors=self.n_neighbors))
        ])

    def fit(self, X_train, y_train):
        """
        训练模型
        :param X_train: 原始训练集特征
        :param y_train: 训练集标签
        """
        self.build_model()
        self.model.fit(X_train, y_train)

    def predict(self, X_val):
        """
        对验证集进行预测
        :param X_val: 验证集特征
        :return: 预测结果
        """
        if self.model is None:
            raise ValueError("请先调用 fit 方法训练模型")
        return self.model.predict(X_val)

    def evaluate(self, X_val, y_val):
        """
        评估模型性能
        :param X_val: 验证集特征
        :param y_val: 验证集标签
        """
        y_pred = self.predict(X_val)
        print("Accuracy:", accuracy_score(y_val, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        return y_pred

    def get_model(self):
        """获取训练好的模型"""
        return self.model