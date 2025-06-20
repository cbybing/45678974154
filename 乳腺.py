# -*- coding: utf-8 -*-
"""
乳腺癌肿瘤预测项目
"""
# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # 添加PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 用户自己导入数据集
# 假设数据集为CSV格式，包含特征和目标标签，最后一列为目标标签
file_path = 'breast_cancer.csv'  # 设置数据文件路径
data = pd.read_csv(file_path)

# 假设数据集的最后一列为标签，其余列为特征
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 获取特征名称和目标名称（如果有列名）
feature_names = data.columns[:-1].tolist()
target_names = np.unique(y).tolist()

# 数据集基本信息
print(f"数据集大小：{X.shape}")
print(f"特征名称：{feature_names}")
print(f"目标名称：{target_names}")

# 绘制诊断结果分布图
plt.figure(figsize=(6, 4))
sns.countplot(y)
plt.title("诊断结果分布图")
plt.xlabel("诊断结果")
plt.ylabel("样本数量")
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 绘制特征相关性热力图
correlation_matrix = pd.DataFrame(X_train).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("特征相关性热力图")
plt.show()

# 使用PCA进行降维并绘制散点图
pca = PCA(n_components=2)  # 降维到2D
X_train_pca = pca.fit_transform(X_train)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', marker='o')
plt.title("PCA降维后的训练集数据分布")
plt.xlabel("主成分1")
plt.ylabel("主成分2")
plt.colorbar(scatter, ticks=np.unique(y_train), label='诊断结果')
plt.show()

# 绘制损失曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("训练样本数")
    plt.ylabel("损失")
    train_sizes, train_loss, test_loss = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_log_loss')
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    plt.plot(train_sizes, train_loss_mean, 'o-', color="r", label="训练损失")
    plt.plot(train_sizes, test_loss_mean, 'o-', color="g", label="验证损失")
    plt.legend(loc="best")
    plt.show()

# 决策树模型训练与优化
def train_decision_tree(X_train, X_test, y_train, y_test):
    # 初始化决策树分类器
    dtree = DecisionTreeClassifier(random_state=42)

    # 超参数网格
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3, 4]
    }

    # 网格搜索
    grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # 输出最优超参数
    print("决策树最佳参数:", grid_search.best_params_)

    # 使用最优参数训练模型
    best_dtree = grid_search.best_estimator_
    y_pred = best_dtree.predict(X_test)

    # 评估模型
    evaluate_model("决策树", y_test, y_pred)

    # 绘制损失曲线
    plot_learning_curve(best_dtree, "决策树损失曲线", X_train, y_train, cv=5, n_jobs=-1)

    return best_dtree

# 逻辑回归模型训练与优化
def train_logistic_regression(X_train, X_test, y_train, y_test):
    # 初始化逻辑回归分类器
    log_reg = LogisticRegression(random_state=42, max_iter=1000)

    # 超参数网格
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs', 'saga'],
        'penalty': ['l1', 'l2', 'elasticnet', 'none']
    }

    # 网格搜索
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # 输出最优超参数
    print("逻辑回归最佳参数:", grid_search.best_params_)

    # 使用最优参数训练模型
    best_log_reg = grid_search.best_estimator_
    y_pred = best_log_reg.predict(X_test)

    # 评估模型
    evaluate_model("逻辑回归", y_test, y_pred)

    # 绘制损失曲线
    plot_learning_curve(best_log_reg, "逻辑回归损失曲线", X_train, y_train, cv=5, n_jobs=-1)

    return best_log_reg

# 模型评估函数
def evaluate_model(model_name, y_true, y_pred):
    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    # 打印评估结果
    print(f"{model_name}模型评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # 打印分类报告
    print(f"{model_name}分类报告:")
    print(classification_report(y_true, y_pred))

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name}混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.show()

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f"{model_name} ROC曲线")
    plt.legend(loc="lower right")
    plt.show()

# 主函数
if __name__ == "__main__":
    # 训练决策树模型
    best_dtree = train_decision_tree(X_train, X_test, y_train, y_test)
    print("\n" + "=" * 50 + "\n")

    # 训练逻辑回归模型
    best_log_reg = train_logistic_regression(X_train, X_test, y_train, y_test)