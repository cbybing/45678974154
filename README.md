# 45678974154
本项目旨在通过对乳腺癌数据集的分析与建模，实现对乳腺癌肿瘤性质的预测（良性或恶性），为医学诊断提供辅助参考。项目采用 Python 语言及相关机器学习库，构建并优化了决策树和逻辑回归两种模型，以实现较高的预测准确性。
安装指南
本项目基于 Python 3.x 开发，需要以下依赖库：
numpy
pandas
matplotlib
seaborn
scikit-learn
可使用以下命令安装所需依赖：
python breast_cancer_prediction.py
代码说明
主要功能与步骤
数据加载与预处理 ：加载乳腺癌数据集，分离特征和目标标签，并对数据进行标准化处理，以消除特征之间量纲的影响，提升模型训练效果。
特征分析与可视化 ：绘制诊断结果分布图、特征相关性热力图以及 PCA 降维后的数据分布散点图，帮助理解数据的基本情况和特征之间的关系。
模型训练与优化 ：分别构建决策树和逻辑回归模型，为每个模型设置超参数网格，利用网格搜索结合交叉验证的方式寻找最优超参数组合，以提升模型的预测性能。
模型评估 ：使用测试集对训练好的模型进行评估，输出包括准确率、精确率、召回率、F1 - score、ROC - AUC 等在内的多项评估指标，并绘制混淆矩阵和 ROC 曲线，直观展示模型的分类效果。
关键代码片段
数据加载与预处理：
  # 用户自己导入数据集
  # 假设数据集为CSV格式，包含特征和目标标签，最后一列为目
  file_path = 'breast_cancer.csv'  # 设置数据文件路径
  data = pd.read_csv(file_path)

  # 假设数据集的最后一列为标签，其余列为特征
  X = data.iloc[:, :-1].values
  y = data.iloc[:, -1].values

  # 特征标准化
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
模型评估：
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
项目结果
项目运行后，将输出决策树和逻辑回归模型的各项评估指标以及对应的可视化图表。通过比较两个模型的评估结果，可以确定哪个模型在当前乳腺癌数据集上的预测效果更好，从而为实际的乳腺癌肿瘤预测任务提供更可靠的模型选择依据。例如，模型的准确率、ROC - AUC 等指标值越高，说明模型的预测性能越佳；混淆矩阵可以直观地展示模型对不同类别（良性、恶性）样本的正确预测和错误预测情况。
