import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# 读取文件并解析数据
def parse_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 去掉换行符并将字符串转换为浮点数
    return np.array([float(line.strip()) for line in lines])

# 解析真实标签
y_true_class0 = parse_data('../ROC/test1_roc_class0.txt')
y_true_class1 = parse_data('../ROC/test1_roc_class1.txt')
y_true_class2 = parse_data('../ROC/test1_roc_class2.txt')
y_true2_class0 = parse_data('../ROC/test2_roc_class0.txt')
y_true2_class1 = parse_data('../ROC/test2_roc_class1.txt')
y_true2_class2 = parse_data('../ROC/test2_roc_class2.txt')

# 合并所有类别的真实标签
y_true_all = np.hstack([y_true_class0, y_true_class1, y_true_class2])
y_true_all2 = np.hstack([y_true2_class0, y_true2_class1, y_true2_class2])

# 定义不同方法的路径
methods = {
    "Prompt-learning": {
        "class0": "result/Prompt_class0.txt",
        "class1": "result/Prompt_class1.txt",
        "class2": "result/Prompt_class2.txt"
    },
    "BERT": {
        "class0": "result/BertDA_class0.txt",
        "class1": "result/BertDA_class1.txt",
        "class2": "result/BertDA_class2.txt"
    },
    # "Roberta":{
    #     "class0": "result/Roberta_DA_class0.txt",
    #     "class1": "result/Roberta_DA_class1.txt",
    #     "class2": "result/Roberta_DA_class2.txt"
    # },
    "Two-stage": {
        "class0": "result/2Stage_class0.txt",
        "class1": "result/2Stage_class1.txt",
        "class2": "result/2Stage_class2.txt"
    }
}
methods2 = {
    "Prompt-learning": {
        "class0": "result2/Prompt_class0.txt",
        "class1": "result2/Prompt_class1.txt",
        "class2": "result2/Prompt_class2.txt"
    },
    "BERT": {
        "class0": "result2/BertDA_class0.txt",
        "class1": "result2/BertDA_class1.txt",
        "class2": "result2/BertDA_class2.txt"
    },
    # "Roberta":{
    #     "class0": "../ROC/result2/Roberta2_DA_class0.txt",
    #     "class1": "../ROC/result2/Roberta2_DA_class1.txt",
    #     "class2": "../ROC/result2/Roberta2_DA_class2.txt"
    # }
    "Two-stage": {
        "class0": "result2/2Stage_class0.txt",
        "class1": "result2/2Stage_class1.txt",
        "class2": "result2/2Stage_class2.txt"
    }
}

# 颜色和线型设置
# colors = ['blue', 'green', 'red','gold']
colors = ['blue', 'green','gold']
line_styles = ['-', '-','-']

# 绘制 PR 曲线
plt.figure(figsize=(8, 4),dpi=500)
ax3=plt.subplot(121)
for i, (method_name, method_paths) in enumerate(methods.items()):
    # 加载该方法的预测概率
    y_score_class0 = parse_data(method_paths["class0"])
    y_score_class1 = parse_data(method_paths["class1"])
    y_score_class2 = parse_data(method_paths["class2"])

    # 合并所有类别的预测概率
    y_score_all = np.hstack([y_score_class0, y_score_class1, y_score_class2])

    # 计算微平均的 Precision-Recall 曲线
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_all, y_score_all)
    average_precision_micro = average_precision_score(y_true_all, y_score_all)

    # 绘制该方法的 PR 曲线
    plt.plot(
        recall_micro,
        precision_micro,
        color=colors[i],
        linestyle=line_styles[i],
        lw=2,
        label=f'{method_name} (AP = {average_precision_micro:.2f})'
    )

# 设置图形属性
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curves for test-1')
plt.legend(loc='lower left')

ax3=plt.subplot(122)
for i, (method_name, method_paths) in enumerate(methods2.items()):
    # 加载该方法的预测概率
    y_score_class0 = parse_data(method_paths["class0"])
    y_score_class1 = parse_data(method_paths["class1"])
    y_score_class2 = parse_data(method_paths["class2"])

    # 合并所有类别的预测概率
    y_score_all = np.hstack([y_score_class0, y_score_class1, y_score_class2])

    # 计算微平均的 Precision-Recall 曲线
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_all2, y_score_all)
    average_precision_micro = average_precision_score(y_true_all2, y_score_all)

    # 绘制该方法的 PR 曲线
    plt.plot(
        recall_micro,
        precision_micro,
        color=colors[i],
        linestyle=line_styles[i],
        lw=2,
        label=f'{method_name} (AP = {average_precision_micro:.2f})'
    )

# 设置图形属性
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curves for test-2')
plt.legend(loc='lower left')
plt.savefig('PR2.png',dpi=500,bbox_inches='tight', pad_inches=0)
plt.show()