import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
# 绘制每个类别的ROC曲线


with open('test1_roc_class0.txt', 'r') as f:
    lines = f.readlines()
with open('test1_roc_class1.txt', 'r') as f2:
    lines2 = f2.readlines()
with open('test1_roc_class2.txt', 'r') as f3:
    lines3 = f3.readlines()

with open('test2_roc_class0.txt', 'r') as f:
    lines_t2 = f.readlines()
with open('test2_roc_class1.txt', 'r') as f2:
    lines2_t2 = f2.readlines()
with open('test2_roc_class2.txt', 'r') as f3:
    lines3_t2 = f3.readlines()

with open('result/2Stage_class0.txt', 'r') as f:
    l1_1= f.readlines()
with open('result/2Stage_class1.txt', 'r') as f2:
    l1_2 = f2.readlines()
with open('result/2Stage_class2.txt', 'r') as f3:
    l1_3 = f3.readlines()

with open('result/BertDA_class0.txt', 'r') as f:
    l2_1= f.readlines()
with open('result/BertDA_class1.txt', 'r') as f2:
    l2_2 = f2.readlines()
with open('result/BertDA_class2.txt', 'r') as f3:
    l2_3 = f3.readlines()

with open('result/Prompt_class0.txt', 'r') as f:
    l3_1= f.readlines()
with open('result/Prompt_class1.txt', 'r') as f2:
    l3_2 = f2.readlines()
with open('result/Prompt_class2.txt', 'r') as f3:
    l3_3 = f3.readlines()


with open('result2/2Stage_class0.txt', 'r') as f:
    l4_1= f.readlines()
with open('result2/2Stage_class1.txt', 'r') as f2:
    l4_2 = f2.readlines()
with open('result2/2Stage_class2.txt', 'r') as f3:
    l4_3 = f3.readlines()

with open('result2/BertDA_class0.txt', 'r') as f:
    l5_1= f.readlines()
with open('result2/BertDA_class1.txt', 'r') as f2:
    l5_2 = f2.readlines()
with open('result2/BertDA_class2.txt', 'r') as f3:
    l5_3 = f3.readlines()

with open('result2/Prompt_class0.txt', 'r') as f:
    l6_1= f.readlines()
with open('result2/Prompt_class1.txt', 'r') as f2:
    l6_2 = f2.readlines()
with open('result2/Prompt_class2.txt', 'r') as f3:
    l6_3 = f3.readlines()

y_test1 = []
y_test2 = []

y_score_2stage = []
y_score_BertDA = []
y_score_Prompt = []

y_score_2staget2 = []
y_score_BertDAt2 = []
y_score_Promptt2 = []
for i in range(len(lines)):
    y_test1.append([int(lines[i]),int(lines2[i]),int(lines3[i])])
    y_score_2stage.append([float(l1_1[i]),float(l1_2[i]),float(l1_3[i])])
    y_score_BertDA.append([float(l2_1[i]),float(l2_2[i]),float(l2_3[i])])
    y_score_Prompt.append([float(l3_1[i]),float(l3_2[i]),float(l3_3[i])])


for i in range(len(lines_t2)):
    y_test2.append([int(lines_t2[i]), int(lines2_t2[i]), int(lines3_t2[i])])
    y_score_2staget2.append([float(l4_1[i]),float(l4_2[i]),float(l4_3[i])])
    y_score_BertDAt2.append([float(l5_1[i]),float(l5_2[i]),float(l5_3[i])])
    y_score_Promptt2.append([float(l6_1[i]),float(l6_2[i]),float(l6_3[i])])
# 计算每个类别的FPR、TPR和AUC值
# 将列表转换为numpy数组以便后续处理
y_test1 = np.array(y_test1)
y_test2 = np.array(y_test2)

y_score_2stage = np.array(y_score_2stage)
y_score_BertDA = np.array(y_score_BertDA)
y_score_Prompt = np.array(y_score_Prompt)

y_score_2staget2 = np.array(y_score_2staget2)
y_score_BertDAt2 = np.array(y_score_BertDAt2)
y_score_Promptt2 = np.array(y_score_Promptt2)

fpr_2stage = dict()
tpr_2stage = dict()
roc_auc_2stage = dict()

fpr_BertDA = dict()
tpr_BertDA = dict()
roc_auc_BertDA = dict()

fpr_Prompt = dict()
tpr_Prompt = dict()
roc_auc_Prompt = dict()

fpr_2staget2 = dict()
tpr_2staget2 = dict()
roc_auc_2staget2 = dict()

fpr_BertDAt2 = dict()
tpr_BertDAt2 = dict()
roc_auc_BertDAt2 = dict()

fpr_Promptt2 = dict()
tpr_Promptt2 = dict()
roc_auc_Promptt2 = dict()



for i in range(3):
    # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])  # 获取每个类别的FPR和TPR
    fpr_2stage[i], tpr_2stage[i], _ = roc_curve([row[i] for row in y_test1], [row[i] for row in y_score_2stage])  # 获取每个类别的FPR和TPR
    fpr_BertDA[i], tpr_BertDA[i], _ = roc_curve([row[i] for row in y_test1], [row[i] for row in y_score_BertDA])  # 获取每个类别的FPR和TPR
    fpr_Prompt[i], tpr_Prompt[i], _ = roc_curve([row[i] for row in y_test1], [row[i] for row in y_score_Prompt])  # 获取每个类别的FPR和TPR

    fpr_2staget2[i], tpr_2staget2[i], _ = roc_curve([row[i] for row in y_test2],
                                                [row[i] for row in y_score_2staget2])  # 获取每个类别的FPR和TPR
    fpr_BertDAt2[i], tpr_BertDAt2[i], _ = roc_curve([row[i] for row in y_test2],
                                                [row[i] for row in y_score_BertDAt2])  # 获取每个类别的FPR和TPR
    fpr_Promptt2[i], tpr_Promptt2[i], _ = roc_curve([row[i] for row in y_test2],
                                                [row[i] for row in y_score_Promptt2])  # 获取每个类别的FPR和TPR

    roc_auc_2stage[i] = auc(fpr_2stage[i], tpr_2stage[i])  # 计算AUC值
    roc_auc_BertDA[i] = auc(fpr_BertDA[i], tpr_BertDA[i])  # 计算AUC值
    roc_auc_Prompt[i] = auc(fpr_Prompt[i], tpr_Prompt[i])  # 计算AUC值

    roc_auc_2staget2[i] = auc(fpr_2staget2[i], tpr_2staget2[i])  # 计算AUC值
    roc_auc_BertDAt2[i] = auc(fpr_BertDAt2[i], tpr_BertDAt2[i])  # 计算AUC值
    roc_auc_Promptt2[i] = auc(fpr_Promptt2[i], tpr_Promptt2[i])  # 计算AUC值

# Compute micro-average ROC curve and ROC area（方法二）
fpr_2stage["micro"], tpr_2stage["micro"], _ = roc_curve(np.array(y_test1).ravel(), np.array(y_score_2stage).ravel())
roc_auc_2stage["micro"] = auc(fpr_2stage["micro"], tpr_2stage["micro"])

fpr_BertDA["micro"], tpr_BertDA["micro"], _ = roc_curve(np.array(y_test1).ravel(), np.array(y_score_BertDA).ravel())
roc_auc_BertDA["micro"] = auc(fpr_BertDA["micro"], tpr_BertDA["micro"])

fpr_Prompt["micro"], tpr_Prompt["micro"], _ = roc_curve(np.array(y_test1).ravel(), np.array(y_score_Prompt).ravel())
roc_auc_Prompt["micro"] = auc(fpr_Prompt["micro"], tpr_Prompt["micro"])


fpr_2staget2["micro"], tpr_2staget2["micro"], _ = roc_curve(np.array(y_test2).ravel(), np.array(y_score_2staget2).ravel())
roc_auc_2staget2["micro"] = auc(fpr_2staget2["micro"], tpr_2staget2["micro"])

fpr_BertDAt2["micro"], tpr_BertDAt2["micro"], _ = roc_curve(np.array(y_test2).ravel(), np.array(y_score_BertDAt2).ravel())
roc_auc_BertDAt2["micro"] = auc(fpr_BertDAt2["micro"], tpr_BertDAt2["micro"])

fpr_Promptt2["micro"], tpr_Promptt2["micro"], _ = roc_curve(np.array(y_test2).ravel(), np.array(y_score_Promptt2).ravel())
roc_auc_Promptt2["micro"] = auc(fpr_Promptt2["micro"], tpr_Promptt2["micro"])


# First aggregate all false positive rates
all_fpr_2stage = np.unique(np.concatenate([fpr_2stage[i] for i in range(3)]))
all_fpr_BertDA = np.unique(np.concatenate([fpr_BertDA[i] for i in range(3)]))
all_fpr_Prompt = np.unique(np.concatenate([fpr_Prompt[i] for i in range(3)]))

all_fpr_2staget2 = np.unique(np.concatenate([fpr_2staget2[i] for i in range(3)]))
all_fpr_BertDAt2 = np.unique(np.concatenate([fpr_BertDAt2[i] for i in range(3)]))
all_fpr_Promptt2= np.unique(np.concatenate([fpr_Promptt2[i] for i in range(3)]))

mean_tpr_2stage = np.zeros_like(all_fpr_2stage)
mean_tpr_BertDA = np.zeros_like(all_fpr_BertDA)
mean_tpr_Prompt = np.zeros_like(all_fpr_Prompt)

mean_tpr_2staget2 = np.zeros_like(all_fpr_2staget2)
mean_tpr_BertDAt2 = np.zeros_like(all_fpr_BertDAt2)
mean_tpr_Promptt2 = np.zeros_like(all_fpr_Promptt2)
for i in range(3):
    mean_tpr_2stage += interp(all_fpr_2stage, fpr_2stage[i], tpr_2stage[i])
    mean_tpr_BertDA += interp(all_fpr_BertDA, fpr_BertDA[i], tpr_BertDA[i])
    mean_tpr_Prompt += interp(all_fpr_Prompt, fpr_Prompt[i], tpr_Prompt[i])

    mean_tpr_2staget2 += interp(all_fpr_2staget2, fpr_2staget2[i], tpr_2staget2[i])
    mean_tpr_BertDAt2 += interp(all_fpr_BertDAt2, fpr_BertDAt2[i], tpr_BertDAt2[i])
    mean_tpr_Promptt2 += interp(all_fpr_Promptt2, fpr_Promptt2[i], tpr_Promptt2[i])
# Finally average it and compute AUC
mean_tpr_2stage /= 3
mean_tpr_BertDA /= 3
mean_tpr_Prompt /= 3

mean_tpr_2staget2 /= 3
mean_tpr_BertDAt2 /= 3
mean_tpr_Promptt2 /= 3

fpr_2stage["macro"] = all_fpr_2stage
fpr_BertDA["macro"] = all_fpr_BertDA
fpr_Prompt["macro"] = all_fpr_Prompt

fpr_2staget2["macro"] = all_fpr_2staget2
fpr_BertDAt2["macro"] = all_fpr_BertDAt2
fpr_Promptt2["macro"] = all_fpr_Promptt2

tpr_2stage["macro"] = mean_tpr_2stage
tpr_BertDA["macro"] = mean_tpr_BertDA
tpr_Prompt["macro"] = mean_tpr_Prompt

tpr_2staget2["macro"] = mean_tpr_2staget2
tpr_BertDAt2["macro"] = mean_tpr_BertDAt2
tpr_Promptt2["macro"] = mean_tpr_Promptt2

roc_auc_2stage["macro"] = auc(fpr_2stage["macro"], tpr_2stage["macro"])
roc_auc_BertDA["macro"] = auc(fpr_BertDA["macro"], tpr_BertDA["macro"])
roc_auc_Prompt["macro"] = auc(fpr_Prompt["macro"], tpr_Prompt["macro"])

roc_auc_2staget2["macro"] = auc(fpr_2staget2["macro"], tpr_2staget2["macro"])
roc_auc_BertDAt2["macro"] = auc(fpr_BertDAt2["macro"], tpr_BertDAt2["macro"])
roc_auc_Promptt2["macro"] = auc(fpr_Promptt2["macro"], tpr_Promptt2["macro"])

# Plot all ROC curves
lw = 2

plt.figure(figsize=(15,10),dpi=500)

ax3=plt.subplot(231)
plt.plot(fpr_Prompt["micro"], tpr_Prompt["micro"],
         label='micro (area = {0:0.2f})'
               ''.format(roc_auc_Prompt["micro"]),
         color='deeppink', linestyle=':', linewidth=2)

plt.plot(fpr_Prompt["macro"], tpr_Prompt["macro"],
         label='macro (area = {0:0.2f})'
               ''.format(roc_auc_Prompt["macro"]),
         color='navy', linestyle=':', linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr_Prompt[i], tpr_Prompt[i], color=color, lw=lw,
             label='class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc_Prompt[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.title('Prompt-learning')
plt.legend(loc="lower right")

ax2=plt.subplot(232)
plt.plot(fpr_BertDA["micro"], tpr_BertDA["micro"],
         label='micro (area = {0:0.2f})'
               ''.format(roc_auc_BertDA["micro"]),
         color='deeppink', linestyle=':', linewidth=2)

plt.plot(fpr_BertDA["macro"], tpr_BertDA["macro"],
         label='macro (area = {0:0.2f})'
               ''.format(roc_auc_BertDA["macro"]),
         color='navy', linestyle=':', linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr_BertDA[i], tpr_BertDA[i], color=color, lw=lw,
             label='class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc_BertDA[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.title('BERT')
plt.legend(loc="lower right")


ax1=plt.subplot(233)
plt.plot(fpr_2stage["micro"], tpr_2stage["micro"],
         label='micro (area = {0:0.2f})'
               ''.format(roc_auc_2stage["micro"]),
         color='deeppink', linestyle=':', linewidth=2)

plt.plot(fpr_2stage["macro"], tpr_2stage["macro"],
         label='macro (area = {0:0.2f})'
               ''.format(roc_auc_2stage["macro"]),
         color='navy', linestyle=':', linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr_2stage[i], tpr_2stage[i], color=color, lw=lw,
             label='class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc_2stage[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Two-stage')
plt.legend(loc="lower right")

ax4=plt.subplot(234)
plt.plot(fpr_Promptt2["micro"], tpr_Promptt2["micro"],
         label='micro (area = {0:0.2f})'
               ''.format(roc_auc_Promptt2["micro"]),
         color='deeppink', linestyle=':', linewidth=2)

plt.plot(fpr_Promptt2["macro"], tpr_Promptt2["macro"],
         label='macro (area = {0:0.2f})'
               ''.format(roc_auc_Promptt2["macro"]),
         color='navy', linestyle=':', linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr_Promptt2[i], tpr_Promptt2[i], color=color, lw=lw,
             label='class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc_Promptt2[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.title('Prompt-learning')
plt.legend(loc="lower right")

ax5=plt.subplot(235)
plt.plot(fpr_BertDAt2["micro"], tpr_BertDAt2["micro"],
         label='micro (area = {0:0.2f})'
               ''.format(roc_auc_BertDAt2["micro"]),
         color='deeppink', linestyle=':', linewidth=2)

plt.plot(fpr_BertDAt2["macro"], tpr_BertDAt2["macro"],
         label='macro (area = {0:0.2f})'
               ''.format(roc_auc_BertDAt2["macro"]),
         color='navy', linestyle=':', linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr_BertDAt2[i], tpr_BertDAt2[i], color=color, lw=lw,
             label='class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc_BertDAt2[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.title('BERT')
plt.legend(loc="lower right")
#
ax6=plt.subplot(236)
plt.plot(fpr_2staget2["micro"], tpr_2staget2["micro"],
         label='micro (area = {0:0.2f})'
               ''.format(roc_auc_2staget2["micro"]),
         color='deeppink', linestyle=':', linewidth=2)

plt.plot(fpr_2staget2["macro"], tpr_2staget2["macro"],
         label='macro (area = {0:0.2f})'
               ''.format(roc_auc_2staget2["macro"]),
         color='navy', linestyle=':', linewidth=2)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr_2staget2[i], tpr_2staget2[i], color=color, lw=lw,
             label='class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc_2staget2[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Two-stage')
plt.legend(loc="lower right")




plt.savefig('roc.png',dpi=500,bbox_inches='tight', pad_inches=0)
plt.show()


