from datasets import load_dataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # （代表仅使用第0，1号GPU）

test_file = ""
# 使用 load_dataset 函数加载数据
dataset = load_dataset('text', data_files={'test': test_file})

# 定义一个函数来处理每一行数据
def process_data(example):
    # 分割文本和标签
    text, label = example['text'].split('\t')
    return {'text': text, 'label': int(label)}
# 应用处理函数到整个数据集
dataset = dataset.map(process_data)


from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('/home/huang/dy_live/prompt_tuning/bert-base-chinese')
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, df):
        # tokenizer分词后可以被自动汇聚
        self.texts = [tokenizer(text,
                                padding='max_length',  # 填充到最大长度
                                max_length = 512, 	# 经过数据分析，最大长度为35
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]
        # Dataset会自动返回Tensor
        self.labels = [label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


test_dataset = MyDataset(dataset['test'])
from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("/home/huang/dy_live/prompt_tuning/bert-base-chinese")
        self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(768, 2)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
class BertClassifier2(nn.Module):
    def __init__(self):
        super(BertClassifier2, self).__init__()
        self.bert = BertModel.from_pretrained("/home/huang/dy_live/prompt_tuning/bert-base-chinese")
        self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(768, 2)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import random
import os
batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 42
save_path = '/home/huang/dy_live/prompt_tuning/BertForClassification/bert_checkpoint'
save_path2 = '/home/huang/dy_live/prompt_tuning/BertForClassification/bert_checkpoint/class3V2'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(random_seed)


# 定义模型
model = BertClassifier2()  # 是否违规
model2 = BertClassifier() # 违规类型
model.load_state_dict(torch.load(os.path.join(save_path, 'stage1.pt')))
model2.load_state_dict(torch.load(os.path.join(save_path2, 'stage2.pt')))


model = model.to(device)
model.eval()
model2 = model2.to(device)
model2.eval()

def evaluate(model, dataset):
    test_loader = DataLoader(dataset, batch_size=8)
    total_acc_test = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            test_label = test_label.to(device)
            output = model(input_id, mask)
            preds = output.argmax(dim=1)
            for i,pred in enumerate(preds):
                tmp_input_id = input_id[i].unsqueeze(0)
                tmp_mask = mask[i].unsqueeze(0)
                if pred == 1:  # 违规
                    output2 = model2(tmp_input_id, tmp_mask)
                    pred2 = output2.argmax(dim=1)

                    # 0涉嫌违规 1严重违规
                    if pred2 == 0:
                        pred2 = torch.tensor([1])
                    else:
                        pred2 = torch.tensor([2])
                    all_preds.extend(pred2.numpy())
                else:
                    pred2 = torch.tensor([0])
                    all_preds.extend(pred2.numpy())

            # all_preds.extend(preds.cpu().numpy())
            all_labels.extend(test_label.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds,average=None)
    recall = recall_score(all_labels, all_preds,average=None)
    f1 = f1_score(all_labels, all_preds,average=None)
    print(f'Test Accuracy: {acc}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')



evaluate(model, test_dataset)


