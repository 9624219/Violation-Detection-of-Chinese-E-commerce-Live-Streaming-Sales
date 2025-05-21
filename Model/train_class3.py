from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # （代表仅使用第0，1号GPU）
# 定义数据文件路径
train_file = ""
test_file = ""
# 使用 load_dataset 函数加载数据
dataset = load_dataset('text', data_files={'train': train_file, 'test': test_file})

# 定义一个函数来处理每一行数据
def process_data(example):
    # 分割文本和标签
    text, label = example['text'].split('\t')
    return {'text': text, 'label': int(label)}


# 应用处理函数到整个数据集
dataset = dataset.map(process_data)



# split_dataset = dataset['train'].train_test_split(test_size=0.2)
# train_subset = split_dataset['train']
# validation_subset = split_dataset['test']

#
# train_dataset = dataset['train']
# test_dataset = dataset['test']

# 定义一个函数来计算每个样本的输入长度
# def get_input_length(example):
#     return {'input_length': len(example['text'])}
# # 计算训练集中每个样本的输入长度
# train_lengths = train_dataset.map(get_input_length)
#
# # 计算验证集中每个样本的输入长度
# test_lengths = test_dataset.map(get_input_length)
# # 计算训练集中最大的输入长度
# max_train_length = max(train_lengths['input_length'])
#
# # 计算验证集中最大的输入长度
# max_test_length = max(test_lengths['input_length'])
#
# print(f"训练集中最大的输入长度: {max_train_length}")
# print(f"验证集中最大的输入长度: {max_test_length}")


# 打印数据集的前几行以检查
# print(dataset['train'][:5])
# print(dataset['test'][:5])
# print(dataset['train'][0])
from transformers import BertTokenizer, BertModel,RobertaTokenizer
# tokenizer = BertTokenizer.from_pretrained('/home/huang/dy_live/prompt_tuning/bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('/home/huang/dy_live/prompt_tuning/chinese-roberta-wwm-ext')
bert_input = tokenizer(dataset['train'][0]['text'], padding='max_length',
                       max_length=512,
                       truncation=True,
                       return_tensors="pt")  # pt表示返回tensor
# print(bert_input)
# model = BertModel.from_pretrained('/home/huang/dy_live/prompt_tuning/bert-base-chinese')
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

# 因为要进行分词，此段运行较久，约40s
train_dataset = MyDataset(dataset['train'])
test_dataset = MyDataset(dataset['test'])

# train_dataset = MyDataset(train_subset)
# test_dataset = MyDataset(validation_subset)

from torch import nn
from transformers import BertModel,RobertaModel

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        # self.bert = BertModel.from_pretrained("/home/huang/dy_live/prompt_tuning/bert-base-chinese")
        self.bert = BertModel.from_pretrained("/home/huang/dy_live/prompt_tuning/chinese-roberta-wwm-ext")
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 3)
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

# 训练超参数
epoch = 5
batch_size = 32
lr = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 42
save_path = '/home/huang/dy_live/prompt_tuning/BertForClassification/bert_checkpoint/class3'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(random_seed)


def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


# 定义模型
model = BertClassifier()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)
model = model.to(device)
criterion = criterion.to(device)

# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(test_dataset, batch_size=batch_size)

# 训练
best_dev_acc = 0
for epoch_num in range(epoch):
    total_acc_train = 0
    total_loss_train = 0
    for inputs, labels in tqdm(train_loader):
        input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
        masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
        labels = labels.to(device)
        output = model(input_ids, masks)

        batch_loss = criterion(output, labels)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = (output.argmax(dim=1) == labels).sum().item()
        total_acc_train += acc
        total_loss_train += batch_loss.item()

    # ----------- 验证模型 -----------
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    all_preds = []
    all_labels = []
    # 不需要计算梯度
    with torch.no_grad():
        # 循环获取数据集，并用训练好的模型进行验证
        for inputs, labels in dev_loader:
            input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)
            preds = output.argmax(dim=1)
            batch_loss = criterion(output, labels)
            acc = (preds == labels).sum().item()
            total_acc_val += acc
            total_loss_val += batch_loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        print(f'''Epochs: {epoch_num + 1} 
          | Train Loss: {total_loss_train / len(train_dataset): .3f} 
          | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
          | Val Loss: {total_loss_val / len(test_dataset): .3f} 
          | Val Accuracy: {total_acc_val / len(test_dataset): .3f}''')

        # 保存最优的模型
        if total_acc_val / len(test_dataset) >= best_dev_acc:
            best_dev_acc = total_acc_val / len(test_dataset)
            save_model('roberta_412.pt')

    model.train()

# 保存最后的模型，以便继续训练
save_model('last.pt')
# todo 保存优化器

