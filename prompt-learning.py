import tqdm
# from openprompt.data_utils.text_classification_dataset import ToxiProcessor,ProsProcessor
import torch
from openprompt.data_utils.utils import InputExample
from sklearn.metrics import *
import argparse
import pandas as pd
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1" 

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt.prompts import ManualTemplate, PtuningTemplate

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--model_name_or_path", default='/home/huang/dy_live/prompt_tuning/bert-base-chinese')
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--kptw_lr", default=0.05, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--learning_rate", default=4e-5, type=str)
parser.add_argument("--batch_size", default=16, type=int)
args = parser.parse_args()
import random

this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed

set_seed(15)

from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm('bert', '/home/huang/dy_live/prompt_tuning/chinese-roberta-wwm-ext')

dataset = {}

dataset = {}
dataset['all'] = []
dataset['train'] = []
dataset['validation'] = []
dataset['test'] = []
with open('data/827/train_allin_827V2.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    cur = 0
    for line in lines:
        content = line.split('\t')[0]
        label2 = line.split('\t')[1]
        if label2.strip() == '0':
            label = 0
        elif label2.strip() == '1':
            label = 1
        else:
            label = 2
        input_example = InputExample(text_a=content, label=label,
                                     guid=cur)
        cur+=1
        dataset['train'].append(input_example)


with open('data/827/test1.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    cur = 0
    for line in lines:
        content = line.split('\t')[0]
        label2 = line.split('\t')[1]
        if label2.strip() == '0':
            label = 0
        elif label2.strip() == '1':
            label = 1
        else:
            label = 2
        input_example = InputExample(text_a=content, label=label,
                                     guid=cur)
        cur+=1
        dataset['test'].append(input_example)
class_labels =['合规','涉嫌违规','严重违规']
cutoff=0.5
max_seq_l = 512
# batch_s = 8
batch_s = 16


mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(path=f"/home/huang/dy_live/prompt_tuning/scripts/manual_template_831.txt", choice=0)


myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff,
                                       pred_temp=1.0, max_token_split=-1).from_file(
    path=f"/home/huang/dy_live/prompt_tuning/scripts/expand_refine2_verbalizer.txt")




from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                       plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()


from openprompt.data_utils.data_sampler import FewShotSampler

sampler = FewShotSampler(num_examples_per_label=120, also_sample_dev=False)
dataset['validation'] = sampler(dataset['train'], seed=2024)
remaining_train_examples = [example for example in dataset['train'] if example not in dataset['validation']]

train_dataloader = PromptDataLoader(dataset=remaining_train_examples, template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    decoder_max_length=3,
                                    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")
#
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         decoder_max_length=3,
                                         batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail")


def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm.tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    print(allpreds)


    acc = accuracy_score(alllabels, allpreds)
    pre = precision_score(alllabels, allpreds,average=None)
    recall = recall_score(alllabels, allpreds,average=None)
    F1score = f1_score(alllabels, allpreds,average=None)
    cal_data = [acc, pre, recall, F1score]
    return cal_data



from transformers import AdamW, get_linear_schedule_with_warmup

loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()





no_decay = ['bias', 'LayerNorm.weight']

# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]


optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=0.05)

tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
scheduler1 = get_linear_schedule_with_warmup(
    optimizer1,
    num_warmup_steps=0, num_training_steps=tot_step)

scheduler2 = None

tot_loss = 0
log_loss = 0
best_val_acc = 0
second_val_acc = 0
best_state = None
second_state = None
for epoch in range(args.max_epochs):
    tot_loss = 0
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss = tot_loss + loss.item()
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()

    cal_data = evaluate(prompt_model, validation_dataloader, desc="Valid")
    val_acc = cal_data[0]
    if val_acc > best_val_acc:
        torch.save(prompt_model.state_dict(), f"./roberta-model/more_data/class3_831_1415.ckpt")
        best_val_acc = val_acc
    print("Epoch {}, val_acc {}".format(epoch, val_acc), flush=True)


prompt_model.load_state_dict(torch.load(f"./roberta-model/more_data/class3_831_1337.ckpt"))
prompt_model = prompt_model.cuda()
data_set = evaluate(prompt_model, test_dataloader, desc="Test")
test_acc = data_set[0]
test_pre = data_set[1]
test_recall = data_set[2]
test_F1scall = data_set[3]
content_write = "=" * 20 + "\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += f"kptw_lr {args.kptw_lr}\t"
content_write += f"bt {args.batch_size}\t"
content_write += f"lr {args.learning_rate}\t"
content_write += "\n"
content_write += f"Acc: {test_acc}\t"
content_write += f"Pre: {test_pre}\t"
content_write += f"Rec: {test_recall}\t"
content_write += f"F1s: {test_F1scall}\t"
content_write += "\n\n"
print(content_write)
