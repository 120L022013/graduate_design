import random
from icecream import ic
from utils import result_test, plot_F1, plot_loss
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

train_data_path =  './data/train_set.json'              # 数据集
test_data_path =  './data/test_set.json'              # 数据集
val_data_path =  './data/val_set.json'              # 数据集
vocab_path = './model/vocab.pkl'             # 词表
save_path = './saved_dict/C2D2E.pth'
CD_save_path = './saved_dict/CD2D.pth'
UNK, PAD = '<UNK>', '<PAD>'                 # 未知字，padding符号
pad_size = 512                               # 每句话处理成的长度(短填长切)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 12
model_name = "./model/mental_roberta_se"
num_epochs = 20
learning_rate= 5e-5

class RobertaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RobertaClassifier, self).__init__()
        self.model_name = "./model/mental_roberta_se"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.config = RobertaConfig.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name)
        self.fc = nn.Linear(8, 7)  # 将模型输出维度调整为7分类

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = self.fc(outputs.logits)
        return logits

def load_dataset(json_path):
    '''
    从 JSON 文件中读取文本和标签，并返回数据集
    :param json_path: JSON 文件路径
    :return: 二元组，包含句子和标签
    '''

    with open(json_path, 'r', encoding='utf-8') as f:
        pos=0
        data = json.load(f)
        print(len(data))
        random.shuffle(data)
        contents = []
        for item in data:
            ispos = item.get('ispos', True)
            label = 1 if ispos else 0
            if not ispos:
                pos+=1
            writings = item.get('Writing', [])
            ID = item.get('ID', False)

            for writing in writings:
                content = writing.get('Text', '').replace('\n', '')  # 从 JSON 中读取文本内容
                if not content or content =='' or content.isspace():
                    continue
            contents.append((''.join(content), label, ID))
        ic(pos)
    return contents


class TextDataset(Dataset):
    def __init__(self, data):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        self.y = torch.LongTensor([x[1] for x in data]).to(self.device)
        self.z = torch.LongTensor([x[2] for x in data]).to(self.device)
    def __getitem__(self,index):
        text = self.x[index]
        label = self.y[index]
        ID = self.z[index]
        return {"text": text, "label": label, "ID": ID}
    def __len__(self):
        return len(self.x)

# class TextDataset(Dataset):
#     def __init__(self, data):
#         self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
#         self.y = torch.LongTensor([x[1] for x in data]).to(self.device)
#     def __getitem__(self,index):
#         text = self.x[index]
#         label = self.y[index]
#         return {"text": text, "label": label}
#     def __len__(self):
#         return len(self.x)

class CD_drepression(nn.Module):
    def __init__(self):
        super(CD_drepression,self).__init__()
        self.model=RobertaClassifier(num_classes=7)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model.load_state_dict(torch.load(save_path))
        self.fc=nn.Linear(7, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = self.fc(outputs)
        logits = self.sigmoid(logits)
        return logits


def train(model, train_dataloader, val_dataloader):
    # 定义损失函数和优化器
    class_weights = torch.tensor([1.0, 5.50], device='cuda:0')
    print(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # 训练模式
    model.train()
    best_f1 = 0
    train_losses = []
    train_f1s = []
    dev_f1s = []

    # 训练模型
    for epoch in range(num_epochs):
        total_loss = 0
        i = 0
        # 遍历数据批次
        for batch in train_dataloader:
            i+=1
            # 将数据移动到指定设备
            input_ids = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")[
                "input_ids"].to(device)
            attention_mask = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")[
                "attention_mask"].to(device)
            labels = batch[1].to(device)

            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i%10==0:
                print(i,'loss:',total_loss/i,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        # 打印每个 epoch 的损失
        _ , _,train_f1 =eval(model, train_dataloader)
        acc, rec, f1 = eval(model, val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_dataloader)},F1: {f1}")
        dev_f1s.append(f1)
        train_losses.append(total_loss / len(train_dataloader))
        train_f1s.append(train_f1)

        if f1 > best_f1:
            torch.save(model.state_dict(), CD_save_path)
            best_f1 = f1
            print("已更新", best_f1)

    plot_F1('fuse',train_f1s,dev_f1s)
    plot_loss('fuse',train_losses,[])

def eval(model, test_dataloader):
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")[
                "input_ids"].to(device)
            attention_mask = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")[
                "attention_mask"].to(device)
            labels = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(batch[1].tolist())
            all_preds.extend(preds.detach().cpu().tolist())
    print(all_labels, '\n', all_preds)
    result_test('fuse',all_labels,all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = (recall*accuracy*2)/(recall+accuracy)
    print('accuracy, recall, f1',accuracy, recall, f1)
    return accuracy, recall, f1

if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    depression_train = load_dataset(train_data_path)
    depression_dev = load_dataset(val_data_path)
    depression_test = load_dataset(test_data_path)
    train_dataloader = DataLoader(depression_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(depression_dev, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(depression_test, batch_size=batch_size, shuffle=True)

    model = CD_drepression().to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    # train(model, train_dataloader, val_dataloader)
    model.load_state_dict(torch.load(CD_save_path))
    test_acc = eval(model, test_dataloader)