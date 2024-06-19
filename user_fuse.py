import os
import random
import xml.dom.minidom
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score
import seaborn as sns
from icecream import ic
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
import json
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence,pack_sequence
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from utils import  result_test,plot_F1,plot_loss
import time
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_path =  './data/train_set.json'              # 数据集
test_data_path =  './data/test_set.json'              # 数据集
val_data_path =  './data/val_set.json'              # 数据集
save_path = './saved_dict/C2D2E.pth'
CD_save_path = './saved_dict/user_fuse.pth'
CDbest_save_path = './saved_dict/best_fuse.pth'
pad_size = 256                               # 每篇文章处理成的长度(短填长切)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
model_name = "./model/mental_roberta_se"
num_epochs = 300
learning_rate=1e-6
input_size = 7  # 输入特征维度为7
hidden_size = 768  # 隐藏层维度为768
num_layers = 2  # LSTM层数为2
num_classes = 2

class RobertaClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(RobertaClassifier, self).__init__()
        self.model_name = "./model/mental_roberta_se"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name)
        self.fc = torch.nn.Linear(8, 7)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.logits)
        return logits


def load_dataset(json_path):
    '''
    从 JSON 文件中读取文本和标签，并返回数据集
    :param json_path: JSON 文件路径
    :return: 二元组，包含句子和标签
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaClassifier(num_classes=7).to(device)
    model.load_state_dict(torch.load(save_path))
    data_set=[]
    with open(json_path, 'r', encoding='utf-8') as f:
        pos=0
        data = json.load(f)
        print(len(data))
        random.shuffle(data)
        for i,item in enumerate(data):
            ispos = item.get('ispos', True)
            if not ispos:
                pos+=1
            writings = item.get('Writing', [])
            contents = []
            label = 1 if ispos else 0
            for j,writing in enumerate(writings):
                if j>pad_size:
                    continue
                content = writing.get('Text', '').replace('\n', '')  # 从 JSON 中读取文本内容
                if not content or content =='' or content.isspace():
                    continue
                contents.append(content)
            if len(contents)==0:
                continue
            # 将内容分成小块
            max_chunk_len = 32 # 假设每个小块最大长度为32
            chunks = [contents[i:i + max_chunk_len] for i in range(0, len(contents), max_chunk_len)]
            # 存储每个小块的预测结果
            chunk_logits = []
            for i,chunk in enumerate(chunks):
                # 将句子列表转换为模型需要的输入格式
                inputs = model.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
                # 获取 input_ids 和 attention_mask
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                # 使用 pad_sequence 对 input_ids 和 attention_mask 进行填充，使其长度一致
                input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=model.tokenizer.pad_token_id)
                attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
                # 进行模型预测
                with torch.no_grad():
                    logits = model(input_ids_padded, attention_mask_padded)
                chunk_logits.append(logits)
            combined_tensor = torch.cat(chunk_logits,dim=0)
            if len(combined_tensor)>pad_size:
                combined_tensor = combined_tensor[:pad_size, :]
            elif len(combined_tensor)<pad_size:
                combined_tensor = torch.nn.functional.pad(combined_tensor, (0, 0, 0, pad_size - len(combined_tensor)))
            data_set.append({'CDs':combined_tensor,"label":label})
        print(data_set[0])
    return data_set

class CD_dep_dataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tensor = sample["CDs"]
        label = sample["label"]
        return {"CDs": tensor, "label": label}


# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)  # 添加 dropout 层
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

def eval(model,dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    val_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['CDs'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(batch["label"].tolist())
            all_preds.extend(preds.detach().cpu().tolist())
    # print(all_labels, '\n', all_preds)
    result_test("fuse_LSTM",all_labels,all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    return accuracy, rec , accuracy*rec*2/(accuracy+rec)

def train(model,train_loader,val_loader,rate):
    class_weights = torch.tensor([1.0, rate], device='cuda:0')
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_f=0.0
    losses = []
    train_fs = []
    dev_fs = []
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['CDs'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        t_a, t_r, t_f = eval(model, train_loader)
        v_a,v_r,v_f=eval(model,val_loader)
        if v_f>best_f:
            torch.save(model.state_dict(), CD_save_path)
            best_f = v_f
        losses.append(total_loss / len(train_loader))
        train_fs.append(t_f)
        dev_fs.append(v_f)
        if (epoch + 1)%20==0:
            print(f"rate {rate} Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)},train_acc,rec,f1: { t_a, t_r, t_f },dev_acc,rec,f1: { v_a, v_r, v_f }",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plot_loss('user_fuse'+str(rate),losses,[])
    plot_F1('user_fuse'+str(rate),train_fs,dev_fs)
    return best_f

def plot_Fs(model_name, best_fs, rates):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    plt.plot(rates, best_fs, alpha=0.9, linewidth=2, label='F1 Score')
    plt.xlabel('Rate')
    plt.ylabel('F1 Score')
    plt.legend(loc='best')
    plt.title('Best F1 Scores for ' + model_name)
    plt.savefig("results/fs_" + model_name + ".png", dpi=400)
    plt.close()

def load_single(path):
    # 获取文件ID
    file_id = os.path.basename(path).split('.')[0]
    # 根据文件名判断是否为正例
    is_positive = True if 'pos' in path else False
    # 解析XML文件
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    writing_list = root.getElementsByTagName('WRITING')
    writings = []
    for item in writing_list:
        # 获取文章的标题、日期和文本内容
        title = item.getElementsByTagName('TITLE')[0].childNodes[0].data
        date = item.getElementsByTagName('DATE')[0].childNodes[0].data
        text = item.getElementsByTagName('TEXT')[0].childNodes[0].data
        # 将文章信息存储为字典
        writing = {'Title': title, 'Date': date, 'Text': text}
        writings.append(writing)
    # 将文件信息存储为字典
    file_info = {'ispos': is_positive, 'ID': file_id, 'Writing': writings, 'nums': len(writings)}
    return file_info

def process_single_data(single_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaClassifier(num_classes=7).to(device)
    model.load_state_dict(torch.load(save_path))
    writing =single_data['Writing']
    label = 1 if single_data['ispos'] else 0
    contents= []
    for j, writing in enumerate(writing):
        content = writing.get('Text', '').replace('\n', '')  # 从 JSON 中读取文本内容
        if not content or content == '' or content.isspace():
            continue
        contents.append(content)
    print(contents)
    max_chunk_len = 32  # 假设每个小块最大长度为32
    chunks = [contents[i:i + max_chunk_len] for i in range(0, len(contents), max_chunk_len)]
    # 存储每个小块的预测结果
    chunk_logits = []
    for i, chunk in enumerate(chunks):
        # 将句子列表转换为模型需要的输入格式
        inputs = model.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
        # 获取 input_ids 和 attention_mask
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        # 使用 pad_sequence 对 input_ids 和 attention_mask 进行填充，使其长度一致
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=model.tokenizer.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        # 进行模型预测
        with torch.no_grad():
            logits = model(input_ids_padded, attention_mask_padded)
        chunk_logits.append(logits)
    combined_tensor = torch.cat(chunk_logits, dim=0)
    CD_features = combined_tensor
    if len(combined_tensor) > pad_size:
        combined_tensor = combined_tensor[:pad_size, :]
    elif len(combined_tensor) < pad_size:
        combined_tensor = torch.nn.functional.pad(combined_tensor, (0, 0, 0, pad_size - len(combined_tensor)))
    final_data={'CDs': combined_tensor, "label": label}
    print(final_data , CD_features)
    return final_data , CD_features

def single_eval(model,final_data,CD_features,user_id):
    model.eval()
    CD_preds = torch.argmax(CD_features, dim=1).cpu().tolist()
    counts = Counter(CD_preds)
    num2classes = {0: 'non distorted', 1: 'Jump to conclusions', 2: 'Personalized Attribution',
                   3: 'emotional reasoning', 4: 'black and white', 5: 'Mislabeling', 6: 'overgeneralization'}
    converted_dict = {num2classes[key]: value for key, value in counts.items()}
    with torch.no_grad():
        inputs = np.expand_dims(final_data['CDs'].cpu(), axis=0)
        inputs = torch.tensor(inputs).cuda()
        print(inputs)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        print(preds.cpu().tolist())
    ans = 'True' if preds[0] == 1 else 'False'
    # 获取字典的键和值
    labels = list(converted_dict.keys())
    sizes = list(converted_dict.values())
    # 创建画布和子图
    fig, ax = plt.subplots()
    # 绘制饼状图
    wedges, _ = ax.pie(sizes, startangle=90)
    # 设置标题
    ax.set_title('Analysis of '+user_id+' cognitive distortion characteristics')
    # 添加文本标注
    ax.text(0, -1.5, "The result of depression was determined:" + ans, horizontalalignment='center', fontsize=12)
    # 创建自定义图例
    legend_labels = [f'{label}: {size}' for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="lower right")
    # 显示图形
    plt.show()
    print(converted_dict)




if __name__ =='__main__':


    # train_data = load_dataset(train_data_path)
    # train_dataset = CD_dep_dataset(train_data)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #
    # val_data = load_dataset(val_data_path)
    # val_dataset = CD_dep_dataset(val_data)
    # val_loader = DataLoader(val_dataset, batch_size=32)

    # test_data = load_dataset(test_data_path)
    # test_dataset = CD_dep_dataset(test_data)
    # test_loader = DataLoader(test_dataset, batch_size=32)

    best = []
    rates=[]
    tests = []
    best_f=0.75
    best_rate=0

    # for i in range(15,20):
    #     # 初始化模型、损失函数和优化器
    #     model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #     print(i)
    #     f1=train(model,train_loader,val_loader,i*0.25+5.0)
    #     rates.append(i*0.25+5.0)
    #     best.append(f1)
    #     if f1>best_f:
    #         best_rate=i*0.25+5.0
    #         model.load_state_dict(torch.load(CD_save_path))
    #         torch.save(model.state_dict(), CDbest_save_path)
    #     ic("best F1 in rate",i,':',f1)
    #     _,_,tf=eval(model, test_loader)
    #     tests.append(tf)
    # print(best,rates,tests)
    # plot_Fs("user_fuse",best,rates)
    # plot_Fs('fuse_test',tests,rates)
    # print(best,tests,rates)

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(torch.load(CDbest_save_path))
    # Acc, Rec, F1 = eval(model, test_loader)
    # ic(Acc, Rec, F1)


    # 单用户评测
    single_path = 'data/2017_cases/2017_cases/'
    user_ID = 'pos/test_subject1445.xml'
    single_path = single_path + user_ID
    single_user_data = load_single(single_path)
    print(single_user_data)
    s_data, CD_features = process_single_data(single_user_data)
    single_eval(model,s_data,CD_features,user_ID)