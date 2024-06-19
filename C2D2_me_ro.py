import random
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import random_split
import json
from icecream import ic

#  数据集、模型、保存地址
C2D2E_path = './data/C2D2E.json'
model_name = "./model/mental_roberta_se"
save_path = './saved_dict/C2D2E.pth'
# Define batch sizes and shuffle options
batch_size = 16
shuffle_train = True
learning_rate = 1e-5
num_epochs = 20
classes_num = 7

# 定义类别字典
classes2num = {'non distorted': 0, 'Jump to conclusions': 1, 'Personalized Attribution': 2, 'emotional reasoning': 3,
               'black and white': 4, 'Mislabeling': 5, 'overgeneralization': 6}
num2classes = {0: 'non distorted', 1: 'Jump to conclusions', 2: 'Personalized Attribution', 3: 'emotional reasoning',
               4: 'black and white', 5: 'Mislabeling', 6: 'overgeneralization'}


# Label: non distorted, Count: 1963
# Label: Jump to conclusions, Count: 1358
# Label: Personalized Attribution, Count: 647
# Label: emotional reasoning, Count: 407
# Label: black and white, Count: 496
# Label: Mislabeling, Count: 721
# Label: overgeneralization, Count: 558

class CustomDataset(Dataset):
    def __init__(self, file_path, classes):
        self.file_path = file_path
        self.classes = classes
        self.data = self.load_data()

    def load_data(self):
        with open(self.file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        thinking = entry["thinking"]
        label = entry["label"]
        label_id = self.classes[label]
        return {"thinking": thinking, "label": label_id}


# 定义 RoBERTa 模型
class RobertaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RobertaClassifier, self).__init__()
        self.model_name = "./model/mental_roberta_se"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.config = RobertaConfig.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name)
        self.fc = nn.Linear(8, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = self.fc(outputs.logits)
        logits = self.sigmoid(logits)
        return logits


def train(model, train_dataloader, val_dataloader):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # 训练模式
    model.train()
    best_acc = 0
    # 训练模型
    for epoch in range(num_epochs):
        total_loss = 0
        # 遍历数据批次
        for batch in train_dataloader:
            # 将数据移动到指定设备
            input_ids = tokenizer(batch["thinking"], padding=True, truncation=True, return_tensors="pt")[
                "input_ids"].to(device)
            attention_mask = tokenizer(batch["thinking"], padding=True, truncation=True, return_tensors="pt")[
                "attention_mask"].to(device)
            labels = batch["label"].to(device)
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 打印每个 epoch 的损失
        acc = eval(model, val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_dataloader)},Acc: {acc}")
        if acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = acc


def eval(model, eval_data):
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in eval_data:
            input_ids = tokenizer(batch["thinking"], padding=True, truncation=True, return_tensors="pt")[
                "input_ids"].to(device)
            attention_mask = tokenizer(batch["thinking"], padding=True, truncation=True, return_tensors="pt")[
                "attention_mask"].to(device)
            labels = batch["label"].to(device)
            print(labels)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(batch["label"].tolist())
            all_preds.extend(preds.detach().cpu().tolist())
    print(all_labels, '\n', all_preds)
    result_test_seven_classes("C2D2",all_labels,all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    f1=f1_score(all_labels,all_preds,average='macro')
    return accuracy , f1

def result_test_seven_classes(model_name, real, pred):
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    recall = recall_score(real, pred, average='macro') # 使用macro平均计算recall
    f1 = f1_score(real, pred, average='macro') # 使用macro平均计算F1-score
    patten = 'acc: %.4f   recall: %.4f   f1: %.4f'
    # print(patten % (acc, recall, f1,))
    labels_seven = ['non distorted', 'Jump to conclusions', 'Personalized Attribution', 'emotional reasoning',
               'black & white', 'Mislabeling', 'overgeneralization'] # 缩短了一些标签
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels_seven)
    plt.figure(figsize=(8, 6))  # 调整图像尺寸
    plt.rcParams.update({'font.size': 6})  # 调整字体大小
    disp.plot(cmap="Blues", values_format='')
    plt.xticks(rotation=45, ha='right')  # 将标签倾斜打印
    plt.tight_layout()  # 调整布局以减少白边
    plt.savefig("results/reConfusionMatrix_"+model_name+".tif", dpi=400)
    plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预训练的模型和tokenizer
    # 初始化模型和 tokenizer
    model = RobertaClassifier(num_classes=7).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # 加载数据集
    # with open('./data/C2D2E.json', "r", encoding="utf-8") as json_file:
    #     dataset = json.load(json_file)
    # # 定义要分配给训练集和测试集的样本比例
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # # 使用 random_split 函数将数据集分割成训练集和测试集
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # # 转换训练集和测试集为列表
    # train_dataset = list(train_dataset)
    # test_dataset = list(test_dataset)
    # with open('./data/CD_train.json', 'w', encoding="utf-8") as f:
    #     json.dump(train_dataset, f, ensure_ascii=False, indent=4)
    # with open('./data/CD_test.json', 'w', encoding="utf-8") as f:
    #     json.dump(test_dataset, f, ensure_ascii=False, indent=4)

    train_dataset, test_dataset = CustomDataset(file_path='./data/CD_train.json', classes=classes2num),CustomDataset(file_path='./data/CD_test.json', classes=classes2num)
    # 使用 DataLoader 加载训练集和测试集数据
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # 训练模型
    # train(model, train_dataloader, test_dataloader)
    model.load_state_dict(torch.load(save_path))
    train_acc, train_f1 = eval(model, train_dataloader)
    test_acc ,test_f1= eval(model, test_dataloader)

    print(f"final_test_Acc: {test_acc} final_test_F1: {test_f1} final_train_Acc: {train_acc} final_train_F1: {train_f1}")
