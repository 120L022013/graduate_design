import torch.optim as optim
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaModel,AutoConfig
from sklearn.metrics import accuracy_score,recall_score
from utils import result_test, plot_F1, plot_loss
import json
import numpy as np
from datetime import datetime
import pytz


def get_beijing_time():
    # 获取当前时间
    current_time = datetime.now()

    # 创建北京时区对象
    beijing_tz = pytz.timezone('Asia/Shanghai')

    # 将当前时间转换为北京时区时间
    beijing_time = current_time.astimezone(beijing_tz)

    return beijing_time
# 加载数据集
train_file_path = '.\\data\\train_set.json'
test_file_path = '.\\data\\test_set.json'
val_file_path = '.\\data\\val_set.json'

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the maximum sequence length
max_length = 512

# Load the pretrained model and tokenizer
pretrained_model = "./model/mental_roberta_se"
tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
model = RobertaModel.from_pretrained(pretrained_model).to(device)


beijing_time = get_beijing_time()
print("当前北京时间：", beijing_time)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# Prepare data function
def prepare_data(data):
    X = []  # Text embeddings
    y = []  # Labels
    print(len(data))
    i = 0
    for doc in data:
        for writing in doc["Writing"]:
            text = writing["Text"]
            # Tokenize and encode the text
            input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                outputs = model(input_ids)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            X.append(embeddings)
            y.append(int(doc["ispos"]))
        i += 1
        # if i==50:
        #     break
        print("finish ", i, " doc")
    X = torch.cat(X, dim=0)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    print(y)
    print(len(X),y.shape)
    return X, y

# Load data
train_data = load_data(train_file_path)
test_data = load_data(test_file_path)
val_data = load_data(val_file_path)

# Prepare data
X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)
X_val, y_val = prepare_data(val_data)

num_classes = 2
hidden_size = 384
batch_size = 64
num_epochs=30
num_layers = 2
learning_rate = 1e-4


class BERT_LSTM_Classifier(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers, bert_model_name=pretrained_model):
        super(BERT_LSTM_Classifier, self).__init__()
        self.bert = RobertaModel.from_pretrained(bert_model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # multiplied by 2 because of bidirectional LSTM
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, embeddings):
        lstm_outputs, _ = self.lstm(embeddings)  # LSTM outputs
        # Take the output of the last time step from both directions
        output = self.fc(lstm_outputs)  # Classification output
        return output

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试函数
def test(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    result_test("BERT_LSTM",all_labels,all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    return accuracy, recall, accuracy * recall*2/(accuracy+recall)

# 数据准备
train_data_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True)

# 模型初始化
model = BERT_LSTM_Classifier(num_classes, hidden_size, num_layers).to(device)

# 定义损失函数和优化器
class_weights= torch.tensor([1.0, 9.0], device='cuda:0')
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
best_f=0
train_losses=[]
train_f1 = []
val_f1=[]
# 训练循环
for epoch in range(num_epochs):
    train_loss = train(model, train_data_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
    train_losses.append(train_loss)
    _, _, F1 = test(model, train_data_loader, device)
    train_f1.append(F1)
    accuracy, recall, F1 = test(model, val_data_loader, device)
    val_f1.append(F1)
    print(f"Val Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {F1:.4f}")
    if F1>best_f:
        best_f=F1
        print("F1 已更新",best_f)
        torch.save(model.state_dict(), './saved_dict/bert_LSTM.pth')
    train_accuracy, train_recall, train_F1 = test(model, train_data_loader, device)
    print(f"Train Accuracy: {train_accuracy:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_F1:.4f}")
plot_loss('BERT_LSTM',train_losses,[])
plot_F1('BERT_LSTM',train_f1,val_f1)
# 在测试集上评估模型
model.load_state_dict(torch.load('./saved_dict/bert_LSTM.pth'))
test_accuracy,recall,F1 = test(model, test_data_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {F1:.4f}")
torch.save(model.state_dict(), './saved_dict/bert_LSTM.pth')
beijing_time = get_beijing_time()
print("当前北京时间：", beijing_time)