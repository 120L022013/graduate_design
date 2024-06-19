# -*- coding: utf-8 -*-
import json
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import torch
from sklearn import metrics
from utils import result_test, plot_F1, plot_loss


# 超参数设置
train_data_path =  './data/train_set.json'              # 数据集
test_data_path =  './data/test_set.json'              # 数据集
val_data_path =  './data/val_set.json'              # 数据集
vocab_path = './model/vocab.pkl'             # 词表
save_path = './saved_dict/HAN.pth'        # 模型训练结果
# 文件路径
glove_file = './model/glove.6B.200d.txt'
def load_glove_model(glove_file):
    print("Loading GloVe Model...")
    word_to_vec_map = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_to_vec_map[word] = coefs
    print("Done.")
    return word_to_vec_map
# 加载GloVe模型
word_to_vec_map = load_glove_model(glove_file)

# 从加载的字典中提取词向量
word_vectors = [word_to_vec_map[word] for word in word_to_vec_map]

# 将列表转换为NumPy数组
numpy_array = np.array(word_vectors, dtype=np.float32)

# 使用NumPy数组创建PyTorch张量
embedding_pretrained = torch.tensor(numpy_array)
                                            # 预训练词向量
embed = embedding_pretrained.size(1)        # 词向量维度
dropout = 0.2                               # 随机丢弃
num_classes = 2                             # 类别数
num_epochs = 30                             # epoch数
batch_size = 128                            # mini-batch大小
pad_size = 30                               # 每句话处理成的长度(短填长切)
learning_rate = 1e-4                        # 学习率
hidden_size = 512                           # 隐藏层
num_layers = 3                              # 层数
MAX_VOCAB_SIZE = 10000                      # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'                 # 未知字，padding符号


def get_data():
    tokenizer = lambda x: [y for y in x]  # 字级别
    vocab = pkl.load(open(vocab_path, 'rb'))
    # print('tokenizer',tokenizer)
    print(f"Vocab size: {len(vocab)}")

    train = load_dataset(train_data_path, pad_size, tokenizer, vocab)
    test = load_dataset(test_data_path, pad_size, tokenizer, vocab)
    dev = load_dataset(val_data_path, pad_size, tokenizer, vocab)
    print(len(test),len(train),len(dev))
    return vocab, train, dev, test

def load_dataset(json_path, pad_size, tokenizer, vocab):
    '''
    从 JSON 文件中读取文本和标签，并返回数据集
    :param json_path: JSON 文件路径
    :param pad_size: 每个序列的大小
    :param tokenizer: 转为字级别
    :param vocab: 词向量模型
    :return: 二元组，包含字ID和标签
    '''
    contents = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            ispos = item.get('ispos', False)
            writings = item.get('Writing', [])
            for writing in writings:
                content = writing.get('Text', '')  # 从 JSON 中读取文本内容
                label = 1 if ispos else 0  # 根据 "ispos" 设置标签
                if not content:
                    continue
                words_line = []
                token = tokenizer(content)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, label))
    print(len(contents))
    return contents

class TextDataset(Dataset):
    def __init__(self, data):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        self.y = torch.LongTensor([x[1] for x in data]).to(self.device)
    def __getitem__(self,index):
        self.text = self.x[index]
        self.label = self.y[index]
        return self.text, self.label
    def __len__(self):
        return len(self.x)

# 以上是数据预处理的部分

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 定义HAN模型
class HAN(nn.Module):
    def __init__(self, word_embed_dim=200, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout):
        super(HAN, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        self.word_lstm = nn.LSTM(word_embed_dim, hidden_size, num_layers,
                                 bidirectional=True, batch_first=True, dropout=dropout)
        self.word_attention = nn.Linear(hidden_size * 2, 1)

        self.sentence_lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers,
                                     bidirectional=True, batch_first=True, dropout=dropout)
        self.sentence_attention = nn.Linear(hidden_size * 2, 1)

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 词嵌入
        word_embedded = self.word_embedding(x)

        # 词级别的注意力
        word_hidden, _ = self.word_lstm(word_embedded)
        word_attention_weights = nn.functional.softmax(self.word_attention(word_hidden), dim=1)
        word_representation = torch.sum(word_attention_weights * word_hidden, dim=1)

        # 句子级别的注意力
        sentence_hidden, _ = self.sentence_lstm(word_representation.unsqueeze(1))
        sentence_attention_weights = nn.functional.softmax(self.sentence_attention(sentence_hidden), dim=1)
        sentence_representation = torch.sum(sentence_attention_weights * sentence_hidden, dim=1)

        # 全连接层
        out = self.fc(sentence_representation)
        return out
# 权重初始化，默认xavier
# xavier和kaiming是两种初始化参数的方法
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass



# 定义训练的过程
def train( model, dataloaders):
    '''
    训练模型
    :param model: 模型
    :param dataloaders: 处理后的数据，包含trian,dev,test
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_weights = torch.tensor([1.0, 9.0], device='cuda:0')
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    dev_best_f = 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Start Training...\n")
    plot_train_F1 = []
    plot_train_loss = []
    plot_dev_F1 = []
    plot_dev_loss = []
    for i in range(num_epochs):
        # 1，训练循环----------------------------------------------------------------
        # 将数据全部取完
        # 记录每一个batch
        step = 0
        train_lossi=0
        train_acci = 0
        for inputs, labels in dataloaders['train']:
            # 训练模式，可以更新参数
            model.train()
            # print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零，防止累加
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            step += 1
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_lossi += loss.item()
            train_acci += metrics.accuracy_score(true, predic)
            # 2，验证集验证----------------------------------------------------------------
        print('dev')
        dev_acc, dev_rec, dev_f, dev_loss = dev_eval(model, dataloaders['dev'], loss_function,Result_test=True)
        if dev_f > dev_best_f:
            dev_best_f = dev_f
            print(dev_best_f,"已更新")
            torch.save(model.state_dict(), save_path)
        print('train')
        train_acc,train_rec, train_f,train_loss = dev_eval(model, dataloaders['train'], loss_function,Result_test=True)

        plot_train_F1.append(train_f)
        plot_train_loss.append(train_loss)
        plot_dev_F1.append(dev_f)
        plot_dev_loss.append(dev_loss)
        print("epoch = {} :  train_loss = {:.3f}, train_acc = {:.2%}, dev_loss = {:.3f}, dev_acc = {:.2%}, dev_rec = {:.2%}, dev_f={:.2%},".
                  format(i+1, train_loss, train_acc, dev_loss, dev_acc, dev_rec, dev_f), time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    plot_F1('HAN',plot_train_F1,plot_dev_F1)
    plot_loss('HAN',plot_train_loss,[])

    # 3，验证循环----------------------------------------------------------------
    model.load_state_dict(torch.load(save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_rec, test_f, test_loss = dev_eval(model, dataloaders['test'], loss_function,Result_test=False)
    print('================'*8)
    print(
        "final:  test_loss = {:.3f}, test_acc = {:.2%}, test_rec = {:.2%}, test_f={:.2%},".
        format( test_loss, test_acc, test_rec, test_f),
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

# 模型评估
def dev_eval(model, data, loss_function,Result_test=True):
    '''
    :param model: 模型
    :param data: 验证集集或者测试集的数据
    :param loss_function: 损失函数
    :return: 损失和准确率
    '''
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data:
            outputs = model(texts)

            loss = loss_function(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    print(labels_all,'\n',predict_all)
    acc = metrics.accuracy_score(labels_all, predict_all)
    rec = metrics.recall_score(labels_all, predict_all)
    f=(2*acc*rec)/(acc+rec)
    if Result_test:
        result_test('HAN',labels_all, predict_all)
    else:
        pass
    print(acc, rec, f)
    return acc, rec, f, loss_total / len(data)

if __name__ == '__main__':
    # 设置随机数种子，保证每次运行结果一致，不至于不能复现模型
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = get_data()
    dataloaders = {
        'train': DataLoader(TextDataset(train_data), batch_size, shuffle=True),
        'dev': DataLoader(TextDataset(dev_data), batch_size, shuffle=True),
        'test': DataLoader(TextDataset(test_data), batch_size, shuffle=True)
    }

    # # 检查数据
    # # 遍历每个数据集的数据加载器
    # for phase, loader in dataloaders.items():
    #     print(f"Dataset: {phase}")
    #     print("-" * 10)
    #     # 遍历每个批次
    #     for i, batch in enumerate(loader, 1):
    #         inputs, labels = batch
    #         print(f"Batch {i}:")
    #         print("Inputs:", inputs)  # 输出输入数据
    #         print("Labels:", labels)  # 输出标签数据
    #
    #         if i == 1:  # 只查看第一个批次的数据
    #             break
    #     print()

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = HAN().to(device)
    init_network(model)
    # train(model, dataloaders)
    class_weights = torch.tensor([1.0, 9.0], device='cuda:0')
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    model.load_state_dict(torch.load('./saved_dict/HAN.pth'))
    dev_eval(model,dataloaders['test'],loss_function)
