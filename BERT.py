import random
from typing import Mapping, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, BertConfig,BertTokenizer
from sklearn.metrics import accuracy_score, recall_score, f1_score
import json
from icecream import ic
from utils import result_test, plot_F1, plot_loss

# Load train, test, and validation data
train_file_path = './data/train_set.json'
test_file_path = './data/test_set.json'
val_file_path = './data/val_set.json'
MODEL = "./model/bert"

# Define batch sizes and shuffle options
batch_size = 16
shuffle_train = True
learning_rate=1e-5
num_epochs = 30
# 定义损失函数（使用类别权重）
class_weights = torch.tensor([1.0, 11.0], device='cuda:0')  # 根据数据分布设置类别权重
criterion = nn.CrossEntropyLoss(weight=class_weights)

class Bert(nn.Module):
    def __init__(self, model_name, train_data, test_data, val_data, batch_size=2, learning_rate=1e-5,
                 num_epochs=30, shuffle_train=True):

        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.shuffle_train = shuffle_train
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.max_length=512
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.config = BertConfig.from_pretrained(self.model_name)
        ic(self.config)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name,config=self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 使用交叉熵损失函数，并设置类别权重
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    # def train(self):
    #     self.model.train()  # Set the model to training mode
    #     for epoch in range(self.num_epochs):
    #         total_loss = 0
    #         num_batches = 0
    #         max_f1=0
    #         for example in self.train_data:
    #             text_list = [writing["Text"] for writing in example["Writing"]]
    #             texts = " ".join(text_list)
    #             label = int(example["ispos"])
    #
    #             encoding = self.tokenizer(
    #                 texts,
    #                 truncation=True,
    #                 padding='max_length',
    #                 max_length=self.max_length,
    #                 return_tensors='pt'
    #             )
    #
    #             input_ids = encoding['input_ids'].to(self.device)
    #             attention_mask = encoding['attention_mask'].to(self.device)
    #             labels = torch.tensor(label).unsqueeze(0).to(self.device)  # Add batch dimension
    #
    #             self.optimizer.zero_grad()  # Clear gradients
    #             outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
    #             loss = outputs.loss
    #
    #             if epoch != 0:
    #                 _, _, f = self.evaluate(self.val_data)
    #                 if f>max_f1:
    #                     max_f1=f
    #                     loss.backward()
    #                     self.optimizer.step()
    #
    #             else:
    #                 loss.backward()
    #                 self.optimizer.step()
    #             total_loss += loss.item()
    #             num_batches += 1
    #
    #         avg_train_loss = total_loss / num_batches
    #         print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
    #         ic(self.evaluate(self.test_data))

    def train(self):
        self.model.train()  # Set the model to training mode
        max_f1 = 0  # Initialize max F1 score
        train_f1=[]
        dev_f1=[]
        train_loss=[]


        for epoch in range(self.num_epochs):
            total_loss = 0
            num_batches = 0

            # Shuffle the training data if required
            if self.shuffle_train:
                random.shuffle(self.train_data)

            for batch_start in range(0, len(self.train_data), self.batch_size):
                batch_data = self.train_data[batch_start:batch_start + self.batch_size]
                batch_input_ids = []
                batch_attention_masks = []
                batch_labels = []
                print('s',len(self.train_data), self.batch_size)
                for example in batch_data:
                    text_list = [writing["Text"] for writing in example["Writing"]]
                    texts = " ".join(text_list)
                    label = int(example["ispos"])

                    encoding = self.tokenizer(
                        texts,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )

                    batch_input_ids.append(encoding['input_ids'])
                    batch_attention_masks.append(encoding['attention_mask'])
                    batch_labels.append(label)

                batch_input_ids = torch.cat(batch_input_ids, dim=0).to(self.device)
                batch_attention_masks = torch.cat(batch_attention_masks, dim=0).to(self.device)
                batch_labels = torch.tensor(batch_labels).to(self.device)

                self.optimizer.zero_grad()  # Clear gradients

                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_masks,
                                     labels=batch_labels)
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1

                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

            # Evaluate on validation set and update max F1 score and model weights if needed
            val_acc, val_rec, val_f1_score = self.evaluate(self.test_data)
            train_acc, train_rec, train_f1_s= self.evaluate(self.train_data)
            print("val_acc, val_rec, val_f1_score ",val_acc, val_rec, val_f1_score )
            print("train_acc, train_rec, train_f1_s",train_acc, train_rec, train_f1_s)
            train_f1.append(train_f1_s)
            dev_f1.append(val_f1_score)
            train_loss.append(avg_train_loss)
            if val_f1_score > max_f1:
                max_f1 = val_f1_score
                print("refreshed" ,max_f1)
                torch.save(self.model.state_dict(), "./saved_dict/bert.pth")  # Save the best model weights
        plot_F1('BERT',train_f1,dev_f1)
        plot_loss('BERT',train_loss,[])
        # Load the best model weights
        self.model.load_state_dict(torch.load("./saved_dict/bert.pth"))
        # Evaluate on test set
        ic(self.evaluate(self.test_data))


    def evaluate(self, eval_data):
        all_labels = []
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for example in eval_data:
                text_list = [writing["Text"] for writing in example["Writing"]]
                texts = " ".join(text_list)
                label = int(example["ispos"])

                encoding = self.tokenizer(
                    texts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                labels = torch.tensor(label).unsqueeze(0).to(self.device)  # Add batch dimension

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                all_labels.append(label)
                all_preds.append(preds.item() if preds.item() in [0, 1] else 1)
        print(all_labels, '\n', all_preds)
        result_test('BERT',all_labels,all_preds)
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = accuracy*recall*2/(accuracy+recall)

        return accuracy, recall, f1
    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        self.model.load_state_dict(state_dict)



# 示例数据
# Load the data
# [{
#     "ispos": False,
#     "ID": "test_subject1005",
#     "Writing": [
#         {
#             "Title": "Non-Americans of r/movies, what movie have you always assumed American culture?",
#             "Date": "2015-07-22 07:55:41",
#             "Text": "As an American, for example, I always assumed Russia exactly like it is in Rocky 4. What're your perceptions?"
#         },
#         {
#             "Title": "Non-Americans of r/movies, what movie have you always assumed American culture?",
#             "Date": "2015-07-22 07:55:41",
#             "Text": "As an American, for example, I always assumed Russia exactly like it is in Rocky 4. What're your perceptions?"
#         }
#     ],
#     "nums": 2
# },{
#     "ispos": False,
#     "ID": "test_subject1005",
#     "Writing": [
#         {
#             "Title": "Non-Americans of r/movies, what movie have you always assumed American culture?",
#             "Date": "2015-07-22 07:55:41",
#             "Text": "As an American, for example, I always assumed Russia exactly like it is in Rocky 4. What're your perceptions?"
#         },
#         {
#             "Title": "Non-Americans of r/movies, what movie have you always assumed American culture?",
#             "Date": "2015-07-22 07:55:41",
#             "Text": "As an American, for example, I always assumed Russia exactly like it is in Rocky 4. What're your perceptions?"
#         }
#     ],
#     "nums": 2
# }]

# Function to load data from JSON file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

if __name__ == '__main__':
    # 加载数据
    train_data = load_data(train_file_path)
    test_data = load_data(test_file_path)
    val_data = load_data(val_file_path)
    # 初始化模型
    sentiment_model = Bert(
        model_name=MODEL,
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        shuffle_train=shuffle_train
    )
    # 训练模型
    sentiment_model.train()
    sentiment_model.load_state_dict(torch.load("./saved_dict/bert.pth"))
    ic(sentiment_model.evaluate(train_data))
    ic(sentiment_model.evaluate(test_data))
    ic(sentiment_model.evaluate(val_data))


