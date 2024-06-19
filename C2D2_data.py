import csv
import json
import chardet
import re
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    return chardet.detect(rawdata)['encoding']


def csv_to_dict(csv_file):
    data = []
    with open(csv_file, 'r', encoding='ISO-8859-1') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    return data

csv_file = ".\\data\\en_good_label_final.csv"
encoding = detect_encoding(csv_file)
print("File encoding:", encoding)
dict_data = csv_to_dict(csv_file)

output_file = ".\\data\\C2D2E.json"
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(dict_data, json_file, ensure_ascii=False, indent=4)

label_counts = {}
label_word_counts = {}

for entry in dict_data:
    label = entry["label"]
    thinking = entry["thinking"]
    word_count = len(re.findall(r'\w+', thinking))  # 使用正则表达式找出句子中的单词数量

    if label not in label_counts:
        label_counts[label] = 0
        label_word_counts[label] = 0

    label_counts[label] += 1
    label_word_counts[label] += word_count

# 计算平均单词数量
for label, count in label_counts.items():
    average_word_count = label_word_counts[label] / count
    print(f"Average word count of 'thinking' for label '{label}': {average_word_count}")

# 输出每个类别中元素的个数
for label, count in label_counts.items():
    print(f"Label: {label}, Count: {count}")

print("Data has been successfully stored in JSON format.")