import json
import random
import xml.dom.minidom
import os
import re
from xml.dom.minidom import parse
path='.\\data\\2017_cases\\2017_cases\\'
# 指定要保存的JSON文件路径
train_file_path = '.\\data\\train_set.json'
test_file_path = '.\\data\\test_set.json'
val_file_path = '.\\data\\val_set.json'

def load_path(path):
    """
    用于加载数据集
    :param path: 文件起始路径
    :param train_mode: 是否返回训练集
    :return: 数据集路径（集合）train_files
    """
    files_paths = []
    # 遍历文件夹中的文件
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith('test') or file.startswith('train'):
                files_paths.append(os.path.join(root, file))
    print("\nTrain Files:")
    for file in files_paths:
        print(file)
    return files_paths



def load_data(data_path):
    """
    加载数据集，并按照ispos分类返回两个数据集

    :param data_path: 文件路径
    :return: 正例数据集和负例数据集
    """
    pos_dataset = []
    neg_dataset = []
    files = load_path(data_path)
    for path in files:
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
        file_info = {'ispos': is_positive, 'ID': file_id, 'Writing': writings, 'nums':len(writings)}
        if is_positive:
            pos_dataset.append(file_info)
        else:
            neg_dataset.append(file_info)
    return pos_dataset, neg_dataset

def split_dataset(pos_dataset, neg_dataset, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
    """
    将正例数据集和负例数据集随机划分为训练集、测试集和验证集

    :param pos_dataset: 正例数据集
    :param neg_dataset: 负例数据集
    :param train_ratio: 训练集比例
    :param test_ratio: 测试集比例
    :param val_ratio: 验证集比例
    :return: 训练集、测试集和验证集
    """

    # 打乱数据集顺序
    random.shuffle(pos_dataset)
    random.shuffle(neg_dataset)
    # 计算各个数据集的大小
    total_size = len(pos_dataset)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    # 划分数据集
    train_set = pos_dataset[:train_size]
    test_set = pos_dataset[train_size:train_size+test_size]
    val_set = pos_dataset[train_size+test_size:train_size+test_size+val_size]
    # 计算各个数据集的大小
    total_size = len(neg_dataset)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    # 划分数据集
    train_set += neg_dataset[:train_size]
    test_set += neg_dataset[train_size:train_size + test_size]
    val_set += neg_dataset[train_size + test_size:train_size + test_size + val_size]
    print(len(pos_dataset)/len(neg_dataset))
    return train_set, test_set, val_set

if __name__ == "__main__":
    pos_dataset, neg_dataset=load_data(path)
    train_set, test_set, val_set=split_dataset(pos_dataset, neg_dataset)
    # 打印前五行
    for i in range(5):
        print(train_set[i])
    print(len(train_set), len(test_set),len(val_set))

    # 将数据集保存为JSON文件
    with open(train_file_path, 'w') as f:
        json.dump(train_set, f, indent=4)
    with open(test_file_path, 'w') as f:
        json.dump(test_set, f, indent=4)
    with open(val_file_path, 'w') as f:
        json.dump(val_set, f, indent=4)

