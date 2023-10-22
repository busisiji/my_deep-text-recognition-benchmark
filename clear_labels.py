import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

# 读取标签文件
def read_label_file(label_file):
    labels = []
    texts = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            labels.append(line[0])
            texts.append(' '.join(line[1:]))
    return labels, texts

# 提取文本特征
def extract_text_features(texts):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)
    return features

# 计算异常分数
def compute_outlier_scores(features):
    clf = IsolationForest(contamination=0.1)
    scores = clf.fit_predict(features)
    return scores

# 根据异常分数排除异常标签
def exclude_outliers(labels, scores, threshold):
    new_labels = []
    for label, score in zip(labels, scores):
        if score >= threshold:
            new_labels.append(label)
    return new_labels

def remove(new_labels):
    for label in new_labels:
        file_path = label.split('\t')[0]
        # 删除文件
        os.remove(file_path)

# 主函数
def main():
    # 标签文件路径
    label_file = 'labels.txt'
    # 异常分数阈值
    threshold = -0.5

    # 读取标签文件
    labels, texts = read_label_file(label_file)

    # 提取文本特征
    features = extract_text_features(texts)

    # 计算异常分数
    scores = compute_outlier_scores(features)

    # 根据异常分数排除异常标签
    new_labels = exclude_outliers(labels, scores, threshold)
    print(new_labels)

    # 输出排除异常标签后的结果
    for label in new_labels:
        print(label)

if __name__ == '__main__':
    main()
