import csv

# 打开CSV文件
with open('gt.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # 打开TXT文件
    with open('label.txt', 'w') as txt_file:
        # 遍历CSV文件的每一行
        for row in csv_reader:
            # 使用split()函数分割字符串
            split_string = row.split('/')
            # 获取文件名
            filename = split_string[-1]
            # 使用replace()函数替换分号为空格
            modified_string = filename.replace(";", " ")
            # 将每一行的数据写入TXT文件
            txt_file.write(','.join(row) + 'n')

