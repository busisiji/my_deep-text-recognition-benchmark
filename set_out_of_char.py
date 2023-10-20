with open(r"D:\Backup\Downloads\test/labels.txt", 'r', encoding='utf-8') as data:
    datalist = data.readlines()

character = ''
nSamples = len(datalist)
for i in range(nSamples):
    # imagePath, label = datalist[i].strip('\n').split('\t')
    a = datalist[i].strip('\n').split('.jpg')
    imagePath, label = datalist[i].strip('\n').split('.jpg')
    imagePath = imagePath + '.jpg'
    character += label

character = ''.join(set(character))
print(character)

# 将字符串 s 保存到文件 example.txt 中
with open('character.txt', 'w') as f:
    f.write(character)
