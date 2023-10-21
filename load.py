import lmdb
import sys

def loadLmdb(path = r'result\train'):
    env = lmdb.open(path,  map_size=int(1e10), readonly=True, lock=False, readahead=False, meminit=False)
    txn = env.begin()
    index = 0
    texts = []
    num = txn.stat()["entries"]
    # while 1:
    print(f'数据库有{num}行')
    for index in range(num):
        index += 1  # lmdb starts with 1
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if not label or not label_key:
            return texts
        label_key = label_key.decode('utf-8')
        label = label.decode('utf-8')
        print(label_key,label)
        texts.append(label_key+'\t'+label.strip()+'\n')
    # print(texts)
    return texts


if __name__ == '__main__':
    # loadLmdb(r'data_lmdb_release\training\MJ\MJ_train')
    # loadLmdb(r'data_lmdb_release\validation')
    # loadLmdb(r'result\train')
    # loadLmdb(r'data_lmdb_release\new')

    print(sys.argv)
    path = sys.argv[1]
    loadLmdb(path)
