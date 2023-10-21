import pandas as pd

def write_result(data):

        # 将数据转换为 DataFrame
        df = pd.DataFrame(data)

        # 将数据写入 Excel 文件
        # writer = pd.ExcelWriter('result_决策树队.xlsx', encoding='utf-8')
        writer = pd.ExcelWriter('result_决策树队.xlsx', engine_kwargs={'encoding': 'utf-8'})
        df.to_excel(writer, index=False)
        writer.save()


def read_result():
        # 读取 Excel 文件
        df = pd.read_excel('result_决策树队.xlsx')

        # 打印数据
        print(df)

if __name__ == '__main__':
    read_result()