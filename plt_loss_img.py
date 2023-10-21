import matplotlib.pyplot as plt
import numpy as np

def plt_loss(num,val,y1,y2):
    print(y1,y2)
    if val > 1:
        x = np.concatenate(([1], np.arange(val, num+1, val)))
    else:
        x = np.arange(val, num+1, val)

    # 创建图表和子图
    fig, ax = plt.subplots()

    # 绘制曲线
    ax.plot(x, y1, label='Loss train')
    ax.plot(x, y2, label='Loss valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')

    # 添加图例
    ax.legend(loc='upper right')

    # 保存为 png 文件
    plt.savefig('loss_决策树队.png')
