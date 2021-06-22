import matplotlib.pyplot as mlp
from matplotlib.ticker import MultipleLocator


def read_log(file_log):   # 做了个接口函数把txt里的东西读到列表里去
    iter_list = []
    loss_list = []
    acc_list = []
    count = 0
    while True:
        line = file_log.readline()
        if line == '':
            break
        else:
            if count % 1 == 0:
                line.strip('\n')
                line.replace('|', ' ')
                list_line = line.split()
                iter_list.append(float(list_line[1]))
                loss_list.append(float(list_line[3]))
                acc_list.append(float(list_line[6][:-1]))
            count += 1
    return iter_list, loss_list, acc_list


def extend(alist):
    blist = []
    j = 0
    while True:
        if j < len(alist):
            for i in range(391):
                blist.append(alist[j])
            j += 1
        else:
            break
    return blist


def read_acc(file_acc):
    accc = []
    while True:
        line = file_acc.readline()
        if line == '':
            break
        else:
            accc.append(float(line[20:26]))
    return accc


def draw_lfc(a, b, i):
    MET = ['cutout', 'cutmix', 'mixup']
    it_list, lo_list, ac_list = read_log(a)
    fig = mlp.figure()
    draw1 = fig.add_subplot(111)
    mlp.title('data augmentation = {}'.format(MET[i]), fontsize=15)
    ax1 = draw1.plot(it_list, lo_list, label='Loss function', color='red',
               linestyle='-')   # 画 loss function
    draw1.set_ylabel('Loss function')
    draw2 = draw1.twinx()
    ax2 = draw2.plot(it_list, ac_list, label='Training Accuracy', color='blue',
               linestyle='-')   # 画 accuracy
    draw2.set_ylabel('Accuracy(%)')
    ax = mlp.gca()
    x_major_locator = MultipleLocator(10)
    ax.yaxis.set_major_locator(x_major_locator)
    draw2.set_ylim([5, 105])

    draw3 = draw1.twinx()
    accc_list = read_acc(b)
    acck = extend(accc_list)
    ax3 = draw3.plot(it_list, acck, color='purple', linestyle='-',
               label='Test Accuracy')
    x_major_locator = MultipleLocator(10)
    ax = mlp.gca()
    ax.yaxis.set_major_locator(x_major_locator)
    draw3.set_ylim([5, 105])

    lns = ax1 + ax2 + ax3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='right')   # 合并图例

    mlp.show()


def main():
    MET = ['cutout','cutmix','mixup']
    for i in range(3):
        f = open('./{}/log{}.txt'.format(MET[i],MET[i]), 'r')
        f2 = open('./{}/acc{}.txt'.format(MET[i],MET[i]), 'r')
        draw_lfc(f, f2, i)
        f.close()


main()