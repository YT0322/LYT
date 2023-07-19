import os
import numpy as np
from datetime import datetime
import sys
from functools import partial
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch
import torch.utils.data as tordata
from torch.optim import Adam
from GaitSet_DataLoader import load_data  # 加载项目模块
from GaitSet_DataLoader import TripletSample, collate_fn_train
from GaitSet_swintrans import GaitSetNet_swintransformer, TripletLoss, np2var
#==================================================================
# 数据集
# 在完成数据集的制作之后，对其进行训练。
# 将样本文件夹perdata放到当前目录下，并编写代码生成数据集对象。
# 从数据集对象中取出一条数据，并显示该数据的详细内容。

# 输出当前CPU-GPU
print("torch V", torch.__version__, "cuda V", torch.version.cuda)
pathstr = './GaitDatasetB_Cutted'      #数据集路径


###训练超参数
label_train_num =74 # 训练集的个数(ST:24 MT:62 LT:74)。 default:74
batch_size = (2,4)  # 定义批次（2个人，每个人16个数据）24    default:p=8,k=16 128 必须两个人以上一起训练
frame_num = 30  # 定义输入集合帧图片                      default: 30
hidden_dim = 256  #
margin=0.2  #一般取0.2 三元组损失边际
learning_rate=1e-4 #学习率

dataconf = {  # 方便导入参数
    'dataset_path': pathstr,
    'imgresize': '64',
    'label_train_num': label_train_num,
    'label_shuffle': True,
}
print("加载训练数据...")
train_source, test_cource = load_data(**dataconf)  # 一次全载入，经过load_data()分别生成训练和测试数据集对象。
print("训练数据集长度", len(train_source))  # label_num * type 10* view 11
# 显示数据集里面的标签
train_label_set = set(train_source.data_label)
print("训练数据集里面的标签：", train_label_set)

dataimg, matedata, lebelimg = train_source.__getitem__(4)  # 从数据集中获取一条数据，并显示其详细信息。

print("图片样本数据形状：", dataimg.shape, " 数据的元信息：", matedata, " 数据标签索引：", lebelimg)


# 实例化采样器：得到对象triplet_sampler。
# print("得到对象triplet_sampler")
triplet_sampler = TripletSample(train_source, batch_size)
# 初始化采样器的处理函数：用偏函数的方法对采样器的处理函数进行初始化。
# print("用偏函数的方法对采样器的处理函数进行初始化")
collate_train = partial(collate_fn_train, frame_num=frame_num)

# 定义数据加载器：每次迭代，按照采样器的索引在train_source中取出数据
# 将对象triplet_sampler和采样器的处理函数collate_train传入tordata.DataLoader，得到一个可用于训练的数据加载器对象train_loader。
# 同时对数据加载器额外启动进程的数量进行了设置，如果额外启动进程的数量num_workers是0，则在加载数据时不额外启动其他进程。

#window系统下num_workers必须为0
train_loader = tordata.DataLoader(dataset=train_source, batch_sampler=triplet_sampler, collate_fn=collate_train,
                                  num_workers=0,pin_memory=True)

# 从数据加载器中取出一条数据
batch_data, batch_meta, batch_label = next(iter(train_loader))
print("该批次数据的总长度：", len(batch_data))  # 输出该数据的详细信息
print("每条数据的形状为", batch_data.shape)
print(batch_label)  # 输出该数据的标签
#===========================================================================================================================================================
# 训练模型并保存权重文件:实例化模型类，并遍历数据加载器，进行训练。
encoder = GaitSetNet_swintransformer(hidden_dim)
# encoder = nn.DataParallel(encoder)  # 使用多卡并行训练
encoder.cuda("cuda:1")  # 将模型转储到GPU
encoder.train()  # 设置模型为训练模型


optimizer = Adam(encoder.parameters(), lr=learning_rate)  # 定义Adam优化器

TripletLossmode = 'full'  # 设置三元损失的模式
triplet_loss = TripletLoss(int(np.prod(batch_size)), TripletLossmode, margin=margin)  # 实例化三元损失，编辑为0.2
# triplet_loss = nn.DataParallel(triplet_loss)  # 使用多卡并行训练
triplet_loss.cuda("cuda:1")  # 将模型转储到GPU

ckp = 'checkpoint_swin'  # 设置模型名称
os.makedirs(ckp, exist_ok=True)
# save_name = '_'.join(map(str, [hidden_dim, int(np.prod(batch_size)), frame_num, 'full']))

ckpfiles = sorted(os.listdir(ckp))  # 载入预训练模型
if len(ckpfiles) > 1:
    modecpk = os.path.join(ckp, "best-encoder.pt")
    optcpk = os.path.join(ckp, "best-optimizer.pt")
    encoder.load_state_dict(torch.load(modecpk))  # 加载模型文件
    optimizer.load_state_dict(torch.load(optcpk))
    print("load cpk !!! ", modecpk)

#===========================================================================================================================================================
# 定义训练参数
hard_loss_metric = []
full_loss_metric = []
full_loss_num = []
dist_list = []
mean_dist = 0.00    #平均
restore_iter = 0
total_iter = 160000# 迭代次数
lastloss = 65535  # 初始的损失值
trainloss = []

_time1 = datetime.now()  # 计算迭代时间
for batch_data, batch_meta, batch_label in train_loader:
    restore_iter += 1
    print("迭代轮次：{}".format(restore_iter))
    # print(batch_data.shape)
    batch_data = np2var(batch_data).float()# torch.cuda.DoubleTensor变为torch.cuda.FloatTensor
    feature = encoder(batch_data)  # 将标签转为张量
    # 将标签转化为张量
    target_label = np2var(np.array(batch_label)).long()     # p*k个标签
    triplet_feature = feature.permute(1, 0, 2).contiguous()  # 对特征结果进行变形，形状变为[62, p*k, 256]
    triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)  # 复制62份标签，[62, p*k]
    # 计算三元损失
    (full_loss_metric_, mean_dist_, full_loss_num_) = triplet_loss(triplet_feature, triplet_label)
    if triplet_loss.hard_or_full == 'full':  # 提取损失值
        loss = full_loss_metric_.mean()
    trainloss.append(loss.data.cpu().numpy())  # 保存损失值

    full_loss_metric.append(full_loss_metric_.mean().data.cpu().numpy())
    full_loss_num.append(full_loss_num_.mean().data.cpu().numpy())
    dist_list.append(mean_dist_.mean().data.cpu().numpy())

    # if loss > 1e-9:  # 若损失值过小，则不参加反向传播
    #     loss.backward()
    #     optimizer.step()
    #
    # else:
    #     print("损失值过小：", loss.item())
    optimizer.zero_grad()  # 梯度清零
    loss.backward()
    optimizer.step()


    if restore_iter % 1000 == 0:
        print("restore_iter 1000 time:", datetime.now() - _time1)
        _time1 = datetime.now()

    if restore_iter % 100 == 0:  # 输出训练结果
        print('iter {}:'.format(restore_iter), end='')
        if triplet_loss.hard_or_full == 'full':
            print(', full_loss_metric={0:.8f}'.format(np.mean(full_loss_metric)), end='')
            print(', full_loss_num={0:.8f}'.format(np.mean(full_loss_num)), end='')

        print(', mean_dist={0:.8f}'.format(np.mean(dist_list)), end='')
        print(', lr=%f' % optimizer.param_groups[0]['lr'], end='')
        print(', hard or full=%r' % TripletLossmode)

        if lastloss > np.mean(trainloss):  # 保存模型
            print("lastloss:", lastloss, " loss:", np.mean(trainloss), "need save!")
            lastloss = np.mean(trainloss)
            # modecpk = os.path.join(ckp, '{}-{:0>5}-encoder.pt'.format(save_name, restore_iter))
            # optcpk = os.path.join(ckp, '{}-{:0>5}-optimizer.pt'.format(save_name, restore_iter))
            modecpk = os.path.join(ckp, 'best-encoder.pt')
            optcpk = os.path.join(ckp, 'best-optimizer.pt')
            # torch.save(encoder.module.state_dict(),modecpk)  # 一定要用encoder对象的module中的参数进行保存。否则模型数的名字中会含有“module”字符串，使其不能被非并行的模型载入。

            torch.save(encoder.state_dict(),modecpk)
            torch.save(optimizer.state_dict(),optcpk)
        else:
            print("lastloss:", lastloss, " loss:", np.mean(trainloss), "don't save")
        print("__________________")
        # sys.stdout.flush()
        full_loss_metric.clear()
        full_loss_num.clear()
        dist_list.clear()
        trainloss.clear()

    if restore_iter == total_iter:  # 如果满足迭代次数，则训练结束
        break