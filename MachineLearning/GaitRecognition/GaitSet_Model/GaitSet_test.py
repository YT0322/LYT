import os
import numpy as np
from datetime import datetime
from functools import partial
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as tordata
from GaitSet_DataLoader import load_data, collate_fn_for_test
from GaitSet import GaitSetNet
from GaitSet import np2var as np2_var_Gaitset
from GaitSet_swintrans import GaitSetNet_swintransformer
from GaitSet_swintrans import np2var as np2var_GaitSet_swintrans

# 为了测试模型识别步态的效果不依赖于拍摄角度和行走条件，可以多角度识别人物步分别取3组行走条件(普通、穿大衣、携带包裹)的样本输入模型，查看该模型所计算出的生征与其他行走条件的匹配程度。
# 1.11 测试模型
print("torch v:", torch.__version__, "cuda v:", torch.version.cuda)

pathstr = './GaitDatasetB_Cutted'
label_train_num = 74  # 训练数据集的个数，剩下是测试数据集
batch_size = (2, 4)
frame_num = 30
hidden_dim = 256

# 设置处理流程
# num_workers = torch.cuda.device_count()
# print("cuda.device_count", num_workers)
# if num_workers <= 1:  # 仅有一块GPU或没有GPU，则使用CPU
#     num_workers = 0
# print("num_workers", num_workers)

dataconf = {  # 初始化数据集参数
    'dataset_path': pathstr,
    'imgresize': '64',
    'label_train_num': label_train_num,  # 训练数据集的个数，剩下的是测试数据集
    'label_shuffle': True,
}

train_source, test_source = load_data(**dataconf)

sampler_batch_size = 4  # 定义采样批次
# 初始化采样数据的二次处理函数
collate_train = partial(collate_fn_for_test, frame_num=frame_num)
# 定义数据加载器：每次迭代，按照采样器的索引在test_source中取出数据
test_loader = tordata.DataLoader(dataset=test_source, batch_size=sampler_batch_size,
                                 sampler=tordata.sampler.SequentialSampler(test_source), collate_fn=collate_train,
                                 num_workers=0)

# 实例化模型
# encoder = GaitSetNet(hidden_dim).float()
encoder=GaitSetNet_swintransformer(hidden_dim).float()
# encoder = nn.DataParallel(encoder)
encoder.cuda()
encoder.eval()

ckp = './checkpoint_swin'  # 设置模型文件路径
# save_name = '_'.join(map(str, [hidden_dim, int(np.prod(batch_size)), frame_num, 'full']))
ckpfiles = sorted(os.listdir(ckp))  # 加载模型
print("ckpfiles::::", ckpfiles)
if len(ckpfiles) > 1:
    # modecpk = ckp + '/'+ckpfiles[-1]
    modecpk = os.path.join(ckp, 'best-encoder.pt')

    encoder.load_state_dict(torch.load(modecpk), False)  # 加载最新的最好的模型文件
    print("load cpk !!! ", modecpk)
else:
    print("No  cpk!!!")


def cuda_dist(x, y):  # 计算距离
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1) \
           - 2 * torch.matmul(x,y.transpose(0,1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def de_diag(acc, each_angle=False):  # 计算多角度准确率,计算与其他拍摄角度相关的准确率
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


def evaluation(data):  # 评估模型函数
    feature, meta, label = data
    view, seq_type = [], []
    for i in meta:
        view.append(i[2])
        seq_type.append(i[1])

    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    probe_seq = [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']]  # 定义采集数据的行走条件
    gallery_seq = [['nm-01', 'nm-02', 'nm-03', 'nm-04']]  # 定义比较数据的行走条件
    num_rank = 5  # 取前5个距离最近的数据
    acc = np.zeros([len(probe_seq), view_num, view_num, num_rank])
    for (p, probe_s) in enumerate(probe_seq):  # 依次将采集的数据与比较数据相比
        for gallery_s in gallery_seq:
            # Start---获取指定条件的样本特征后，按照采集数据特征与比较数据特帧之间的距离大小匹配对应的标签，并计算其准确率。
            # 步骤如下:
            # ①计算采集数据特征与比较数据特征之间的距离。
            # ②对距离进行排序，返回最小的前5个排序索引。
            # ③按照索引从比较数据中取出前5个标签，并与采集数据中的标签做比较。
            # ④将比较结果的正确数量累加起来，使每个样本对应5个记录，分别代表前5个果中的识别正确个数。如[True,True,True,False,False]，
            #   累加后结果为[1,2,3,3,3]，表明离采集数据最近的前3个样本特征中识别出来3个正确结果，前5个样本特征中识别出来3个正确结果。
            # ⑤将累加结果与0比较，并判断每个排名中大于0的个数。
            # ⑥将排名1-5的识别正确个数分别除以采集样本个数，再乘以100，便得到每个排名的准确率
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):  # 遍历所有视角
                    gseq_mask = np.isin(seq_type, gallery_s) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]  # 取出样本特征
                    gallery_y = label[gseq_mask]  # 取出标签

                    pseq_mask = np.isin(seq_type, probe_s) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]  # 取出样本特征
                    probe_y = label[pseq_mask]  # 取出标签

                    if len(probe_x) > 0 and len(gallery_x) > 0:
                        dist = cuda_dist(probe_x, gallery_x)  # 计算特征之间的距离
                        idx = dist.sort(1)[1].cpu().numpy()  # 对距离按照由小到大排序，返回排序后的索引（【0】是排序后的值）
                        # 分别计算前五个结果的精确率：步骤③~⑥
                        rank_data = np.round(
                            np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                   0) * 100 / dist.shape[0], 2)
                        # End---获取指定条件的样本特征后，按照采集数据特征与比较数据特之间的距离大小匹配对应的标签，并计算其准确率。
                        acc[p, v1, v2, 0:len(rank_data)] = rank_data
    return acc


print('test_loader', len(test_loader))
time = datetime.now()
print('开始评估模型...')
feature_list = list()
view_list = list()
seq_type_list = list()
label_list = list()
batch_meta_list = []

# 在遍历数据集前加入了withtorch.nograd()语句。该语句可以使模型在运行时，不额外创建梯度相关的内存。
# 在显存不足的情况下，使用withtorch.nogradO语句非常重要，它可以节省系统资源。
# 虽然在实例化模型时，使用了模型的eval方法来设置模型的使用方式，但这仅注意是修改模型中具有状态分支的处理流程(如dropout或BN等)，并不会省去创建显存存放梯度的开销。
with torch.no_grad():
    for i, x in tqdm(enumerate(test_loader)):  # 遍历数据集
        batch_data, batch_meta, batch_label = x
        batch_data = np2var_GaitSet_swintrans(batch_data).float()  # [2, 212, 64, 44]

        feature = encoder(batch_data)  # 将数据载入模型 [4, 62, 64]
        feature_list.append(feature.view(feature.shape[0], -1).data.cpu().numpy())  # 保存特征结果，共sampler_batch_size 个特征
        batch_meta_list += batch_meta
        label_list += batch_label  # 保存样本标签

# 将样本特征、标签以及对应的元信息组合起来
test = (np.concatenate(feature_list, 0), batch_meta_list, label_list)
acc = evaluation(test)  # 对组合数据进行评估
print('评估完成. 耗时:', datetime.now() - time)

for i in range(1):  # 计算第一个的精确率
    print('===Rank-%d 准确率===' % (i + 1))
    print('携带包裹: %.3f,\t普通: %.3f,\t穿大衣: %.3f' % (
        np.mean(acc[0, :, :, i]),
        np.mean(acc[1, :, :, i]),
        np.mean(acc[2, :, :, i])))

for i in range(1):  # 计算第一个的精确率（除去自身的行走条件）
    print('===Rank-%d 准确率(除去自身的行走条件)===' % (i + 1))
    print('携带包裹: %.3f,\t普通: %.3f,\t穿大衣: %.3f' % (
        de_diag(acc[0, :, :, i]),
        de_diag(acc[1, :, :, i]),
        de_diag(acc[2, :, :, i])))

np.set_printoptions(precision=2, floatmode='fixed')  # 设置输出精度
for i in range(1):  # 显示多拍摄角度的详细结果
    print('===Rank-%d 的每个角度准确率 (除去自身的行走条件)===' % (i + 1))
    print('携带包裹:', de_diag(acc[0, :, :, i], True))
    print('普通:', de_diag(acc[1, :, :, i], True))
    print('穿大衣:', de_diag(acc[2, :, :, i], True))