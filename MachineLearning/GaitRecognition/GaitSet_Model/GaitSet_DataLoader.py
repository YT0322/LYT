import numpy as np  # 引入基础库
import os
import torch.utils.data as tordata
from PIL import Image
from tqdm import tqdm
import random


# 定义函数，加载文件夹的文件名称
# load_data函数， 分为3个步骤：
#
def load_data(dataset_path, imgresize, label_train_num, label_shuffle):  # 完成了整体数据集的封装
    # 主要分为三个步骤
    # ①以人物作为标签，将完整的数据集分为两部分，分别用于训练和测试。
    # ②分别根据训练集和测试集中的人物标签遍历文件夹，获得对应的图片文件名称。
    # ③用torch.utils.data接口将图片文件名称转化为数据集， 使其能够将图片载入并返回。
    label_str = sorted(os.listdir(dataset_path))  # 以人物为标签
    # 将不完整的样本忽略，只载入完整样本
    removelist = ['005', '026', '037','048','068','079', '088','096','109' ]  # 对数据集中样本不完整的人物标签进行过滤，留下可用样本。代码中不完整的人物标签可以通过调用load_dir函数来查找。
    for removename in removelist:
        if removename in label_str:
            label_str.remove(removename)
    print("label_str", label_str)
    # -start--------根据乱序标志来处理样本标签顺序，并将其分为训练集和测试集----
    label_index = np.arange(len(label_str))  # 序列数组
    if label_shuffle:
        # 打乱数组顺序
        np.random.seed(0)
        label_shuffle_index = np.random.permutation(len(label_str))
        train_list = label_shuffle_index[0:label_train_num]
        test_list = label_shuffle_index[label_train_num:]
    else:
        train_list = label_index[0:label_train_num]
        test_list = label_index[label_train_num:]
    # -end--------根据乱序标志来处理样本标签顺序，并将其分为训练集和测试集----
    print('train',train_list)
    print("test_list", test_list)
    # 加载人物列表中的图片文件名称
    data_seq_dir, data_label, meta_data = load_dir(dataset_path, train_list,
                                                   label_str)  # 代码调用load_dir函数，将标签列表所对应的图片文件名称载入。①
    test_data_seq_dir, test_data_label, test_meta_data = load_dir(dataset_path, test_list,
                                                                  label_str)  # 代码调用load_dir函数，将标签列表所对应的图片文件名称载入。②
    # 将图片文件名称转化为数据集
    train_source = DataSet(data_seq_dir, data_label, meta_data, imgresize,
                           True)  # 调用自定义类DataSet， 返回PyTorch支持的数据集对象，且只对训练集进行缓存处理，测试集不做缓存处理。①
    # test数据不缓存
    test_source = DataSet(test_data_seq_dir, test_data_label, test_meta_data, imgresize,
                          False)  # 调用自定义类DataSet， 返回PyTorch支持的数据集对象，且只对训练集进行缓存处理，测试集不做缓存处理。②
    return train_source, test_source


# 实现load_dir函数加载图片文件名称,
def load_dir(dataset_path, label_index, label_str):
    # 在load_dir函数中， 通过文件夹的逐级遍历， 将标签列表中每个人物的图片文件名称载入。
    # 该函数返回3个列表对象：图片文件名称、图片文件名称对应的标签索引、图片文件名称对应的元数据(人物、行走条件、拍摄角度)
    data_seq_dir, data_label, meta_data = [], [], []
    for i_label in label_index:  # 获取样本个体
        label_path = os.path.join(dataset_path, label_str[i_label])  # 拼接目录

        for _seq_type in sorted(os.listdir(label_path)):  # 获取样本类型，普通条件、穿大衣、携带物品
            seq_type_path = os.path.join(label_path, _seq_type)  # 拼接目录
            for _view in sorted(os.listdir(seq_type_path)):  # 获取拍摄角度
                _seq_dir = os.path.join(seq_type_path, _view)  # 拼接图片目录
                if len(os.listdir(_seq_dir)) > 0:  # 有图片
                    data_seq_dir.append(_seq_dir)  # 图片目录
                    data_label.append(i_label)  # 图片目录对应的标签
                    meta_data.append((label_str[i_label], _seq_type, _view))
                else:
                    print("No files:", _seq_dir)  # 输出数据集中样本不完整的标签。
                    # 当发现某个标签文件夹中没有图片时会将该标签输出。在使用时，可以先用load_dir函数将整个数据集遍历一遍， 并根据输出样本不完整的标签，回填到第18行代码。

    return data_seq_dir, data_label, meta_data  # 返回结果


# 实现定义数据类DataSet
# PyTorch提供了一个torch.utils.data接口，可以用来对数据集进行封装。
# 在实现时，只需要继承torch.utils.data.Dataset类，并重载其__getitem__方法。
# 在使用时，框架会向getitem方法传入索引index。在__getitem__方法内部，根据指定index加载数据。
class DataSet(tordata.DataLoader):
    def __init__(self, data_seq_dir, data_label, meta_data, imgresize, cache=True):  # 初始化
        self.data_seq_dir = data_seq_dir  # 存储图片文件名称
        self.data = [None] * len(self.data_seq_dir)  # 存放图片
        self.cache = cache  # 缓存标志
        self.meta_data = meta_data  # 数据的元信息
        self.data_label = np.asarray(data_label)  # 存放标签

        self.imgresize = int(imgresize)  # 载入的图片大小

        # self.cut_padding = int(float(imgresize) / 64 * 10)  # 指定图片裁剪的大小

    def load_all_data(self):  # 加载所有数据
        for i in tqdm(range(len(self.data_seq_dir))):
            self.__getitem__(i)

#=================================裁剪=============================================
    def __loader__(self, path):  # 读取图
        frame_imgs = self.img2xarray(path) / 255.0
        return frame_imgs


    def __getitem__(self, index):  # 加载指定索引数据
        if self.data[index] is None:  # 第一次加载
            data = self.__loader__(self.data_seq_dir[index])
        else:
            data = self.data[index]
        if self.cache:  # 保存到缓存里
            self.data[index] = data

        return data, self.meta_data[index], self.data_label[index]

    def img2xarray(self, file_path):  # 读取指定路径的数据
        frame_list = []  # 存放图片数据
        imgs = sorted(list(os.listdir(file_path)))
        for _img in imgs:  # 读取图片，放到数组里
            _img_path = os.path.join(file_path, _img)
            if os.path.isfile(_img_path):
                img = np.asarray(Image.open(_img_path))
                if len(img.shape) == 3:  # 加载预处理后的图片
                    frame_list.append(img[..., 0])
                else:
                    frame_list.append(img)
        return np.asarray(frame_list, dtype=np.float)  # [帧数，高，宽]

    def __len__(self):  # 计算数据集长度
        return len(self.data_seq_dir)


# 实现自定义采集器
# 步态识别模型需要通过三元损失进行训练。三元损失可以辅助模型特征提取的取向，使相同标签的特征距离更近，不同标签的特征距离更远。
# 由于三元损失需要输入的批次数据中，要包含不同标签(这样才可以使用矩阵方式进行正/负样本的采样)，需要额外对数据集进行处理。
# 这里使用自定义采样器完成含有不同标签数据的采样功能。
# torch.utils.data.sampler类需要配合torch.utils.data.Data Loader模块一起使用。
# torch.utils.data.DataLoader是PyTorch中的数据集处理接口。
# 根据torch.utils.data.sampler类的采样索引，在数据源中取出指定的数据，并放到collate_fn中进行二次处理，最终返回所需要的批次数据。
# 实现自定义采样器TripletSampler类，来从数据集中选取不同标签的索引，并将其返回。
# 再将两个collate_fn函数collate_fn_for_train、collate_fn_for_test分别用于对训练数据和测试数据的二次处理。

class TripletSample(tordata.sampler.Sampler):  # 继承torch.utils.data.sampler类，实现自定义采样器。
    # TripletSampler类的实现，在该类的初始化函数中，支持两个参数传入：数集与批次参数。其中批次参数包含两个维度的批次大小，分别是标签个数与样本个数。
    def __init__(self, dataset, batch_size):
        self.dataset = dataset  # 获得数据集
        self.batch_size = batch_size  # 获得批次参数，形状为（标签个数，样本个数）
        self.label_set = list(set(dataset.data_label))  # 标签集合

    def __iter__(self):  # 实现采样器的取值过程:从数据集中随机抽取指定个数的标签，并在每个标签中抽取指定个数的样本，最终以生成器的形式返回。
        while (True):
            sample_indices = []
            # 随机抽取指定个数的标签
            label_list = random.sample(self.label_set, self.batch_size[0])
            # 在每个标签中抽取指定个数的样本
            for _label in label_list:  # 按照标签个数循环
                data_index = np.where(self.dataset.data_label == _label)[0]
                index = np.random.choice(data_index, self.batch_size[1], replace=False)
                sample_indices += index.tolist()
            yield np.asarray(sample_indices)  # 以生成器的形式返回

    def __len__(self):
        return len(self.dataset)  # 计算长度


# 用于训练数据的采样器处理函数
def collate_fn_train(batch, frame_num):
    # collate_fn_train函数会对采样器传入的批次数据进行重组，并对每条数据按照指定帧数frame_num进行抽取。
    # 同时也要保证每条数据的帖数都大于等于帧数frame_num。如果帧数小于frame_num，则为其添加重复帧。
    batch_data, batch_label, batch_meta = [], [], []
    batch_size = len(batch)  # 获得数据条数
    for i in range(batch_size):  # 依次对每条数据进行处理
        batch_label.append(batch[i][2])  # 添加数据的标签
        batch_meta.append(batch[i][1])  # 添加数据的元信息
        data = batch[i][0]  # 获取该数据的样本信息
        if data.shape[0] < frame_num:  # 如果帧数较少，则随机加入几个
            # 复制帧，用于帧数很少的情况
            multy = (frame_num - data.shape[0]) // data.shape[0] + 1
            # 额外随机加入的帧的个数
            choicenum = (frame_num - data.shape[0]) % data.shape[0]
            choice_index = np.random.choice(data.shape[0], choicenum, replace=False)
            choice_index = list(range(0, data.shape[0])) * multy + choice_index.tolist()
        else:  # 随机抽取指定个数的帧
            choice_index = np.random.choice(data.shape[0], frame_num, replace=False)
        batch_data.append(data[choice_index])  # 增加指定个数的帧数据
    # 重新组合合成用于训练的样本数据
    batch = [np.asarray(batch_data), batch_meta, batch_label]
    return batch


def collate_fn_for_test(batch, frame_num):  # 用于测试数据的采样器处理函数
    # collate_fn_for_test函数会对采样器传入的批次数据进行重组，并按照批次数据中最大帧数进行补0对齐。
    # 同时也要保证母条数据的帧数都大于等于帧数frame_num。如果帧数小于frame_num，则为其添加重复帧。
    batch_size = len(batch)  # 获得数据的条数
    batch_frames = np.zeros(batch_size, np.int)
    batch_data, batch_label, batch_meta = [], [], []
    for i in range(batch_size):  # 依次对每条数据进行处理
        batch_label.append(batch[i][2])  # 添加数据的标签
        batch_meta.append(batch[i][1])  # 添加数据的元信息
        data = batch[i][0]  # 获取该数据的帧样本信息
        if data.shape[0] < frame_num:  # 如果帧数较少，随机加入几个
            print(batch_meta, data.shape[0])
            multy = (frame_num - data.shape[0]) // data.shape[0] + 1
            choicenum = (frame_num - data.shape[0]) % data.shape[0]
            choice_index = np.random.choice(data.shape[0], choicenum, replace=False)
            choice_index = list(range(0, data.shape[0])) * multy + choice_index.tolist()
            data = np.asarray(data[choice_index])
        batch_frames[i] = data.shape[0]  # 保证所有的都大于等于frame_num
        batch_data.append(data)
    max_frame = np.max(batch_frames)  # 获得最大的帧数
    # 对其他帧进行补0填充
    batch_data = np.asarray(
        [np.pad(batch_data[i], ((0, max_frame - batch_data[i].shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
         for i in range(batch_size)])
    # 重新组合成用于训练的样本数据
    batch = [batch_data, batch_meta, batch_label]
    return batch