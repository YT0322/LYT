import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from swin_transformer import SwinTransfomer

##亚像素卷积（2016 CVPR espcn）
def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)

class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)

    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)


# 定义GaitSetNet_swintransformer类：
# ①实现3个MGP。
# ②对MGP的结果进行HPM处理。每层MGP的结构是由两个卷积层加一次下采样组成的。在主分支下采样之后，与辅助分支所提取的帧级特征加和，传入下一个MGP中。
class GaitSetNet_swintransformer(nn.Module):

    def __init__(self, hidden_dim,patch_size=2):
        super(GaitSetNet_swintransformer, self).__init__()
        self.hidden_dim = hidden_dim  # 输出的特征维度
        self.patch_size = patch_size

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.patch_merge = PixelShuffle(1 / 2)

        self.swin_transformer_1=SwinTransfomer(dim=32, depth=2, num_heads=8, window_size=7, qkv_bias=True)
        self.conv_1=nn.Sequential(
            nn.Conv2d(128,64,kernel_size=1,padding=0,stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.swin_transformer_2=SwinTransfomer(dim=64,depth=2,num_heads=8,window_size=7,qkv_bias=True)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(128)
        )


        self.swin_transformer_3 = SwinTransfomer(dim=128, depth=6, num_heads=8, window_size=7, qkv_bias=True)


        self.bin_num = [1, 2, 4, 8, 16]  # 定义HPM bin大小
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num) * 2, 128, hidden_dim)))])

    def frame_max(self, x, n):  # 用最大特征方法提取集合级特征：
        # 调用torch.max函数，实现从形状[批次个数，帧数，通道数，高度，宽度]的特征中，沿着帧维度，提取最大值，得到形状[批次个数，通道数，高度，宽度]的特征提取集合特征的过程。
        return torch.max(x.view(n, -1, x.shape[1], x.shape[2], x.shape[3]), 1)[0]  # 取max后的值

    def forward(self, xinput):  # 定义前向处理方法
        n = xinput.size()[0]  # 形状为[批次个数,帧数，高，宽]
        x = xinput.reshape(-1, 1, xinput.shape[-2], xinput.shape[-1])

        pad_l = pad_t = 0
        pad_b = (self.patch_size - xinput.shape[-2] % self.patch_size) % self.patch_size
        pad_r = (self.patch_size - xinput.shape[-1] % self.patch_size) % self.patch_size

        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        del xinput  # 删除不用的变量

        x=self.patch_embedding(x)
        x=self.swin_transformer_1(x)
        x=self.conv_1(self.patch_merge(x))
        gl=self.frame_max(x,n)

        gl=self.swin_transformer_2(gl)
        x=self.swin_transformer_2(x)
        gl=self.conv_2(self.patch_merge(gl))
        x=self.conv_2(self.patch_merge(x))
        gl=gl+self.frame_max(x,n)

        gl=self.swin_transformer_3(gl)
        x=self.swin_transformer_3(x)
        x=self.frame_max(x,n)
        gl=gl+x

        # srart-------HPM处理：按照定义的特征尺度self.bin_num，将输入特征分成不同尺度，并对每个尺度的特征进行均值和最大化计算，从而组合成新的特征，放到列表feature中。
        feature = list()  # 用于存放HPM特征
        n, c, h, w = gl.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
            z = gl.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
        # end-------HPM处理：按照定义的特征尺度self.bin_num，将输入特征分成不同尺度，并对每个尺度的特征进行均值和最大化计算，从而组合成新的特征，放到列表feature中。
        # 对HPM特征中的特征维度进行转化
        # srart-------将每个特征维度由128转化为指定的输出的特征维度hidden_dim。因为输入数据是三维的，无法直接使用全连接API，所以使用矩阵相乘的方式实现三维数据按照最后一个维度进行全连接的效果。
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()  # 62 n c
        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 0, 2).contiguous()
        # end-------将每个特征维度由128转化为指定的输出的特征维度hidden_dim。因为输入数据是三维的，无法直接使用全连接API，所以使用矩阵相乘的方式实现三维数据按照最后一个维度进行全连接的效果。
        return feature  # 返回结果



# 1.9 实现 自定义三元损失类
# 定义三元损失（TripletLoss）类， 实现三元损失的计算。具体步骤：
#   ①对输入样本中的标签进行每两个一组自由组合，生成标签矩阵，从标签矩阵中得到正/负样本对的掩码
#   ②对输入样本中的特征进行每两个一组自由组合，生成特征矩阵，计算出特征矩阵的欧氏距离。
#   ③按照正/负样本对的掩码，对带有距离的特征矩阵进行提取，得到正/负两种标签的距离。
#   ④将正/负两种标签的距离相减，再减去间隔值，得到三元损失。
class TripletLoss(nn.Module):  # 定义三元损失类
    def __init__(self, batch_size, hard_or_full, margin):  # 初始化
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin  # 正/负样本的三元损失间隔
        self.hard_or_full = hard_or_full  # 三元损失方式

    def forward(self, feature, label):  # 定义前向传播方法：
        # 接收的参数feature为模型根据输入样本所计算出来的特征。该参数的形状为[n.m.d]，n：HPM处理时的尺度个数62。m：样本个数32。d：维度256。
        # 在计算过程中，将三元损失看作n份，用矩阵的方式对每份m个样本、d维度特征做三元损失计算，最后将这n份平均。
        n, m, d = feature.size()  # 形状为[n,m,d]
        # 生成标签矩阵，并从中找出正/负样本对的编码，输出形状[n,m,m]并且展开
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2))
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2))
        hp_mask=hp_mask.view(-1)
        hn_mask=hn_mask.view(-1)

        dist = self.batch_dist(feature)  # 计算出特征矩阵的距离
        mean_dist = dist.mean(1).mean(1)  # 计算所有的平均距离
        dist = dist.view(-1)

        # # start-----计算三元损失的hard模式
        # hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
        # hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
        # hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)  # 要让间隔最小化，到0为止
        # # 对三元损失取均值，得到最终的hard模式loss[n]
        # hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)
        # # end-----计算三元损失的hard模式

        # start-----计算三元损失的full模式
        # 计算三元损失的full模型
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1,
                                                               1)  # 按照编码得到所有正向样本距离[n,m,正样本个数,1] [62, 32, 8, 1]
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1,
                                                               -1)  # 照编码得到所有负向样本距离[n,m,1,负样本个数] [62, 32, 1, 24]
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)  # 让正/负间隔最小化，到0为止 [62,32*8*24]

        # 计算[n]中每个三元损失的和
        full_loss_metric_sum = full_loss_metric.sum(1)  # 计算[62]中每个loss的和
        # 计算[n]中每个三元损失的个数(去掉矩阵对角线以及符合条件的三元损失)
        full_loss_num = (full_loss_metric != 0).sum(1).float()  # 计算[62]中每个loss的个数
        # 计算均值
        full_loss_metric_mean = full_loss_metric_sum / full_loss_num  # 计算平均值
        full_loss_metric_mean[full_loss_num == 0] = 0  # 将无效值设为0
        # end-----计算三元损失的full模式

        return full_loss_metric_mean, mean_dist, full_loss_num  # ,loss

    def batch_dist(self, x):  # 计算特征矩阵的距离
        x2 = torch.sum(x ** 2, 2)  # 平方和 [62, 32]
        # dist [62, 32, 32]
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))  # 计算特征矩阵的距离
        dist = torch.sqrt(F.relu(dist))  # 对结果进行开平方
        return dist


def ts2var(x):
    return autograd.Variable(x).cuda("cuda:0")


def np2var(x):
    return ts2var(torch.from_numpy(x))

if __name__ == '__main__':
    x=torch.randn(12,30,64,64)
    model=GaitSetNet_swintransformer(hidden_dim=256)
    y=model(x)
    print(y.shape)