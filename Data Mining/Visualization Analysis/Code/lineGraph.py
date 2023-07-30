import pandas as pd
import matplotlib.pyplot as plt

# 用panda中的read_csv函数读入数据
# parse_dates：True 表示将字符串解析成时间对象
#index_col=0 表示将第一列数据作为索引
data = pd.read_csv('E:\Study\大三上\数据挖掘导论\实验三 可视化分析实验\shop_payNum_new.csv', parse_dates=True, index_col=0)
data.head() # 查看前5行数据

# 1） 绘制所有便利店的10月的客流量折线图
 # 首先找出所有便利店convenience store
convenience_store = data[data['cate_2_name'] == 'convenience store']
 # 找出便利店10月的记录
 # 因为列表中的索引为第一列日期，因此可以通过.index.month访问
Oct_payment = convenience_store[convenience_store.index.month == 10]
 # 获取索引 + 对数据进行去重（否则循环的时候相同的图像会展示很多次）
 # keep： {‘first’}保留第一次出现的重复行，删除后面的重复行
 # inplace：Ture 直接在原来的DataFrame上删除重复项
 # inplace：False 删除重复项或删除重复项后返回副本
id = Oct_payment['shop_id'].drop_duplicates(keep='first', inplace = False)

for i in id:
    temp = Oct_payment[Oct_payment['shop_id'] == i]
    temp.plot(y = 'pay_num' , kind = 'line' , title = "Convenience store traffic in October(id=" +str(i)+")")
    # 设置横坐标
    plt.xlabel('day')
    plt.show()