import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('E:\Study\大三上\数据挖掘导论\实验三 可视化分析实验\shop_payNum_new.csv', parse_dates=True, index_col=0)
data.head() # 查看前5行数据

# （7）统计10月各个类别商店总客流量占该月总客流量的比例，绘制饼图。
monthOct = data[data.index.month == 10]
# 按类别分组
storeKind = data.groupby(data['cate_2_name'])
print(type(storeKind))
print(storeKind)
# 计算比率
cusRate = storeKind.sum() / data['pay_num'].sum()
print(type(cusRate))
print(cusRate)
# 画饼图
cusRate['pay_num'].plot(kind='pie', autopct='%.2f')# 设置为保留2位小数
plt.title("October")
plt.show()
