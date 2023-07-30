import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('E:\Study\大三上\数据挖掘导论\实验三 可视化分析实验\shop_payNum_new.csv', parse_dates=True, index_col=0)
data.head() # 查看前5行数据

# （3）选择一个商家（便利店第268号），统计每月的总客流量，绘制柱状图
convenience_store = data[data['cate_2_name'] == 'convenience store']
store59th = convenience_store[convenience_store['shop_id'] == 268]
# 按月分组
month = store59th.groupby(store59th.index.month).sum()
month.plot(y='pay_num', kind='bar', title="Total passenger traffic per month of convenience stop 268")
plt.xlabel('month')
plt.show()