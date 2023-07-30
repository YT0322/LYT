import pandas as pd
import scipy
import matplotlib.pyplot as plt
data = pd.read_csv('E:\Study\大三上\数据挖掘导论\实验三 可视化分析实验\shop_payNum_new.csv', parse_dates=True, index_col=0)
data.head() # 查看前5行数据

# （6）选择一个商家，绘制客流量密度图。
convenience_store = data[data['cate_2_name'] == 'convenience store']
store59th = convenience_store[convenience_store['shop_id'] == 268]

store59th.plot(y='pay_num', kind = 'kde', title="Total passenger traffic per month of convenience stop 268")
plt.show()