import os

import numpy as np
import pandas as pd
def gray_model(filename,savename,rho = 0.5):
    '''
    :params: filename: 读取的数据文件
    :params: savename: 保存的文件
    :params: rho: 比重
    :params: ord: 计算的范数
    '''
    dataout = pd.read_excel(filename)
    # IC50nm,pIC50
    pIC50_IC50nm = dataout.iloc[:,1:3]
    # 其他的列的名称
    x_value_names = dataout.columns[3:].values
    x_values = dataout.iloc[:,3:]
    # 进行列归一化处理
    for name in x_value_names:
        d = x_values[name].apply(lambda x: (x - x_values[name].min()) / (x_values[name].max() - x_values[name].min()))
        x_values = x_values.drop(name, axis=1)
        x_values[name] = d
    IC_names = pIC50_IC50nm.columns.values
    for name in IC_names:
        d = pIC50_IC50nm[name].apply(lambda x: (x - pIC50_IC50nm[name].min()) / (pIC50_IC50nm[name].max() - pIC50_IC50nm[name].min()))
        pIC50_IC50nm = pIC50_IC50nm.drop(name, axis=1)
        pIC50_IC50nm[name] = d
    # 进行灰度分析模型筛选处对应的相关性值
    n_val = len(x_values)
    m_val = len(x_value_names)
    val_a_list = []
    val_b_list = []
    r_mat = np.zeros(shape=(n_val,m_val),dtype=float)
    for i in range(m_val):
        a = (pIC50_IC50nm.iloc[:,1]-x_values.iloc[:,i]).abs().min()
        val_a_list.append(a)
        b = (pIC50_IC50nm.iloc[:,1]-x_values.iloc[:,i]).abs().max()
        val_b_list.append(b)
    a = min(val_a_list)
    b = max(val_b_list)
    r_val = [["descriptor","relative score"]]
    for i in range(m_val):
        d = np.abs(pIC50_IC50nm.iloc[:,1]-x_values.iloc[:,i].to_numpy())
        v = (a+rho*b)/(d+rho*b+0.1)
        r_mat[:,i] = v
    out_v = np.mean(r_mat,axis=0)
    for i in range(m_val):
        r_val.append([x_value_names[i],out_v[i]])
    dataout = pd.DataFrame(r_val)
    dataout.to_excel(savename,index=None,header=True)

