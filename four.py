import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import parse_args,check_path
from src.log import Logger
def calculate(args):
    PIC50 = ["MDEC-23", "SwHBa", "MLogP", "BCUTp-1h", "n6Ring", "C2SP2"]
    CACO_2 = ["ECCEN", "nsCH3", "minHBd", "nHBd", "IEpsilon_2", "MLFER_L"]
    CYP3A4 = ["SP-4", "IEpsilon_D", "HybRatio", "maxsOH", "TA_Beta_s", "nHBAcc"]
    HERG = ["MDEO-11", "Kier2", "WPATH", "minssCH2", "maxHsOH", "hmin"]
    HOB = ["BCUTc-1h", "sdO", "maxssO", "BCUTw-1h", "MDEC-44", "nHCsatu"]
    MN = ["WTPT-5", "Lipinski", "nssCH2", "nN", "maxsssCH", "TopoPSA"]
    list_values = [PIC50,CACO_2,CYP3A4,HERG,HOB,MN]
    list_names = ["PIC50","CACO_2","CYP3A4","HERG","HOB","MN"]
    filename = os.path.join(args.processed_path,"filter_fourth.xlsx")
    outdata = pd.read_excel(filename)
    randkdata = outdata['rank'].values
    root_path = os.path.join(args.log_path,"four")
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    for k in range(len(list_values)):
        try:
            for name in list_values[k]:
                y_value = outdata[name].values
                values_tmp = np.vstack([randkdata,y_value])
                values_tmp = values_tmp.T.tolist()
                values_tmp = sorted(values_tmp,key=lambda x:x[0])
                length = len(values_tmp)
                for j in range(length-1):
                    plt.plot([values_tmp[j][0],values_tmp[j+1][0]],[values_tmp[j][1],values_tmp[j+1][1]],c="b")
                fig_file = os.path.join(root_path,list_names[k]+"_"+name+".png")
                plt.savefig(fig_file)
                plt.close()
        except Exception:
             print(name)
def centering_cluster(args):
    root_path = os.path.join(args.log_path, "four")
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    filename = os.path.join(args.processed_path, "filter_fourth1.xlsx")
    outdata = pd.read_excel(filename)
    rankc_name = "relative C"
    rankc_values = outdata[rankc_name].values
    weights = np.exp(rankc_values)/(np.sum(np.exp(rankc_values)))
    out_values = outdata.iloc[:,6:].values
    name_values = outdata.iloc[:,6:].columns.values
    m_val = out_values.shape[1]
    list_all_values = [["names","values"]]
    for k in range(m_val):
        values = np.sum(out_values[:,k]*weights)
        list_all_values.append([name_values[k],values])
    save_path_file = os.path.join(root_path,"values_four.xlsx")
    pd.DataFrame(list_all_values).to_excel(save_path_file,index=None)
if __name__ == '__main__':
    args = parse_args()
    check_path(args)
    logger = Logger(args)
    #root_path = os.path.join(args.log_path, "four")
    #calculate(args)
    centering_cluster(args)
