import os
import pandas as pd
import pickle
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR

from config import parse_args
from src.log import Logger
from src.utils import Jaccard,Cosine,Peason

def save_args(args, logger):
    save_args_file = os.path.join(args.root_path, "args.txt")
    line = str(args)
    with open(save_args_file, mode="w", encoding="utf-8") as wfp:
        wfp.write(line + "\n")
    logger.info("Args saved in file%s" % save_args_file)
def check_path(args):
    assert os.path.exists("./data")
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.exists(args.processed_path):
        os.mkdir(args.processed_path)
def xgb_train(args):
    root_path = os.path.join(args.log_path,"third")
    # dataset preparing
    # determine inputs dtype
    value_mol_file = os.path.join(args.raw_path, "Molecular_Descriptor.xlsx")
    admet_file = os.path.join(args.raw_path, "ADMET.xlsx")
    admet_mat_train = pd.read_excel(admet_file, sheet_name="training")
    admet_mat_test = pd.read_excel(admet_file, sheet_name="test")
    admet_mat_test_ext = admet_mat_test.copy()
    x_values = pd.read_excel(value_mol_file, sheet_name="training")
    x_values_test = pd.read_excel(value_mol_file, sheet_name="test")
    names_list = admet_mat_train.columns.values[1:]
    all_result = [["names", "Pearson", "Jaccard", "Cosine"]]

    # booster:
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': 10,
              'lambda': 10,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 1,
              'eta': 0.025,
              'seed': 0,
              'nthread': 8,
              'silent': 1,
              'gamma': 0.25,
              'learning_rate': 0.2}

    for raw_name in names_list:

        y_values = admet_mat_train[raw_name]

        length = len(x_values)
        train_len = int(0.75 * length)
        train_x = x_values.iloc[:train_len, 1:]
        value_names = train_x.columns.values

        validate_x = x_values.iloc[train_len:, 1:].values

        train_y = y_values.iloc[:train_len].values
        train_y = train_y.reshape(train_len, 1)
        validate_y = y_values.iloc[train_len:].values
        validate_y = validate_y.reshape(length - train_len, 1)
        # 算法参数
        dtrain = xgb.DMatrix(train_x.values, label=train_y)
        dvalidate = xgb.DMatrix(validate_x, label=validate_y)

        watchlist = [(dvalidate, 'val')]

        # 建模与预测:NUM_BOOST_round迭代次数和数的个数一致
        bst_model = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)
        # 对测试集进行预测
        test_x = x_values_test.iloc[:, 1:].values
        dtest = xgb.DMatrix(test_x)
        dvalidate = xgb.DMatrix(validate_x)
        test_y = bst_model.predict(dtest)
        predict_y = bst_model.predict(dvalidate)


        validate_y = validate_y.reshape(length - train_len)
        pea = Peason(predict_y, validate_y)
        jac = Jaccard(predict_y, validate_y)
        cos = Cosine(predict_y, validate_y)
        all_result.append([raw_name, pea, jac, cos])
        length = len(test_y)
        admet_mat_test[raw_name] = test_y.reshape(length, 1)
        admet_mat_test_ext[raw_name] = np.round(test_y.reshape(length, 1))
        xgb.plot_importance(bst_model,max_num_features=20)
        save_fig_result = os.path.join(root_path, "xgb_features_importance_" + raw_name + ".png")
        '''
        save_xlsx_result = os.path.join(root_path, "xgb_features_importance_" + raw_name + ".xlsx")
        output = bst_model.feature_importances_
        importance_list = ["names", "importance"]
        for k in range(len(value_names)):
            importance_list.append([value_names[k], output[k]])
        pd.DataFrame(importance_list).to_excel(save_xlsx_result, index=None)

        '''
        plt.savefig(save_fig_result)
        plt.close()
    save_test_result = os.path.join(root_path, "ADMET_xgb_result.xlsx")
    admet_mat_test.to_excel(save_test_result, index=None)
    save_test_result = os.path.join(root_path, "ADMET_xgb_result_binary.xlsx")
    admet_mat_test_ext.to_excel(save_test_result, index=None)
    save_test_result = os.path.join(root_path, "xgb_result.xlsx")
    pd.DataFrame(all_result).to_excel(save_test_result, index=None)

    save_test_params = os.path.join(root_path, "ADMET_xgb_params.txt")
    with open(save_test_params, encoding="utf8", mode="w") as wfp:
        wfp.write(str(params))
def lgb_train(args):
    # dataset preparing
    # determine inputs dtype
    value_mol_file = os.path.join(args.raw_path, "Molecular_Descriptor.xlsx")
    root_path = os.path.join(args.log_path, "third")
    admet_file = os.path.join(args.raw_path, "ADMET.xlsx")
    admet_mat_train = pd.read_excel(admet_file,sheet_name="training")
    admet_mat_test = pd.read_excel(admet_file,sheet_name="test")
    admet_mat_test_ext = admet_mat_test.copy()
    x_values = pd.read_excel(value_mol_file,sheet_name="training")
    x_values_test = pd.read_excel(value_mol_file,sheet_name="test")
    names_list = admet_mat_train.columns.values[1:]
    all_result = [["names","Pearson","Jaccard","Cosine"]]

    ### 开始训练
    params = {
        'boosting_type': 'gbdt',
        'boosting': 'dart',
        'objective': 'binary',
        'metric': 'binary_logloss',

        'learning_rate': 0.5,
        'num_leaves': 35,
        'max_depth': 6,

        'max_bin': 20,
        'min_data_in_leaf': 6,

        'feature_fraction': 0.6,
        'bagging_fraction': 1,
        'bagging_freq': 0,

        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'min_split_gain': 2
    }
    for raw_name in names_list:

        y_values = admet_mat_train[raw_name]

        length = len(x_values)
        train_len = int(0.75*length)
        train_x = x_values.iloc[:train_len, 1:]
        value_names = train_x.columns.values

        validate_x = x_values.iloc[train_len:, 1:].values

        train_y = y_values.iloc[:train_len].values
        train_y = train_y.reshape(train_len,1)
        validate_y = y_values.iloc[train_len:].values
        validate_y = validate_y.reshape(length-train_len,1)

        ### 数据转换
        lgb_train = lgb.Dataset(train_x, train_y, free_raw_data=False)
        lgb_eval = lgb.Dataset(validate_x, validate_y, reference=lgb_train,free_raw_data=False)

        gbm = lgb.train(params,                     # 参数字典
                lgb_train,                  # 训练集
                num_boost_round=2000,       # 迭代次数
                valid_sets=lgb_eval,        # 验证集
                early_stopping_rounds=5)   # 早停系数
        test_x = x_values_test.iloc[:,1:].values
        test_y = gbm.predict(test_x)
        predict_y = gbm.predict(validate_x)

        validate_y = validate_y.reshape(length - train_len)
        pea = Peason(predict_y,validate_y)
        jac = Jaccard(predict_y, validate_y)
        cos = Cosine(predict_y, validate_y)
        all_result.append([raw_name,pea,jac,cos])
        length = len(test_y)
        admet_mat_test[raw_name] = test_y.reshape(length,1)
        admet_mat_test_ext[raw_name] = np.round(test_y.reshape(length,1))
        lgb.plot_importance(gbm,max_num_features=20)
        save_fig_result = os.path.join(root_path, "lgbm_features_importance_"+raw_name+".png")
        save_xlsx_result = os.path.join(root_path, "lgbm_features_importance_" + raw_name + ".xlsx")
        output = gbm.feature_importance()
        importance_list = [["names", "importance"]]
        for k in range(len(value_names)):
            importance_list.append([value_names[k],output[k]])
        pd.DataFrame(importance_list).to_excel(save_xlsx_result,index=None)
        plt.savefig(save_fig_result)
        plt.close()
    save_test_result = os.path.join(root_path,"ADMET_lgbm_result.xlsx")
    admet_mat_test.to_excel(save_test_result,index=None)
    save_test_result = os.path.join(root_path, "ADMET_lgbm_result_binary.xlsx")
    admet_mat_test_ext.to_excel(save_test_result, index=None)
    save_test_result = os.path.join(root_path,"lgbm_result.xlsx")
    pd.DataFrame(all_result).to_excel(save_test_result,index=None)

    save_test_params = os.path.join(root_path, "ADMET_lgbm_params.txt")
    with open(save_test_params,encoding="utf8",mode="w") as wfp:
        wfp.write(str(params))

def svm_train(args):
    value_mol_file = os.path.join(args.raw_path, "Molecular_Descriptor.xlsx")
    root_path = os.path.join(args.log_path, "third")
    admet_file = os.path.join(args.raw_path, "ADMET.xlsx")
    admet_mat_train = pd.read_excel(admet_file, sheet_name="training")
    admet_mat_test = pd.read_excel(admet_file, sheet_name="test")
    admet_mat_test_ext = admet_mat_test.copy()
    x_values = pd.read_excel(value_mol_file, sheet_name="training")
    x_values_test = pd.read_excel(value_mol_file, sheet_name="test")
    names_list = admet_mat_train.columns.values[1:]
    all_result = [["names", "Pearson", "Jaccard", "Cosine"]]
    for raw_name in names_list:

        y_values = admet_mat_train[raw_name]

        length = len(x_values)
        train_len = int(0.75 * length)
        train_x = x_values.iloc[:train_len, 1:]
        value_names = train_x.columns.values
        train_x = train_x.values

        validate_x = x_values.iloc[train_len:, 1:].values

        train_y = y_values.iloc[:train_len].values
        validate_y = y_values.iloc[train_len:].values
        validate_y = validate_y.reshape(length - train_len, 1)


        svm_poly_reg = SVR(kernel="poly", degree=3, C=100, epsilon=0.1)
        test_x = x_values_test.iloc[:, 1:].values

        svm_poly_reg.fit(train_x, train_y)

        test_y = svm_poly_reg.predict(test_x)
        predict_y = svm_poly_reg.predict(validate_x)


        pea = Peason(predict_y, validate_y)
        jac = Jaccard(predict_y, validate_y)
        cos = Cosine(predict_y, validate_y)
        all_result.append([raw_name, pea, jac, cos])
        length = len(test_y)
        admet_mat_test[raw_name] = test_y.reshape(length, 1)
        admet_mat_test_ext[raw_name] = np.round(test_y.reshape(length, 1))
    save_test_result = os.path.join(root_path, "ADMET_svm_result.xlsx")
    admet_mat_test.to_excel(save_test_result, index=None)
    save_test_result = os.path.join(root_path, "ADMET_svm_result_binary.xlsx")
    admet_mat_test_ext.to_excel(save_test_result, index=None)
    save_test_result = os.path.join(root_path, "svm_result.xlsx")
    pd.DataFrame(all_result).to_excel(save_test_result, index=None)
def lgb_classifiy_train(args):
    # dataset preparing
    # determine inputs dtype
    value_mol_file = os.path.join(args.raw_path, "Molecular_Descriptor.xlsx")
    root_path = os.path.join(args.log_path, "third")
    admet_file = os.path.join(args.raw_path, "ADMET.xlsx")
    admet_mat_train = pd.read_excel(admet_file, sheet_name="training")
    admet_mat_test = pd.read_excel(admet_file, sheet_name="test")
    x_values = pd.read_excel(value_mol_file, sheet_name="training")
    x_values_test = pd.read_excel(value_mol_file, sheet_name="test")
    names_list = admet_mat_train.columns.values[1:]
    all_result = [["names", "Pearson", "Jaccard", "Cosine"]]

    params = {'task': 'train',
              'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'num_class': 2,
              'metric': 'multi_logloss',
              'learning_rate': 0.002296,
              'max_depth': 7,
              'num_leaves': 17,
              'feature_fraction': 0.4,
              'bagging_fraction': 0.6,
              'bagging_freq': 17}
    for raw_name in names_list:

        y_values = admet_mat_train[raw_name]

        length = len(x_values)
        train_len = int(0.75 * length)
        train_x = x_values.iloc[:train_len, 1:]
        value_names = train_x.columns.values

        validate_x = x_values.iloc[train_len:, 1:].values

        train_y = y_values.iloc[:train_len].values
        train_y = train_y.reshape(train_len, 1)
        validate_y = y_values.iloc[train_len:].values
        validate_y = validate_y.reshape(length - train_len, 1)

        ### 数据转换
        lgb_train = lgb.Dataset(train_x, train_y, free_raw_data=False)
        lgb_eval = lgb.Dataset(validate_x, validate_y, reference=lgb_train, free_raw_data=False)

        gbm = lgb.train(params,  # 参数字典
                        lgb_train,  # 训练集
                        num_boost_round=2000,  # 迭代次数
                        valid_sets=lgb_eval,  # 验证集
                        early_stopping_rounds=5)  # 早停系数
        test_x = x_values_test.iloc[:, 1:].values
        test_y = gbm.predict(test_x)
        predict_y = gbm.predict(validate_x)

        validate_y = validate_y.reshape(length - train_len)
        predict_y = np.argmax(predict_y,axis=1)

        pea = Peason(predict_y, validate_y)
        jac = Jaccard(predict_y, validate_y)
        cos = Cosine(predict_y, validate_y)
        all_result.append([raw_name, pea, jac, cos])
        length = len(test_y)
        admet_mat_test[raw_name] = np.argmax(test_y,axis=1).reshape(length, 1)
        lgb.plot_importance(gbm, max_num_features=20)
        save_fig_result = os.path.join(root_path, "lgbm_classifier_features_importance_" + raw_name + ".png")
        save_xlsx_result = os.path.join(root_path, "lgbm_classifier_features_importance_" + raw_name + ".xlsx")
        output = gbm.feature_importance()
        importance_list = [["names", "importance"]]
        for k in range(len(value_names)):
            importance_list.append([value_names[k], output[k]])
        pd.DataFrame(importance_list).to_excel(save_xlsx_result, index=None)
        plt.savefig(save_fig_result)
        plt.close()
    save_test_result = os.path.join(root_path, "ADMET_classifier_lgbm_result.xlsx")
    admet_mat_test.to_excel(save_test_result, index=None)
    save_test_result = os.path.join(root_path, "lgbm_classifier_result.xlsx")
    pd.DataFrame(all_result).to_excel(save_test_result, index=None)

    save_test_params = os.path.join(root_path, "ADMET_classifier_lgbm_params.txt")
    with open(save_test_params, encoding="utf8", mode="w") as wfp:
        wfp.write(str(params))
if __name__ == '__main__':
    args = parse_args()
    check_path(args)
    logger = Logger(args)
    root_path = os.path.join(args.log_path, "third")
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        lgb_train(args)
        lgb_classifiy_train(args)
        xgb_train(args)
        svm_train(args)
