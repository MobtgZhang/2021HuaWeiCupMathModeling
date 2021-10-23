from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import theano

from .utils import split_train_test
from .data import MolDataSet,DataLoader
from .bpnet import BpNet
from .utils import Jaccard,Cosine,Peason
from .dbn import DBNMlp

### BP神经网络进行讨论分类处理

def test_bpnet(bpnet,test_dataset):
    train_x = test_dataset.train_x
    train_y = test_dataset.train_y
    predict_y = bpnet.forward(train_x)
    if bpnet.normalize_y:
        predict_y = test_dataset.target_transform.inverse_transform(predict_y)
    pea = Peason(train_y, predict_y)
    jac = Jaccard(train_y, predict_y)
    cos = Cosine(train_y, predict_y)
    return pea,jac,cos
def train_bpnet(args):
    processed_merge_file = os.path.join(args.processed_path, "filter_processed_new.xlsx")
    root_path = os.path.join(args.log_path, "second")
    if args.normalize_x:transform = MinMaxScaler()
    else:transform = None
    if args.normalize_x:target_transform = MinMaxScaler()
    else:target_transform = None
    if args.all_value:
        train_x,train_y = split_train_test(processed_merge_file)
        train_dataset = MolDataSet(train_x,train_y, transform=transform, target_transform=target_transform)
        train_dataloader = DataLoader(train_dataset, args.batch_size)
    else:
        (train_x,train_y),(test_x,test_y) = split_train_test(processed_merge_file,raw=False)
        train_dataset = MolDataSet(train_x, train_y, transform=transform, target_transform=target_transform)
        test_dataset = MolDataSet(test_x, test_y, transform=transform, target_transform=target_transform)
        train_dataloader = DataLoader(train_dataset, args.batch_size)
        test_dataloader = DataLoader(test_dataset, args.test_batch_size)
    if args.normalize_x:
        scalar_file_x_train = os.path.join(root_path, "scalar_x_train.pt")
        with open(scalar_file_x_train, 'wb') as f:
            pickle.dump(train_dataset.transform, f, protocol=pickle.HIGHEST_PROTOCOL)
        if not args.all_value:
            scalar_file_x_test = os.path.join(root_path, "scalar_x_test.pt")
            with open(scalar_file_x_test, 'wb') as f:
                pickle.dump(test_dataset.transform, f, protocol=pickle.HIGHEST_PROTOCOL)
    if args.normalize_y:
        scalar_file_y_train = os.path.join(root_path, "scalar_y_train.pt")
        with open(scalar_file_y_train, 'wb') as f:
            pickle.dump(train_dataset.target_transform, f, protocol=pickle.HIGHEST_PROTOCOL)
        if not args.all_value:
            scalar_file_y_test = os.path.join(root_path, "scalar_y_test.pt")
            with open(scalar_file_y_test, 'wb') as f:
                pickle.dump(test_dataset.target_transform, f, protocol=pickle.HIGHEST_PROTOCOL)

    bpnet = BpNet(args.in_size, args.hid_size, args.out_size,args.normalize_x,args.normalize_y)
    # training model
    train_data_list = []
    if not args.all_value:
        test_data_list = []
    loss_list = []
    for k in range(args.epoches):
        loss = 0
        for (x_value, y_value) in train_dataloader:
            err = bpnet.train(x_value, y_value, args.learning_rate, args.lambd)
            loss_list.append(err)
        train_dataloader.clear()
        pea_train, jac_train, cos_train = test_bpnet(bpnet, train_dataset)
        train_data_list.append([pea_train, jac_train, cos_train])
        if not args.all_value:
            pea_test, jac_test, cos_test = test_bpnet(bpnet, test_dataset)
            test_data_list.append([pea_test, jac_test, cos_test])
    loss_list = np.array(loss_list)
    x_list = np.linspace(0, len(loss_list), len(loss_list))
    plt.plot(x_list, loss_list)
    fig_file_path = os.path.join(root_path, "bpnet_loss.png")
    plt.xlabel("The dataset training times")
    plt.ylabel("The loss")
    plt.title("The training process for the BpNet model")
    plt.savefig(fig_file_path)
    plt.show()
    # draw the pictures
    x_list = np.linspace(0, len(train_data_list), len(train_data_list))
    train_data_list = np.array(train_data_list)
    if not args.all_value:test_data_list = np.array(test_data_list)
    list_names = ["peason","Jaccard","cosine"]
    types_n = len(list_names)
    for k in range(types_n):
        plt.plot(x_list, train_data_list[:, k], label="train")
        if not args.all_value:
            plt.plot(x_list, test_data_list[:, k], label="test")
        fig_file_path = os.path.join(root_path, "bpnet_"+list_names[k]+".png")
        plt.xlabel("The dataset training times")
        plt.ylabel("The "+list_names[k])
        plt.legend(loc="right")
        plt.title("The "+list_names[k]+" score for the training")
        plt.savefig(fig_file_path)
        plt.show()
        plt.close()
    model_file_path = os.path.join(root_path, "bpnet.pt")
    with open(model_file_path, 'wb') as f:
        pickle.dump(bpnet, f, protocol=pickle.HIGHEST_PROTOCOL)
def predict_bpnet(args):
    predict_file = os.path.join(args.processed_path, "filter_processed_new.xlsx")
    root_path = os.path.join(args.log_path, "second")
    bpnet_model_file = os.path.join(root_path, "bpnet.pt")
    test_data = pd.read_excel(predict_file,sheet_name="test")
    x_value = test_data.iloc[:,3:]
    with open(bpnet_model_file, 'rb') as f:
        bpnet = pickle.load(f)
    if bpnet.normalize_x:
        scalar_file_x = os.path.join(root_path, "scalar_x_train.pt")
        with open(scalar_file_x, 'rb') as f:
            scalar_x = pickle.load(f)
    if bpnet.normalize_y:
        scalar_file_y = os.path.join(root_path, "scalar_y_train.pt")
        with open(scalar_file_y, 'rb') as f:
            scalar_y = pickle.load(f)
        x_value = scalar_x.transform(x_value)
    predict_pIC50 = bpnet.forward(x_value)
    if bpnet.normalize_y:
        predict_pIC50 = scalar_y.inverse_transform(predict_pIC50)
    predict_IC50nm = np.power(10,9-predict_pIC50)
    names = ["IC50_nM","pIC50"]
    test_data.drop(names[0], axis=1)
    test_data[names[0]] = predict_IC50nm
    test_data.drop(names[1], axis=1)
    test_data[names[1]] = predict_pIC50
    save_result_file = os.path.join(root_path,"bpnet_ER_alpha_results.xlsx")
    test_data.iloc[:,:3].to_excel(save_result_file,index=None)
### DBN神经网络进行讨论分类处理
def test_dbnmlp(dbnmlpnet,test_dataset):
    train_x = test_dataset.train_x
    train_y = test_dataset.train_y
    prediction = dbnmlpnet.last_layer.output
    predict = theano.function(inputs=[dbnmlpnet.x], outputs=prediction)
    predict_y = predict(train_x)
    if dbnmlpnet.normalize_y:
        predict_y = test_dataset.target_transform.inverse_transform(predict_y)
    pea = Peason(train_y, predict_y)
    jac = Jaccard(train_y, predict_y)
    cos = Cosine(train_y, predict_y)
    return pea, jac, cos
def train_dbnmlp(args):
    processed_merge_file = os.path.join(args.processed_path, "filter_processed_new.xlsx")
    root_path = os.path.join(args.log_path, "second")
    if args.normalize_x:
        transform = MinMaxScaler()
    else:
        transform = None
    if args.normalize_x:
        target_transform = MinMaxScaler()
    else:
        target_transform = None
    if args.all_value:
        train_x, train_y = split_train_test(processed_merge_file)
        train_dataset = MolDataSet(train_x, train_y, transform=transform, target_transform=target_transform)
        train_dataloader = DataLoader(train_dataset, args.batch_size)
    else:
        (train_x, train_y), (test_x, test_y) = split_train_test(processed_merge_file, raw=False)
        train_dataset = MolDataSet(train_x, train_y, transform=transform, target_transform=target_transform)
        test_dataset = MolDataSet(test_x, test_y, transform=transform, target_transform=target_transform)
        train_dataloader = DataLoader(train_dataset, args.batch_size)
        test_dataloader = DataLoader(test_dataset, args.test_batch_size)
    if args.normalize_x:
        scalar_file_x_train = os.path.join(root_path, "scalar_x_train.pt")
        with open(scalar_file_x_train, 'wb') as f:
            pickle.dump(train_dataset.transform, f, protocol=pickle.HIGHEST_PROTOCOL)
        if not args.all_value:
            scalar_file_x_test = os.path.join(root_path, "scalar_x_test.pt")
            with open(scalar_file_x_test, 'wb') as f:
                pickle.dump(test_dataset.transform, f, protocol=pickle.HIGHEST_PROTOCOL)
    if args.normalize_y:
        scalar_file_y_train = os.path.join(root_path, "scalar_y_train.pt")
        with open(scalar_file_y_train, 'wb') as f:
            pickle.dump(train_dataset.target_transform, f, protocol=pickle.HIGHEST_PROTOCOL)
        if not args.all_value:
            scalar_file_y_test = os.path.join(root_path, "scalar_y_test.pt")
            with open(scalar_file_y_test, 'wb') as f:
                pickle.dump(test_dataset.target_transform, f, protocol=pickle.HIGHEST_PROTOCOL)

    # defination of the layers
    input_size = args.in_size
    output_size = args.out_size
    str_line = args.hid_size
    if type(str_line) == int:
        hidden_layers_sizes = [str_line]
    else:
        hidden_layers_sizes = [int(str_line) for s in str_line.split(",")]
    pretrain_times = args.pretrain_times
    training_epoches = args.epoches
    learning_rate = args.learning_rate

    numpy_rng = np.random.RandomState(1245)
    dbnmlp = DBNMlp(numpy_rng, theano_rng=None, n_ins=input_size, n_outs=output_size,
                    hidden_layers_sizes=hidden_layers_sizes,normalize_x=args.normalize_x,normalize_y=args.normalize_y)
    prediction = dbnmlp.last_layer.output
    # the RBM costs all
    pretrain_func = dbnmlp.pretraining_functions()
    pre_train_loss = []
    for k in range(pretrain_times):
        d = []
        for k, (train, target) in enumerate(train_dataloader):
            c = []
            for j in range(len(dbnmlp.rbm_layers)):
                loss = pretrain_func[j](train)
                c.append(loss)
            d.append(np.mean(c))
        train_dataloader.clear()
        loss = np.mean(d)
        pre_train_loss.append(loss)
    pre_train_loss = np.array(pre_train_loss)
    x_list = np.linspace(0, len(pre_train_loss), len(pre_train_loss))
    plt.plot(x_list, pre_train_loss)
    fig_save_pretrain = os.path.join(root_path,"dbn_pretrain.png")
    plt.xlabel("The dataset training times")
    plt.ylabel("The energy")
    plt.title("The pretraining process for the DBN model")
    plt.savefig(fig_save_pretrain)
    plt.show()
    plt.close()
    # than train the model
    trainDbnMlpFunc = dbnmlp.build_finetune_function(learning_rate)
    dbn_train_loss = []
    train_data_list = []
    if not args.all_value:test_data_list = []
    for k in range(training_epoches):
        d = []
        for k, (train, target) in enumerate(train_dataloader):
            loss = trainDbnMlpFunc(train, target)
            d.append(loss)
        train_dataloader.clear()
        pea_train, jac_train, cos_train = test_dbnmlp(dbnmlp, train_dataset)
        train_data_list.append([pea_train, jac_train, cos_train])
        if not args.all_value:
            pea_test, jac_test, cos_test = test_dbnmlp(dbnmlp, test_dataset)
            test_data_list.append([pea_test, jac_test, cos_test])
        loss = np.mean(d)
        dbn_train_loss.append(loss)
    dbn_train_loss = np.array(dbn_train_loss)
    x_list = np.linspace(0, len(dbn_train_loss), len(dbn_train_loss))
    plt.plot(x_list, dbn_train_loss)
    plt.xlabel("The dataset training times")
    plt.ylabel("The loss")
    plt.title("The training process for the DBN network model")
    fig_save_trained = os.path.join(root_path, "dbn_train_loss.png")
    plt.savefig(fig_save_trained)
    plt.show()
    plt.close()

    # draw the pictures
    x_list = np.linspace(0, len(train_data_list), len(train_data_list))
    train_data_list = np.array(train_data_list)
    if not args.all_value:test_data_list = np.array(test_data_list)
    list_names = ["peason", "Jaccard", "cosine"]
    types_n = len(list_names)
    for k in range(types_n):
        plt.plot(x_list, train_data_list[:, k], label="train")
        if not args.all_value:
            plt.plot(x_list, test_data_list[:, k], label="test")
        fig_file_path = os.path.join(root_path, "dbnmlp_" + list_names[k] + ".png")
        plt.xlabel("The dataset training times")
        plt.ylabel("The " + list_names[k])
        plt.legend(loc="right")
        plt.title("The " + list_names[k] + " score for the training")
        plt.savefig(fig_file_path)
        plt.show()
        plt.close()
    model_save_file = os.path.join(root_path,'DBNMlp.pt')
    with open(model_save_file, 'wb') as f:
        pickle.dump(dbnmlp, f, protocol=pickle.HIGHEST_PROTOCOL)
def predict_dbnmlp(args):
    predict_file = os.path.join(args.processed_path, "filter_processed_new.xlsx")
    root_path = os.path.join(args.log_path, "second")
    dbnmlp_model_file = os.path.join(root_path, "DBNMlp.pt")
    test_data = pd.read_excel(predict_file,sheet_name="test")
    x_value = test_data.iloc[:,3:]
    with open(dbnmlp_model_file, 'rb') as f:
        dbnmlp = pickle.load(f)
    if dbnmlp.normalize_x:
        scalar_file_x = os.path.join(root_path, "scalar_x_train.pt")
        with open(scalar_file_x, 'rb') as f:
            scalar_x = pickle.load(f)
    if dbnmlp.normalize_y:
        scalar_file_y = os.path.join(root_path, "scalar_y_train.pt")
        with open(scalar_file_y, 'rb') as f:
            scalar_y = pickle.load(f)
        x_value = scalar_x.transform(x_value)

    prediction = dbnmlp.last_layer.output
    predict = theano.function(inputs=[dbnmlp.x], outputs=prediction)
    predict_pIC50 = predict(x_value)
    if dbnmlp.normalize_y:
        predict_pIC50 = scalar_y.inverse_transform(predict_pIC50)
    predict_IC50nm = np.power(10,9-predict_pIC50)
    names = ["IC50_nM","pIC50"]
    test_data.drop(names[0], axis=1)
    test_data[names[0]] = predict_IC50nm
    test_data.drop(names[1], axis=1)
    test_data[names[1]] = predict_pIC50
    save_result_file = os.path.join(root_path,"dbn_ER_alpha_results.xlsx")
    test_data.iloc[:,:3].to_excel(save_result_file,index=None)
