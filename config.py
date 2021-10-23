import argparse
import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path",type=str,default="./rawdata",help="preprocess raw data path.")
    parser.add_argument("--log-path",type=str,default="./log",help="data log path.")
    parser.add_argument("--processed-path",type=str,default="./data",help="data processed path.")
    parser.add_argument("--ord",type=int,default=2,help="the distance of the vector calculating.")
    parser.add_argument("--batch-size",type=int,default=400,help="the batch size of training process.")
    parser.add_argument("--test-batch-size", type=int, default=100, help="the batch size of test training process.")
    parser.add_argument("--in-size",type=int,default=8,help="the size of model input.")
    parser.add_argument("--hid-size",default=40,help="the size of model hidden.")
    parser.add_argument("--out-size",type=int,default=1,help="the size of model output.")
    parser.add_argument("--epoches", type=int, default=5, help="the epoches of model.")
    parser.add_argument("--learning_rate", type=float, default=0.15, help="the learning rate of model.")
    parser.add_argument("--lambd", type=float, default=0.0, help="the lambda of model.")
    parser.add_argument("--type", type=str, default="bpnet", help="the model type including bpnet,dbn,VBnet.")
    parser.add_argument("--pretrain-times", type=int, default=7, help="the model dbn pretrain times.")
    parser.add_argument("--normalize-x",action='store_false', help="the model dbn pretrain times.")
    parser.add_argument("--normalize-y", action='store_false', help="the model dbn pretrain times.")
    parser.add_argument("--all-value", action='store_true', help="the model dbn pretrain times.")
    args = parser.parse_args()
    return args
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
