import os
from config import parse_args,check_path
from src.log import Logger
from src.train import train_bpnet,predict_bpnet,train_dbnmlp,predict_dbnmlp
if __name__ == '__main__':
    args = parse_args()
    check_path(args)
    logger = Logger(args)
    root_path = os.path.join(args.log_path,"second")
    bpnet_model_file = os.path.join(root_path,"bpnet.pt")
    if not os.path.exists(bpnet_model_file):
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        train_bpnet(args)
    else:
        logger.info("model files saved in file:%s" % bpnet_model_file)
    save_result_file = os.path.join(args.log_path, "bpnet_ER_alpha_results.xlsx")
    if not os.path.exists(save_result_file) and args.all_value:
        predict_bpnet(args)
    else:
        logger.info("predict bpnet model files saved in file:%s" % save_result_file)

    dbnmlp_model_file = os.path.join(args.log_path, "DBNMlp.pt")
    if not os.path.exists(dbnmlp_model_file):
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        train_dbnmlp(args)
    else:
        logger.info("model files saved in file:%s" % dbnmlp_model_file)
    save_result_file = os.path.join(args.log_path, "dbn_ER_alpha_results.xlsx")
    if not os.path.exists(save_result_file) and args.all_value:
        predict_dbnmlp(args)
    else:
        logger.info("predict dbn model files saved in file:%s" % save_result_file)
    logger.save_log()
