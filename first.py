import os
from src.gray import gray_model
from src.log import Logger
from config import parse_args,check_path
if __name__ == '__main__':
    args = parse_args()
    check_path(args)
    logger = Logger(args)
    raw_er_activity_file = os.path.join(args.raw_path, "ERα_activity.xlsx")
    raw_molecular_file = os.path.join(args.raw_path, "Molecular_Descriptor.xlsx")
    processed_merge_file = os.path.join(args.processed_path, "combine_processed.xlsx")
    first_path = os.path.join(args.log_path,"first")
    gray_result_file = os.path.join(first_path, "gray_result.xlsx")
    if not os.path.exists(first_path):
        os.mkdir(first_path)
        gray_model(processed_merge_file, gray_result_file)
        logger.info("Save in file %s" % gray_result_file)
    else:
        logger.info("The file is in %s" % gray_result_file)
    logger.save_log()
