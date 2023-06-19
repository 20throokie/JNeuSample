import argparse
import os
import jittor as jt
import logging
import time
from utils.utils import set_random_seed
from runner.train import traindetector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', help='train or test')
    parser.add_argument('--work_dir', default='./out', help='the dir to save logs and models')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists(os.path.abspath(args.work_dir)):
        os.mkdir(os.path.abspath(args.work_dir))

    # create log file
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.work_dir, f'{timestamp}.log')
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 时间生成
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)


    if args.stage == 'train':
        traindetector(logger, args)
    else:
        pass
        #test()






if __name__ == '__main__':
    jt.flags.use_cuda = 1
    set_random_seed(66)
    main()