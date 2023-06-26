import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import jittor as jt
import logging
import time
from utils.utils import set_random_seed
from runner.train import trainrenderer, fine_tune, test2
from runner.test import test

CLASSES = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='test2', help='train or test')
    parser.add_argument('--obj_class', default='lego')
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--work_dir', default='./out', help='the dir to save logs and models')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists(os.path.abspath(os.path.join(args.work_dir, args.obj_class))):
        os.mkdir(os.path.abspath(os.path.join(args.work_dir, args.obj_class)))

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
        trainrenderer(logger, args)
    elif args.stage == 'test':
        test(logger, args)
    elif args.stage == 'fine_tune':
        fine_tune(logger, args)
    else:
        test2(logger, args)




if __name__ == '__main__':
    jt.flags.use_cuda = 1
    set_random_seed(0)
    main()