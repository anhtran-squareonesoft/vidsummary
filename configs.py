# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint

project_dir = Path(__file__).resolve().parent
# dataset_dir = Path('/data1/jysung710/tmp_sum/360video/').resolve()
dataset_dir = Path('/school/Adversarial_Video_Summary-master/data/tvsum50/ydata-tvsum50-v1_1/ydata-tvsum50-video/video').resolve()

# video_list = ['360airballoon', '360parade', '360rowing', '360scuba', '360wedding']
video_list = ['-esJrBWj2d8',
'0tmA_C6XwfM',
'37rzWOQsNIw',
'3eYKfiOEJNs',
'4wU_LUjG5Ic',
'91IHQYk1IQM',
'98MoyGZKHXc',
'akI8YFjEmUw',
'AwmHb44_ouw',
'b626MiF1ew4',
'Bhxk-O1Y7Ho',
'byxOvuiIJV0',
'cjibtmSLxQ4',
'E11zDS9XGzg',
'EE-bNr36nyA',
'eQu1rNs0an0',
'EYqVtI9YWJA',
'fWutDQy1nnY',
'GsAD1KT1xo8',
'gzDbaEs1Rlg',
'Hl-__g2gn_A',
'HT5vyqe0Xaw',
'i3wAGJaaktw',
'iVt07TCkFM0',
'J0nA4VgnoCo',
'jcoYJXDG9sw',
'JgHubY5Vw3Y',
'JKpqYvAdIsw',
'kLxoNp-UchI',
'LRw_obCPUt0',
'NyBmCxDoHJU',
'oDXZc0tZe04',
'PJrm840pAUI',
'qqR6AEXwxoQ',
'RBCABdttQmI',
'Se3oxnaPsz0',
'sTEELN-vY30',
'uGu_10sucQo',
'vdmoEJ5YbrQ',
'VuWGsYPqAX8',
'WG0MBPpPC6I',
'WxtbjNsCQ8A',
'XkqCExn6_Us',
'xmEERLqJ2kU',
'xwqBXPGE9pQ',
'xxdtq8mxegs',
'XzYM3PfTM4w',
'Yi4Ij2NM7U4',
'z_6gVvQb2d0',
'_xMr-HKMfVA']
save_dir = Path('/saved/SUM_GAN/')
score_dir = Path('/data1/common_datasets/tmp_sum/360video/results/SUM-GAN/')


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='360airballon'):
        if self.preprocessed:
            # self.video_root_dir = dataset_dir.joinpath('resnet101_feature', video_type, self.mode)
            # self.video_root_dir = dataset_dir.joinpath(self.mode)
            self.video_root_dir = "dataset_tvsum"
        else:
            # self.video_root_dir = dataset_dir.joinpath('video_subshot', video_type, 'test')
            self.video_root_dir = "dataset_tvsum"
        self.save_dir = save_dir.joinpath(video_type)
        self.log_dir = self.save_dir
        self.ckpt_path = self.save_dir.joinpath(f'epoch-{self.epoch}.pkl')
        self.score_dir = score_dir

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--verbose', type=str2bool, default='true')
    parser.add_argument('--preprocessed', type=str2bool, default='True')
    parser.add_argument('--video_type', type=str, default='-esJrBWj2d8')

    # Model
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--summary_rate', type=float, default=0.3)

    # Train
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_lr', type=float, default=1e-5)
    parser.add_argument('--discriminator_slow_start', type=int, default=15)

    # load epoch
    parser.add_argument('--epoch', type=int, default=2)
    
    if parse:
        # kwargs = parser.parse_args()
        kwargs, unknown = parser.parse_known_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    import ipdb
    ipdb.set_trace()
