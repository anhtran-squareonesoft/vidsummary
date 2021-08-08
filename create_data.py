import argparse
from util.generate_dataset import Generate_Dataset

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options

parser.add_argument('--input', '--split', type=str, default='dataset_tvsum', help="input video")
parser.add_argument('--output', type=str, default='dataset_tvsum/data.h5', help="out data")

args, unknown = parser.parse_known_args()
if __name__ == "__main__":
    gen = Generate_Dataset(args.input, args.output)
    gen.generate_dataset()
    gen.h5_file.close()