import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset import ASVspoof2019
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np

def get_eer_tdcf(feat_model_path):
    
    dirname = os.path.dirname
    dir_path = dirname(feat_model_path)
    print(dir_path)
    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score_eval_all.txt'),"/home/sarfaraz/ASVSpoof_2019_expts/data/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./models/ocsoftmax")
    
    args = parser.parse_args()
    
    get_eer_tdcf(args.model_dir)
