from train import train
import os
import torch
import argparse
from hparams import hparams

here = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    train(hparams)
