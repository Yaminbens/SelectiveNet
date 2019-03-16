import argparse

from models.cifar10_vgg_selectivenet import cifar10vgg as cifar10Selective
from selectivnet_utils import *

MODELS = {"cifar_10": cifar10Selective}



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar_10')

parser.add_argument('--model_name', type=str, default='test')
parser.add_argument('--baseline', type=str, default='none')
parser.add_argument('--alpha', type=float, default=0.5)

args = parser.parse_args()

model_cls = MODELS[args.dataset]
model_name = args.model_name
baseline_name = args.baseline

coverages = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]


if baseline_name == "none":
    results = train_full_coverage(model_name, cifar10Selective, coverages, alpha=args.alpha)

