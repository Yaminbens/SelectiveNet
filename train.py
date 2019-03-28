import argparse
import datetime
from models.cifar10_vgg_selectivenet import cifar10vgg as cifar10Selective
from selectivnet_utils import *

MODELS = {"cifar_10": cifar10Selective}



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar_10')

parser.add_argument('--model_name', type=str, default='coverage_loss')
parser.add_argument('--baseline', type=str, default='none')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--llambda', type=float, default=1/2)
parser.add_argument('--risk', type=float, default=0.048)

args = parser.parse_args()

model_cls = MODELS[args.dataset]
model_name = args.model_name +'_'+ datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
baseline_name = args.baseline
lamda = args.llambda
risk = args.risk
risks = np.flip(np.arange(risk, step=risk/10))[:6]


if baseline_name == "none":
        for risk in risks:
                results = train_by_risk(model_name, cifar10Selective, lamda, risk, regression=False, alpha=args.alpha)
# else:
#     model_baseline = model_cls(train=to_train("{}.h5".format(baseline_name)),
#                                filename="{}.h5".format(baseline_name),
#                                baseline=True)
#     results = train_profile(model_name, model_cls, coverages, model_baseline=model_baseline, alpha=args.alpha)
