import argparse
from dataset import *
from train import *
from layers import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora")
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--weight_decay', type=float, default = 0.005)
parser.add_argument('--early_stopping', type=int, default = 0)
parser.add_argument('--hidden', type=int, default= 16)
parser.add_argument('--dropout', type=float, default = 0.55)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=4)
args = parser.parse_args()
# Cora lr=0.02 w=0.005 d=0.4
# for data in ["Cora", 'CiteSeer', 'PubMed']:
#     args.dataset = data
dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
# dataset = get_coauthor_dataset(args.dataset, args.normalize_features)
# # dataset = get_amazon_dataset(args.dataset, args.normalize_features)
# permute_masks = random_coauthor_amazon_splits

print("Data:", dataset[0])

    # if data=="PubMed":
    #     args.dropout = 0.4
    # elif data=="Cora":
    #     args.dropout = 0.55
    #     args.l=0.025
    #     args.weight_decay=0.005

    # 3-32 w=0.002 d=0.55 32,64,128 w=0.005 d=0.55
    # else
    #     args.dropout = 0.35
    #     dropout = 0.35
    #     args.l=0.02
    #     args.weight_decay=0.008
# #
# for k in range(1 ,32):
#     args.K = k
#     run(dataset, Net(dataset,args), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping,args, permute_masks=None,lcc = False)
for k in range(2,7):
    args.K = pow(2,k)
    run(dataset, Net(dataset,args), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping,args, permute_masks=None,lcc = False)