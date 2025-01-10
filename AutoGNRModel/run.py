import time
import os
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--result', type=str, default="./")
parser.add_argument('--device', type=int, default=5)
args = parser.parse_args()

# run 100% training set
result_path = args.result + "1/"
if not os.path.isdir(result_path):
    os.makedirs(result_path)
for i in range(5):
    for j in range(10):
        os.system(
            "python train_heter_search.py --device {} --seed {} --labels ../data/ACM/labels_5_fold_cross_validation_{}.pkl --dataset ACM --n_hid 64 --num_hops 3 --lr 0.01".format(
                args.device, j, i))
        os.system(
            "python train_heter_model.py --device {} --seed {} --labels ../data/ACM/labels_5_fold_cross_validation_{}.pkl --result {} --dataset ACM --n_hid 64 --num_hops 3 --lr 0.01".format(
                args.device, j, i, result_path))

for i in range(5):
    for j in range(10):
        os.system(
            "python train_heter_search.py --device {} --seed {} --labels ../data/DBLP/labels_5_fold_cross_validation_{}.pkl --dataset DBLP --n_hid 64 --num_hops 3 --lr 0.01".format(
                args.device, j, i))
        os.system(
            "python train_heter_model.py --device {} --seed {} --labels ../data/DBLP/labels_5_fold_cross_validation_{}.pkl --result {} --dataset DBLP --n_hid 64 --num_hops 3 --lr 0.01".format(
                args.device, j, i, result_path))

for i in range(5):
    for j in range(10):
        os.system(
            "python train_heter_search.py --device {} --seed {} --labels ../data/IMDB/labels_5_fold_cross_validation_{}.pkl --dataset IMDB --n_hid 64 --num_hops 3 --lr 0.01".format(
                args.device, j, i))
        os.system(
            "python train_heter_model.py --device {} --seed {} --labels ../data/IMDB/labels_5_fold_cross_validation_{}.pkl --result {} --dataset IMDB --n_hid 64 --num_hops 3 --lr 0.01".format(
                args.device, j, i, result_path))

# run 10%, 25% and 50% of training set
for k in [0.1, 0.25, 0.5]:
    result_path = args.result + "{}/".format(k)
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    for i in range(5):
        for j in range(10):
            os.system(
                "python train_heter_search.py --device {} --seed {} --labels ../data/ACM/{}/{}_labels_5_fold_cross_validation_{}.pkl --dataset ACM --n_hid 64 --num_hops 3 --lr 0.01".format(
                    args.device, j, k, k, i))
            os.system(
                "python train_heter_model.py --device {} --seed {} --labels ../data/ACM/{}/{}_labels_5_fold_cross_validation_{}.pkl --result {} --dataset ACM --n_hid 64 --num_hops 3 --lr 0.01".format(
                    args.device, j, k, k, i, result_path))

    for i in range(5):
        for j in range(10):
            os.system(
                "python train_heter_search.py --device {} --seed {} --labels ../data/DBLP/{}/{}_labels_5_fold_cross_validation_{}.pkl --dataset DBLP --n_hid 64 --num_hops 3 --lr 0.01".format(
                    args.device, j, k, k, i))
            os.system(
                "python train_heter_model.py --device {} --seed {} --labels ../data/DBLP/{}/{}_labels_5_fold_cross_validation_{}.pkl --result {} --dataset DBLP --n_hid 64 --num_hops 3 --lr 0.01".format(
                    args.device, j, k, k, i, result_path))

    for i in range(5):
        for j in range(10):
            os.system(
                "python train_heter_search.py --device {} --seed {} --labels ../data/IMDB/{}/{}_labels_5_fold_cross_validation_{}.pkl --dataset IMDB --n_hid 64 --num_hops 3 --lr 0.01".format(
                    args.device, j, k, k, i))
            os.system(
                "python train_heter_model.py --device {} --seed {} --labels ../data/IMDB/{}/{}_labels_5_fold_cross_validation_{}.pkl --result {} --dataset IMDB --n_hid 64 --num_hops 3 --lr 0.01".format(
                    args.device, j, k, k, i, result_path))
