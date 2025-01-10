import argparse
import pickle
import random

from sklearn.model_selection import cross_val_score


import numpy as np

# Arguments for setting
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--label_path", type=str, default="../data/IMDB/labels.pkl", help="label path")
parser.add_argument("--output_label_path_style", type=str,
                    default="../data/IMDB/labels_{k}_fold_cross_validation_{number}.pkl",
                    help="it is output label path style, you need to give {k}, {number} in the style")
parser.add_argument("--k", type=int, default=5, help="k fold cross validation")
args = parser.parse_args()
random.seed(args.seed)


# load labels file
def load(labels):
    with open(labels, "rb") as f:
        label_original = pickle.load(f)
        f.close()
    return label_original


# drop the former split, concentrate all the data and labels
def cat(labels):
    train_idx = np.array(labels[0])[:, 0]
    train_target = np.array(labels[0])[:, 1]
    valid_idx = np.array(labels[1])[:, 0]
    valid_target = np.array(labels[1])[:, 1]
    test_idx = np.array(labels[2])[:, 0]
    test_target = np.array(labels[2])[:, 1]

    idx = np.concatenate([train_idx, valid_idx, test_idx])
    target = np.concatenate([train_target, valid_target, test_target])

    return idx, target


# calculate the length of training, validation and test set.
def k_fold_length(length, k):
    test_len = int(length / k)
    val_len = int((length - test_len) / 3)
    train_len = length - test_len - val_len
    return train_len, val_len, test_len


# split to k folds, and generate the split index based on which fold for test
def k_fold_index(train_len, val_len, test_len, length, k):
    assert train_len + val_len + test_len == length
    index = list(range(length))
    result = []
    for i in range(k):
        test_index = index[i * test_len: (i + 1) * test_len]
        train_val_index = index[:i * test_len] + index[(i + 1) * test_len:]
        train_index = train_val_index[:train_len]
        val_index = train_val_index[train_len:]
        result.append([train_index, val_index, test_index])
    return result


# random shuffle the index for dataset index and label, to reorder the dataset index and label
def reorder(idx, target, length):
    index = list(range(length))
    random.shuffle(index)
    idx_new = idx[index]
    target_new = target[index]
    return idx_new, target_new


# split following the lengths provided
def split(idx_new, target_new, lengths):
    train_idx = idx_new[lengths[0]]
    train_target = target_new[lengths[0]]
    valid_idx = idx_new[lengths[1]]
    valid_target = target_new[lengths[1]]
    test_idx = idx_new[lengths[2]]
    test_target = target_new[lengths[2]]

    train_labels = np.vstack((train_idx, train_target)).transpose().tolist()
    valid_labels = np.vstack((valid_idx, valid_target)).transpose().tolist()
    test_labels = np.vstack((test_idx, test_target)).transpose().tolist()

    return [train_labels, valid_labels, test_labels]


# save the split for the number(th) of k folds
def save(result, style, number, k):
    with open(style.format(**{"k": k, 'number': number}), "wb") as f:
        pickle.dump(result, f)
        f.close()


# the whole process of k fold cross validation split
label_ori = load(args.label_path)   # load original file
label_idx, label_target = cat(label_ori)   # change the file content to two objects: dataset index and label
label_train_len, label_val_len, label_test_len = k_fold_length(len(label_idx), args.k)  # calculate the lengths of
# train, val and test
k_fold_result = k_fold_index(label_train_len, label_val_len, label_test_len, len(label_idx), args.k)  # generate the
# split index
label_idx_new, label_target_new = reorder(label_idx, label_target, len(label_idx))  # random reorder dataset index and label
for step, single in enumerate(k_fold_result):
    split_result = split(label_idx_new, label_target_new, single)  # use each fold split index to get split result
    # from new dataset index and label
    save(split_result, args.output_label_path_style, step, args.k)  # save the result
