# AutoGNR
AutoGNR (Automated GNN with Non-Recursive massage passing) is an automated method for heterogeneous information networks.  
All the command sections start on the current folder. 

## Environment

The experiment is running on GPU. So you may need to download these from Nvidia:

    cuda ~= 11.3 
    cudnn ~= 8.2.0

It is recommended to use Linux. 
The code is working on python. The python environment is as follows:

    python ~= 3.8.13
    torch ~= 1.11.0  # You need to download the correct version following your Cuda version.
    numpy ~= 1.21.5
    scikit-learn ~= 1.0.2
    scipy ~= 1.7.3

It is suggested that you can create a conda environment: autognr. Our sh files for Linux always contain:
```angular2html
conda activate autognr
conda deactivate
```
So, if you are running these files, make sure that you use the correct environment. If you are in the current 
python environment, you can just delete them. If you use another name, you can modify it.
## Data

### Normal-Scale Data

You can download the processed data from 
[here](https://drive.google.com/file/d/1qOZ3QjqWMIIvWjzrIdRe3EA4iKzPi6S5/view?usp=sharing),
which contains three datasets: ACM, DBLP and IMDB.
It is provided from this [repository](https://github.com/seongjunyun/Graph_Transformer_Networks).
If you use the data, please cite their work.

    Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, Hyunwoo J. Kim, Graph Transformer Networks, In Advances in Neural Information Processing Systems (NeurIPS 2019).

After downloading the data, unzip it and copy three folders (ACM/DBLP/IMDB) to `./data`, which also has these three folders.
For each dataset, it consists of edges.pkl, labels.pkl, and node_features.pkl. And for our repository,
we provide each dataset's node types file: node_types.npy. You can also generate it from edges.pkl by yourselves.

If you want to process data on your own, the original dataset is generated by HAN [repository](https://github.com/Jhy1993/HAN).
They offer their cite style

    @article{han2019,
    title={Heterogeneous Graph Attention Network},
    author={Xiao, Wang and Houye, Ji and Chuan, Shi and  Bai, Wang and Peng, Cui and P. , Yu and Yanfang, Ye},
    journal={WWW},
    year={2019}
    }

### Large-Scale Data

You can download the data from [HNE](https://github.com/yangji9181/HNE).
The DBLP and PubMed datasets are provided by them, which are named as DBLP2 and PubMed in our paper.

You can cite their work if you use the data:

    @article{yang2020heterogeneous,
    title={Heterogeneous Network Representation Learning: A Unified Framework with Survey and Benchmark},
    author={Yang, Carl and Xiao, Yuxin and Zhang, Yu and Sun, Yizhou and Han, Jiawei},
    journal={TKDE},
    year={2020}
    }

However, it uses a different format. You need to convert it to the format of the normal-scale data. Besides that,
you need to do random walk. 

We provide a reference code to process in `./additional/process_new_data.py`. It includes both DBLP2 and PubMed.
And we provide the updated `train_heter_search.py` and `train_heter_model.py` to run for random walk in the
same folder.

## Split Dataset

The processed data `labels.pkl` also offer the split of training, validation and test. But it is fixed. We provide 
our split on five-fold cross validation which is shown on each dataset as `labels_5_fold_cross_validation_{which fold for test}.pkl`.
 It is applied to the 100% training set. In addition, we also provide other percentage: 10% (0.1), 25% (0.25), 50% (0.5).
The path is `./data/{dataset}/{percentage: 0.1, 0.25 and 0.5}/{percentage}_labels_5_fold_cross_validation_{fold}.pkl`. 

The split is randomly chosen by controlling a random seed. We provide our code for the split: `./split/try_split.py` and
`./split/percentage.py`. Here we introduce how to use if you want to use it in further study.
If you are in a Linux environment, you 
can generate split directly:
```
cd split
bash -i split.sh
python percentage.py
```
If you are not Linux, here is the replacement method:
```
cd split
python try_split.py --label_path ../data/IMDB/labels.pkl --output_label_path_style ../data/IMDB/labels_{k}_fold_cross_validation_{number}.pkl
python try_split.py --label_path ../data/ACM/labels.pkl --output_label_path_style ../data/ACM/labels_{k}_fold_cross_validation_{number}.pkl
python try_split.py --label_path ../data/DBLP/labels.pkl --output_label_path_style ../data/DBLP/labels_{k}_fold_cross_validation_{number}.pkl
python percentage.py
```

We provide some arguments for `try_split.py` for building your split. Notice that it may change the format
 of the file names, so other codes may need to be modified for using the new format.  

    --seed SEED               random seed, default 0
    --label_path LABEL_PATH   origin label path 
    --output_label_path_style OUTPUT_LABEL_PATH_STYLE
                              it is output label path style, you need to give {k}, {number} in the style
    --k K                     k fold cross validation

As for `percentage.py`, it is running to directly process all three datasets on the default setting. Another mode for it is 
that you can build your ones which target on single labels file.

    --default DEFAULT     if True, it will process all the default datasets with 5 fold and default path, you do not need to provide other args
    --seed SEED           random seed
    --label_path LABEL_PATH
                          label path
    --output_label_path_style OUTPUT_LABEL_PATH_STYLE
                          it is output label path style, you need to give {percentage}, {ori_name} in the style
    --percentage PERCENTAGE
                          the percentage for training set

## Train

You can simply train on all platforms as follows:

    cd AutoGNRModel
    python run.py --result ./ --device 0 
    # the default results folder is ./ (results folder should contain / on the right) and the default GPU number is 0 

There is an alternative method on Linux, like:

    cd AutoGNRModel
    bash -i run.sh ./ 0  

It will record different percentage results on `{results folder}/{percentage: 0.1, 0.25, 0.5, 1}/results.csv`.  
AutoGNR has two parts, namely search and retrain part, which runs on `./AutoGNRModel/train_heter_search.py`
and `./AutoGNRModel/train_heter_model.py`. Here is an example of an independent one run:

    cd AutoGNRModel
    python train_heter_search.py --dataset ACM --labels ../data/ACM/labels_5_fold_cross_validation_0.pkl --seed 0
    python train_heter_model.py --dataset ACM --labels ../data/ACM/labels_5_fold_cross_validation_0.pkl --seed 0 --result ./1/



## Statistics 

There are so many results collected. We write a simple tool to calculate the average results and deviation. A direct 
use example:

    cd AutoGNRModel/{percentage}
    python ../../csvprocess/result.py --csv results.csv --output statistics.csv

More details for this tool:

    python ./csvprocess/result.py --help

## Reference

We are based on the code from [DARTS](https://github.com/quark0/darts) and [DiffMG](https://github.com/AutoML-Research/DiffMG).
