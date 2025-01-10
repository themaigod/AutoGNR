#!/bin/bash
conda activate autognr
if [ -n $1 ] ; then
    resultL=$1
else
    resultL="./"
fi
if [ -n $2 ] ; then
    gpuN=$2
else
    gpuN="0"
fi
echo $resultL
echo $gpuN
for p in 0.1 0.25 0.5
do
    mkdir -p "$resultL$p/"
    for n in 0.01
    do
        for m in 64
        do
            for k in 3
            do
                for i in {0..4}
                do
                    for j in {0..9}
                    do
                        python train_heter_search.py --device $gpuN --seed $j --labels ../data/ACM/$p/${p}_labels_5_fold_cross_validation_$i.pkl --dataset ACM --n_hid $m --num_hops $k --lr $n
                        python train_heter_model.py --device $gpuN --seed $j --labels ../data/ACM/$p/${p}_labels_5_fold_cross_validation_$i.pkl --result "$resultL$p/" --dataset ACM --n_hid $m --num_hops $k --lr $n
                    done
                done
                for i in {0..4}
                do
                    for j in {0..9}
                    do
                        python train_heter_search.py --device $gpuN --seed $j --labels ../data/DBLP/$p/${p}_labels_5_fold_cross_validation_$i.pkl --dataset DBLP --n_hid $m --num_hops $k --lr $n
                        python train_heter_model.py --device $gpuN --seed $j --labels ../data/DBLP/$p/${p}_labels_5_fold_cross_validation_$i.pkl --result "$resultL$p/" --dataset DBLP --n_hid $m --num_hops $k --lr $n
                    done
                done        
                for i in {0..4}
                do
                    for j in {0..9}
                    do
                        python train_heter_search.py --device $gpuN --seed $j --labels ../data/IMDB/$p/${p}_labels_5_fold_cross_validation_$i.pkl --dataset IMDB --n_hid $m --num_hops $k --lr $n
                        python train_heter_model.py --device $gpuN --seed $j --labels ../data/IMDB/$p/${p}_labels_5_fold_cross_validation_$i.pkl --result "$resultL$p/" --dataset IMDB --n_hid $m --num_hops $k --lr $n
                    done
                done
            done
        done
    done    
done
for n in 0.01
do
    mkdir -p "${resultL}1/"
    for m in 64
    do
        for k in 3
        do
            for i in {0..4}
            do
                for j in {0..9}
                do
                    python train_heter_search.py --device $gpuN --seed $j --labels ../data/ACM/labels_5_fold_cross_validation_$i.pkl --dataset ACM --n_hid $m --num_hops $k --lr $n
                    python train_heter_model.py --device $gpuN --seed $j --labels ../data/ACM/labels_5_fold_cross_validation_$i.pkl --result ${resultL}1/ --dataset ACM --n_hid $m --num_hops $k --lr $n
                done
            done
            for i in {0..4}
            do
                for j in {0..9}
                do
                    python train_heter_search.py --device $gpuN --seed $j --labels ../data/DBLP/labels_5_fold_cross_validation_$i.pkl --dataset DBLP --n_hid $m --num_hops $k --lr $n
                    python train_heter_model.py --device $gpuN --seed $j --labels ../data/DBLP/labels_5_fold_cross_validation_$i.pkl --result ${resultL}1/ --dataset DBLP --n_hid $m --num_hops $k --lr $n
                done
            done        
            for i in {0..4}
            do
                for j in {0..9}
                do
                    python train_heter_search.py --device $gpuN --seed $j --labels ../data/IMDB/labels_5_fold_cross_validation_$i.pkl --dataset IMDB --n_hid $m --num_hops $k --lr $n
                    python train_heter_model.py --device $gpuN --seed $j --labels ../data/IMDB/labels_5_fold_cross_validation_$i.pkl --result ${resultL}1/ --dataset IMDB --n_hid $m --num_hops $k --lr $n
                done
            done
        done
    done
done 
conda deactivate