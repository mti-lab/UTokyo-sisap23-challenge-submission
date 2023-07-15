#! /bin/bash
python search/search.py --algo PCA600,NSG32,Flat --size 300K -k 10 --threads 64 --ep kmeans --n-ep 58 --alpha 0.93619 --search-l 26 --outdir result
