#! /bin/bash
python search/search.py --algo NSG32,Flat --size 100M -k 10 --threads 64 --ep kmeans --n-ep 8 --alpha 0.85 --search-l 16 --outdir result
