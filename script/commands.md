## BLCA

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset tcga_blca \
--data_root_dir  /home/wzhang/data/tcga_blca/ \
--model blca_surmoe \
--gene_dir ./csv/blca_signatures.csv \
--num_pathway 64
```



## BRCA

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset tcga_brca \
--data_root_dir  /home/wzhang/data/tcga_brca/  \
--model brca_surmoe \ 
--gene_dir ./csv/brca_signatures.csv \
--num_pathway 284 
```



## UCEC

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset tcga_ucec \
--data_root_dir  /home/wzhang/data/tcga_ucec/ \
--model ucec_surmoe \
--gene_dir ./csv/ucec_signatures.csv \
--num_pathway 188
```



## LUAD

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset tcga_luad  \
--data_root_dir  /home/wzhang/data/tcga_luad/ \
--model luad_surmoe \
--gene_dir ./csv/luad_signatures.csv \
--num_pathway 284 
```



## GBMLGG

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset tcga_gbmlgg \
--data_root_dir  /home/wzhang/data/tcga_gbmlgg/ \
--model gbmlgg_surmoe \
--gene_dir ./csv/gbmlgg_signatures.csv \
--num_pathway 282
```





