# FAT-RE


## Background

This is a Transformer-based model with Filtering and Aggregation mechanisms for the task of Relation Extraction. Previous methods adopt tree pruning to keep free from the noisy words, which may hurt the semantic integrity. FAT-RE treats a sentence as a fully-connected graph, and let the model decide  which connections are important. 

<img src="https://github.com/LifangD/FAT-RE/blob/master/imgs/tree%20pruning.png" width="60%">

## Architecture 
<img src="https://github.com/LifangD/FAT-RE/blob/master/imgs/arc.png" width="60%" >


## Prepare
   1. Dataset
      - TACRED
      - SemEval2010 Task8  
   2. Word Embedding
      - Glove
      - Word2vec
      - BERT Embedding
   3. Prepare vocab  
      - scripts/prepare_vocab.sh
       



## Train & Test
Please check if the resourses are prepared and the paths/arguments are specified. Example is shown in scripts/train.sh 
```
python main.py --xxx 
```

```
python eval.py --xxx
```


## Result 
<img src="https://github.com/LifangD/FAT-RE/blob/master/imgs/result.png" width="60%">


## Reference
- https://github.com/qipeng/gcn-over-pruned-trees
- Yuhao Zhang, Peng Qi, Christopher D. Manning:
Graph Convolution over Pruned Dependency Trees Improves Relation Extraction. EMNLP 2018: 2205-2215
