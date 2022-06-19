# Fugue: A simple batch-correction method that is scalable for integrating super large-scale single-cell transcriptomes



## Introduction of Fugue

Batch effects are fundamental issues to be addressed for integration of single-cell transcriptomes. Here, we present **Fugue**, a simple yet efficient solution for batch-correction of super large-scale single-cell transcriptomes. 

Fugue is a musical genre in which the principle theme is repeated or imitated in different time and music scale. In concept, the gene expression profile could be seen as superposition of the biological information and batch information, which is likely this musical genre. 

Fugue algorithm is based on a self-supervised learning-based framework. We encode batch information as trainable parameters and add them into expression profiles. A contrastive learning method is applied to narrow the gap between single-cell and its various augmentation one. The batch information can be properly represented after training. By taking batch information as trainable variable, Fugue is scalable in atlasing-scale data integration with fixed memory usage.

 

## Architecture of Fugue

![image](https://github.com/xilinshen/Fugue/blob/master/images/flow_chart.png)

Given a set of uncorrected single-cells (A), Fugue embedded their batch information as a learnable matrix and added them to the expression profile for feature encoder training (B) . The feature encoder was trained with contrastive loss  (C). At the feature extraction stage, single-cell expression profiles were provided to the feature encoder to extract embedding representation  (D) . The embedding representation could be utilized for downstream analysis such as visualization and cell clustering  (E).



## Installation

The Fugue package is available through the following codes:

```
git clone https://github.com/xilinshen/Fugue.git
```

The following packages are required:

```
python==3.8.8 
numpy==1.21.4
torch==1.8.0
torchvision==0.9.0
scanpy==1.7.0rc1
```

Some source codes were copied from [facebookresearch/MoCo](https://github.com/facebookresearch/moco).



## Training Fugue on own data

### Data preprocessing

We recommend the user to normalize the single cell count matrix as counts per million normalization (CPM) and took logarithmic transformation (i.e. log2(CPM+1)). 

```
import numpy as np
from utils.preprocessing import *

count_matrix = np.load("./data/splatter_simulation_count.npz")["x"] # expression profiles
batch = np.load("./data/splatter_simulation_count.npz")["y"] # batch labels
celltype = np.load("./data/splatter_simulation_count.npz")["g"] # cell type labels

X = data_preprocessing(count_matrix)

np.savez_compressed("./data/splatter_simulation.npz", x = X, y = batch, g = celltype)
```

The data needs to be preprocessed as numpy compressd `.npz` format.  Key `"x"` should be the profile and `"y"` should be batch labels. 



### Model training

The model was trained on GPUs:

```
python main.py \
--file "./data/splatter_simulation.npz" \
--arch densenet21 \
--batch-size 128 \
--dist-url "tcp://localhost:10000" \
--outdir "./result/" \
--mlp --moco-k 1408 --moco-m 0.999 \
--in_features 2000 \
--num_batches 5 \
--shuffle-ratio 0.1 \
--randomzero-ratio 0.3 \
--multiprocessing-distributed 
```



## Extract batch removing representation of single cells from the pretrained feature encoder

```python
import utils
import numpy as np
import scanpy as sc

# load data
X = np.load("./data/splatter_simulation.npz")["x"] # expression profile
batch = np.load("./data/splatter_simulation.npz")["y"] # batch label
celltype = np.load("./data/splatter_simulation.npz")["g"] # cell type label

in_features = 2000
assert in_features == X.shape[1]

# load the pretrained feature encoder
arch = "densenet21"
checkpoint = "./result/checkpoint_0029.pth.tar"
model = utils.load_pretrained_model(arch, in_features, checkpoint, return_feature=True)

# extract embeddings of single-cells
features = utils.extract_features(model, X)

# visualization
adata=sc.AnnData(features)
adata.obs["celltype"] = celltype
adata.obs["batch"] = batch
adata.obs["celltype"] = adata.obs["celltype"].astype("category")
adata.obs["batch"] = adata.obs["batch"].astype("category")

sc.pp.neighbors(adata)
sc.tl.leiden(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["celltype","batch"])
```



