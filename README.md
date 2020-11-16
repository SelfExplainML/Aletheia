# Aletheia 
A Python package for unwrapping ReLU Networks

## Installation 

The following environments are required:

- Python 3.7 or above
- matplotlib>=3.3
- numpy>=1.19.1 
- pandas>=1.1.2
- seaborn>=0.9.0
- scikit-learn>=0.23.0
- csaps==0.11.0

```shell
pip install git+https://github.com/SelfExplainML/aletheia.git
```

## Usage

Load data
```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

random_state = 0

x, y = make_circles(n_samples=2000, noise=0.1, random_state=random_state)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=random_state)

plt.figure(figsize=(10,8))
scatter = plt.scatter(x[:, 0], x[:, 1], c=y)
plt.legend(*scatter.legend_elements(), loc="upper right")
plt.show()
```

Train a ReLU Network
```
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=[40] * 4, max_iter=2000, early_stopping=True, 
                    n_iter_no_change=100, validation_fraction=0.2,
                    solver='adam', activation="relu", random_state=random_state, 
                    learning_rate_init=0.001)
mlp.fit(train_x, train_y)
```

Run UnwrapperClassifier in Aletheia
```python
from aletheia import *
clf = UnwrapperClassifier(mlp.coefs_, mlp.intercepts_)
clf.fit(train_x, train_y)
clf.summary()
```

![CoCircleSummary](https://github.com/SelfExplainML/Aletheia/blob/master/examples/results/CoCircle_SummaryTable.png)

Partitioned regions
```python 
clf.visualize2D_regions(figsize=(8, 8), meshsize=300, show_label=False)
```
![CoCircleRegions](https://github.com/SelfExplainML/Aletheia/blob/master/examples/results/CoCircle_Regions.png)

Simplification
```python 
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

datanum = train_x.shape[0]
indices = np.arange(datanum)
idx1, idx2 = train_test_split(indices, test_size=0.2, random_state=random_state)
val_fold = np.ones((len(indices)))
val_fold[idx1] = -1

grid = GridSearchCV(MergerClassifier(unwrapper=None, 
                                     weights=mlp.coefs_, 
                                     biases=mlp.intercepts_,
                                     min_samples=30,
                                     n_neighbors=np.round(clf.nllms * 0.01).astype(int),
                                     refit_model=LogisticRegression()),
                                     param_grid={"n_clusters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]},
                                     scoring={"auc": make_scorer(roc_auc_score, needs_proba=True)},
                                     cv=PredefinedSplit(val_fold), refit="auc", n_jobs=10, error_score=np.nan)
grid.fit(train_x, train_y)
clf_merge = grid.best_estimator_
clf_merge.summary()
```
<img src="https://github.com/SelfExplainML/Aletheia/blob/master/examples/results/CoCircle_MergeSummaryTable.png" width="480">

Local Inference
```python 
tmpid = 0
clf_merge.visualize2D_one_line(tmpid, figsize=(8, 8))
clf_merge.local_inference_wald(tmpid).round(4)
```
<img src="https://github.com/SelfExplainML/Aletheia/blob/master/examples/results/CoCircle_Local.png" width="480">
<img src="https://github.com/SelfExplainML/Aletheia/blob/master/examples/results/CoCircle_Inference.png" width="480">

## Reference
Agus Sudjianto, William Knauth, Rahul Singh, Zebin Yang and Aijun Zhang. 2020. Unwrapping The Black Box of Deep ReLU Networks: Interpretability, Diagnostics, and Simplification. arXiv preprint arXiv:2011.04041. [link](https://arxiv.org/abs/2011.04041)
