# Introduction
Item Matching tool

# How to setup
Create your environment with conda
```
conda create -n <your_name> python=3.11 -y && conda activate <your_name>
```

Install with pip:
```
pip install item-matching
```

Matching:
```
path = 'item_matching'
path_db = 'database.csv'
path_q = 'query.csv'
Matching(path, path_db, path_q)
```
