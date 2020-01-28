# OpenVQA

<div>
	<a href="https://openvqa.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/openvqa/badge/?version=latest"/></a>
</div>

## Visualisation:

Input -
1. Two npy files, each containing a numpy vector X (num_samples, feature_vector_dimension).

Output -
1. vis.png file

![vis.png](vis.png)

Example use - 
```
python3 vis.py -f1 z1.npy -f2 z2.npy -n 100
```
This will use the files z1.npy and z2.npy and visualise the first 100 examples from each.

Help - 
```
python3 vis.py -h
```






