# ProjectX

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
python vis.py -v baseline -e 10 -n 1000
```
This will use the files saved/baseline/z_proj_11.npy, saved/baseline/z_ans_11 and saved/baseline/z_fused_11 visualise the first 1000 examples from each.

Help - 
```
python3 vis.py -h
```

## Visualisations

- Baseline + gru + fusion (11th epoch)

![visualisation](images/11f.png)

- Baseline + gru without fusion (11th epoch) 

![visualisation](images/11wof.png)

## Asking a new question while testing

Make --USE_NEW_QUESTION='True'

Give the question as string to --NEW_QUESTION='<Question>'

Give the image id on which the question to be asked --IMAGE_ID=<int>

Provide the model and its checkpoint version and checkpoint epoch

Sample command -> 

```
python run.py --MODEL='baseline_wa' --DATASET='vqa' --RUN='test' --GPU='0' --VERSION='testing with one question' --CKPT_V='baseline_wa_gru' --CKPT_E=13 --USE_NEW_QUESTION='True' --NEW_QUESTION='What are you doing' --IMAGE_ID=1
```
