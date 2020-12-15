# Train Aspect-distinctive Word Embedding

- To generate betweenAspect file by randomly selecting words from different aspects
```python
python script.py
```
- To train the aspect-distinctive embedding
```bash
make
time ./word2vec -train ../data/restaurant/train.txt 
                -output wv.txt 
                -cbow 0 
                -min_count 2 
                -size 200 
                -negative 25 
                -hs 1 
                -sample 1e-4 
                -threads 30 
                -binary 0 
                -iter 20 
                -betweenAspectfile betweenAspect.txt 
                -lambda_between 0.6 
                -withinAspectfile withinAspect.txt 
                -lambda_within 0.0
```
