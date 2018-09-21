# deep-parity
TreeNN for recognizing parity of arithmetic expressions (like 1 + 3 - 2 + 1)

# Genarating data
Run this:
```
mkdir data
python3 utils/generate.py
	-n 30000 \
	--modulo 2 \
	--numbers 0,1,2 \
	--lengths 3,4,5,6,7,8,9,10,11 \
		> data/examples
```
where `-n` determines number of examples.
Generated examples should look like this:
```
0 1+1-2-2+0-2+1+1
1 0-0-0-1+0+2-0+0+1+1-0
0 1-1-0-1-0+1+0
1 1+2+2-1-1+1+1
0 0+0+0-2+2-2
1 0+2-1-1+2-1
1 2-0+2+2-1
0 1+2+1-0-1+1-0-0+2
0 0+2-0-2+0-0-2+0+0-2
```
Now we need to split this into training, validation and testing part, removing
possible duplicates along the way. Let's do this by running another script.
```
mkdir data/split
python3 utils/split.py data/examples --dirname data/split \
									 --train 0.5 --valid 0.3 --test 0.2
```

# Training a model
```
python3 models_definitions/tree_nn.py \
	--train_set data/split/train \
	--valid_set data/split/valid \
	--epoch 500
```
