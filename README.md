# deep-parity
TreeNN for recognizing parity of arithmetic expressions (like 1 + 3 - 2 + 1)

# Genarating data
Run this:
```
mkdir data
python3 utils/generate.py \
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

# Training models
We have two models available: TreeNN and bidirectional RNN.

## Training TreeNN
```
python3 models_definitions/tree_nn.py \
	--train_set data/split/train \
	--valid_set data/split/valid \
	--epoch 500
```
## Training RNN
```
python3 models_definitions/r-nn.py \
	--train_set data/split/train \
	--valid_set data/split/valid \
	--vocab data/split/vocab \
	--batch_size 128 \
	--epochs 64 \
	--embed_dim 8 \
	--rnn_cell_dim 16 \
	--num_dense_layers 2 \
	--dense_layer_units 32
```

### Quering trained RNN model.
```
python3 utils/predict_r_nn.py \
	--model models_pretrained/rnn/first_try/model \
	--pairs models_pretrained/rnn/first_try/data/split/test_no_labels \
	--vocab models_pretrained/rnn/first_try/data/split/vocab
```


# Plotting logs
```
Rscript utils/plot.R logs/some.log plots/
```
