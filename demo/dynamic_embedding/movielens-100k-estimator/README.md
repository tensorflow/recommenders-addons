# A distributed training demo for `tfra.dynamic_embedding`:

- dataset: [movielen/100k-ratings](https://www.tensorflow.org/datasets/catalog/movielens#movielens100k-ratings)
- model: DNN
- Running API: using estimator APIs

## start train:
By default, this shell will start a train task with 1 PS and 1 workers and 1 chief on local machine.
```
sh train.sh
```

## start export for serving:
By default, this shell will start a export for serving task with 1 PS and 1 workers and 1 chief on local machine.
```
sh export.sh
```

## stop.train
```
run sh stop.sh
```

## Compress exported model's embedding table values from float32 to float16

For example, exported model is located in `export_dir/1657094918`

Test exported fp32 model:
```
python movielens-100k-estimator.py --mode test --export_dir export_dir/1657094918
```

Compress exported model:
```
python compress_model.py --model_dir export_dir/1657094918
```

Test compressed fp16 model:
```
python movielens-100k-estimator.py --mode test --export_dir compressed_model
```
