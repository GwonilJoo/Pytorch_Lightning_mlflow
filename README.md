# Pytorch_Lightning_mlflow

Pytroch Lightning과 mlflow 연동에 대해 공부한 레포입니다.

## 실행 방법

### train
- default 실행
```
python train.py
```

- argument 지정 실행
```
python train.py \
--seed 21
--batch_size 500 \
--n_epochs 30 \
--lr 1e-3 \
--n_gpus 0 \
--save_dir ./saved/pl/ \
--mode pl
```

### test
- default 실행
```
python test.py
```

- argument 지정 실행
```
python test.py \
--seed 21 \
--batch_size 500 \
--n_gpus 0 \
--save_path ./saved/pl/best_val_loss.ckpt \
--mode pl
```
