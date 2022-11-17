For the Argoverse 1.1 dataset:

1. Download the Motion Forecasting (v1.1) data from and extract the corresponding files in /home/your_user/Downloads (Host):

```
wget https://s3.amazonaws.com/argoai-argoverse/forecasting_train_v1.1.tar.gz
wget https://s3.amazonaws.com/argoai-argoverse/forecasting_val_v1.1.tar.gz
wget https://s3.amazonaws.com/argoai-argoverse/forecasting_test_v1.1.tar.gz
tar xvf forecasting_train_v1.1.tar.gz
tar xvf forecasting_val_v1.1.tar.gz
tar xvf forecasting_test_v1.1.tar.gz
```

2. After cloning this repository:

```
cd data
mkdir -p datasets/argoverse/motion-forecasting
cd datasets/argoverse/motion-forecasting
ln -s ~/Downloads/forecasting_train_v1.1 train
ln -s ~/Downloads/forecasting_val_v1.1 val
ln -s ~/Downloads/forecasting_test_v1.1 test
```