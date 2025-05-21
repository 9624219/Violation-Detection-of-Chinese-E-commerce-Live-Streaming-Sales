## CLIVESVD

This repo is the implementation of "Violation Detection of Chinese E-commerce Live Streaming Sales".

## Download Codes

1. Download Codes

```
git clone https://github.com/9624219/Violation-Detection-of-Chinese-E-commerce-Live-Streaming-Sales.git
```

2. Download Models

You need to download the model from [here](https://huggingface.co/google-bert/bert-base-chinese/tree/main), and put it into folder "/bert-base-chinese". You can also download other models. After downloading, please modify the model path in the code accordingly.

## Environment Setup

Install the required dependencies (might take some time)

```
conda create -n <your env name> python=3.10
conda activate <your env name>
pip install -r requirement.txt
```

Regarding datasets, you can use a crawler tool to collect additional live stream data and apply our method for violation detection. For the crawler tool, you may refer to [here](https://github.com/ihmily/DouyinLiveRecorder).

## Quickstart

1. Train the model

Modify the dataset path in `train.py`, then run the following command to train the model:

```
python train.py
```

2. Test the model

After updating the model path, use the following command to test the model:

```
python Model/2stage_test.py
```

## Results

![Result](https://github.com/9624219/Violation-Detection-of-Chinese-E-commerce-Live-Streaming-Sales/blob/master/assets/roc.png)
