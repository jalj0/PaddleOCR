## Introduction
PaddleOCR aims to create multilingual, awesome, leading, and practical OCR tools that help users train better models and apply them into practice.

# 1) Training the Detection part with Custom Dataset:
## 1.1) Env setup:
```python
conda install python=3.8
```
paddlepaddle install:<br/>
a) if you are usinf GPU:
```python
conda install paddlepaddle-gpu==2.6.1 cudatoolkit=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```
b) if you are using CPU:
```python
conda install paddlepaddle==2.6.1 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```
check paddle installed properly:
```python	
 python
>>import paddle
>>paddle.utils.run_check()
>>import paddle.distributed as dist
>>print(dist.ParallelEnv().dev_id)
```
```python
conda install anaconda::yaml
```
Install library dependencies using,
```python
pip install -r requirements.txt
```
## 1.2) Dataset Preperation:
1. Dataset used: [Indian Vechile License Plate](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset) ==> contains images with corresponding yml file.
2. First thing first, I accumulated all the images into same folder "dataset/"
3. Using simple python code we can split the dataset into train, test and validation; follow split_train_test_val.py
```python
python split_train_test_val.py
```
4. After spliting the dataset, create txt file where first column contains the filename(with path) and 2nd column contains the (key,value) type of pairs: follow create_anno_dec.py. For the Detector we need to create two .txt files for training and validation respectively with the format as follows:<br/>
image_path\t[{“points”:[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], “transcription”:text_annotation}, {“points”:…..}]\n
![d2](https://github.com/user-attachments/assets/d8c8f5a0-ea3c-411e-a110-6617eead70ee)
5. Repeat 4 for test & val set also.
Upto here you have created the dataset required for PaddleOCR training of Detection part with Custom dataset.

## 1.3) Download the pre-trained weights
1. In the directory PaddleOCR/,
```python
mkdir pretrain_models
cd pretrain_models
```
2. Download the weights in pretrain_models/ directory,
```python
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams
```
You can download any other pretrained model but remember to choose the correct config file for training.

## 1.4) Configure your YML file
For my project i choose "MobileNetV3_large_x0_5_pretrained" model with [Differential Binarization](https://arxiv.org/pdf/1911.08947) algorithm. Therefore corresponding .yml has to be choosen i.e., "det_mv3_db.yml".
You will find the required config file at "configs/dec/det_mv3_db.yml".
You can change parameters like number of epochs, learning rate, GPU specification, the number of epochs after which the model state is saved, etc. You will also have to provide the path to your train data and train labels under the train section, Similarly, provide the path to the validation data and labels in the evaluation section. Do not touch any of the part of architecture of the model. Let’s look at all the lines you will have to change in this file in order for you to run the training script.
![dec4](https://github.com/user-attachments/assets/df506190-6974-460f-929a-1e303c437dba)


## 1.5) Fine Tuning the Model
1. Fine-tune the model with your dataset:
```python
python3 tools/train.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model=./pretrain_models/pretrain_models/MobileNetV3_large_x0_5_pretrained
```
2. Continue fine-tuning, you need to provide your output checkpoint file path:
```python
python3 tools/train.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model=./pretrain_models/pretrain_models/MobileNetV3_large_x0_5_pretrained Global.checkpoints={path to output model folder}/latest
```
Global.checkpoints has greater priority than pretrained_model. Hence the latest saved model will be used.

## 1.6) Exporting the Fine Tuned model to an inference model
From the directory PaddleOCR/
```python
python3 tools/export_model.py -c {path to yml file inside the fine tuned model folder}  -o Global.pretrained_model={path to model folder} Global.save_inference_dir={path to inference model folder}
```
For me the below code worked,
```python
python3 tools/export_model.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model=./output/db_mv3/best_accuracy Global.save_inference_dir=./output/det_db_inference/
```
Now you should have a new folder (at the path you provided), with the .pdi files that are required to infer with the model.

## 📚 Documentation
Full documentation can be found on [docs](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html).

