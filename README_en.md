## Introduction
PaddleOCR aims to create multilingual, awesome, leading, and practical OCR tools that help users train better models and apply them into practice.

# Training the Recognizer part with Custom Dataset:
## Env setup:
```python
conda install python=3.8
```
paddlepaddle install:
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
## Dataset Preperation:
1. Dataset used: [Indian Vechile License Plate](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset) ==> contains images with corresponding yml file.
2. First thing first, I accumulated all the images into same folder "dataset/"
3. Using simple python code we can split the dataset into train, test and validation; follow split_train_test_val.py
```python
python split_train_test_val.py
```
4. After spliting the dataset, create csv file where first column contains the filename(with path) and 2nd column contains the label(license number): follow create_path_label_csv.py
```python
python create_path_label_csv.py
```
5. Create Annotation file: PaddleOCR takes annotations in txt file. You can create txt file for all(train.csv, test.csv, val.csv) with the help of a script called gen_label.py in the PaddleOCR/ppocr/utils folder in the PaddleOCR package.
```python
python gen_label.py --mode="rec" --input_path={path to csv file} --output_label=(folder to output txt.file}
```
![git1](https://github.com/user-attachments/assets/d3e228c4-1405-4cd9-8173-c1342cbb974a)
Upto here you have created the dataset required for PaddleOCR training of Recognition part with Custom dataset.

## Download the pre-trained weights
1. In the directory PaddleOCR/,
```python
mkdir pretrain_models
cd pretrain_models
```
2. Download the weights in pretrain_models/ directory,
```python
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar
tar -xf en_PP-OCRv3_rec_train.tar && rm -rf en_PP-OCRv3_rec_train.tar
```
## Configure your YAML file
You can change parameters like number of epochs, learning rate, GPU specification, the number of epochs after which the model state is saved, etc. You will also have to provide the path to your train data and train labels under the train section, Similarly, provide the path to the validation data and labels in the evaluation section. Do not touch any of the part of architecture of the model. Let’s look at all the lines you will have to change in this file in order for you to run the training script.
![git2](https://github.com/user-attachments/assets/b1ec4ab2-829c-4e67-b7a8-767f1ffa68c5)



## Fine Tuning the Model
1. Fine-tune the model with your dataset:
```python
python3 tools/train.py -c {path to config file} -o Global.pretrained_model={path to pretrained model}/best_accuracy Global.checkpoints={path to pretrained model}/latest
```
2. Continue fine-tune, you need to provide your output checkpoint folder path:
```python
python3 tools/train.py -c {path to config file} -o Global.pretrained_model={path to pretrained model}/best_accuracy Global.checkpoints={path to output model folder}/latest
```

## Exporting the Fine Tuned model to an inference model
From the directory PaddleOCR/
```python
python3 tools/export_model.py -c {path to yml file inside the fine tuned model folder}  -o Global.pretrained_model={path to model folder} Global.save_inference_dir={path to inference model folder}
```
Now you should have a new folder (at the path you provided), with the .pdi____ files that are required to infer with the model.

## Write an inference script to test out the fine-tuned model.
1. Just copy the code below and store it in a file called test.py:
```python
from paddleocr import PaddleOCR,draw_ocr

ocr = PaddleOCR(rec_model_dir="/scratch/PaddleOCR/pretrain_models/model_inference",use_angle_cls=True, lang='en') 
img_path = '/scratch/PaddleOCR/dataset/val/5.png'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
print(txts)
```
Change the rec_model_dir argument to the path of your fine-tuned model, and the image path.
2. Run the inference script, test.py:
```python
python /scratch/PaddleOCR/test.py
```
## 📚 Documentation
Full documentation can be found on [docs](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html).

