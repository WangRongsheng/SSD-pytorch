SSD：Single-Shot MultiBox Detector目标检测模型在Pytorch当中的实现

## 训练步骤
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的`Annotation`中。  
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的`JPEGImages`中。  
4. 在训练前利用`voc2ssd.py`文件生成对应的txt。  
5. 再运行根目录下的`voc_annotation.py`，运行前需要将`classes`改成你自己的`classes`。**注意不要使用中文标签，文件夹中不要有空格！**   
```python
classes = ["aeroplane", "bicycle"]
```
6. 此时会生成对应的`2007_train.txt`，每一行对应其**图片位置**及其**真实框的位置**。  
7. **在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类**，示例如下：   
`model_data/new_classes.txt`文件内容为：   
```python
cat
dog
...
```
8. 将`utils.config`的`num_classes`修改成所**需要分的类的个数+1**，运行`train.py`即可开始训练。

> 训练所需的`ssd_weights.pth`可以在[百度云下载（提取码: uqnw）](https://pan.baidu.com/s/11WUye_Xy4cTJvpZlmB6g3g)。 下载并放入`model_data`。


## 预测步骤

> 如果你的训练不是在`pytorch<=1.3`进行的，预测会出错，你需要转化模型，可以使用`load_save_model.py`。

1. 按照训练步骤训练。  
2. 在`ssd.py`文件里面，在如下部分修改`model_path`和`classes_path`使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    "model_path": 'model_data/ssd_weights.pth',
    "classes_path": 'model_data/voc_classes.txt',
    "model_image_size" : (300, 300, 3),
    "confidence": 0.5,
    "cuda": True,
}
```
3. 运行`predict.py`，输入  
```python
img/street.jpg
```
4. 利用`video.py`可进行摄像头检测。  

## 评估步骤

步骤是一样的，不需要自己再建立`get_dr_txt.py`、`get_gt_txt.py`等文件。  
1. 本文使用VOC格式进行评估。  
2. 评估前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的`Annotation`中。  
3. 评估前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的`JPEGImages`中。  
4. 在评估前利用`voc2ssd.py`文件生成对应的txt，评估用的txt为`VOCdevkit/VOC2007/ImageSets/Main/test.txt`，需要注意的是，如果整个VOC2007里面的数据集都是用于评估，那么直接将`trainval_percent`设置成0即可。  
5. 在`yolo.py`文件里面，在如下部分修改`model_path`和`classes_path`使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
6. 运行`get_dr_txt.py`和`get_gt_txt.py`，在`./input/detection-results`和`./input/ground-truth`文件夹下生成对应的txt。  
7. 运行`get_map.py`即可开始计算模型的mAP。
