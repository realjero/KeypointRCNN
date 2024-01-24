# Pose Estimation using Keypoint RCNN

**`Mert Özmeral & Jerome Habanz`**

### 🏚️Dataset

For training, we have used the [COCO2017-Dataset](https://cocodataset.org/)

#### 📁Folder Structure

````
root
├───coco
│   ├───annotations
│   │   ├───person_keypoints_train2017.json
│   │   └───person_keypoints_val2017.json
│   ├───train2017
│   │   └───(...).jpg
│   └───val2017
│       └───(...).jpg
...
````

### 🔰Usage
Make sure the required libraries are installed!

#### 🏋️Training
```
python train.py
```

#### ✅Validation
```
python validation.py
```

#### 🎥Live Showcase
```
python webcam.py
```



#### 📚Libraries
⬇️ or check requirements.txt
```
matplotlib
numpy
opencv_python
Pillow
pycocotools
torch
torchvision
tqdm
```

### 🐢References
- https://pytorch.org/vision/main/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn
- https://cocodataset.org