# Pose Estimation using Keypoint RCNN

**`Mert Özmeral & Jerome Habanz`**

### 🏚️Dataset

For training, we have used the [COCO2017-Dataset](https://cocodataset.org/)

#### 📁Folder Structure

```
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
```
<details open>
  <summary><h3>🔰Usage </h3></summary>

Make sure the required libraries are installed!

#### 🏋️Training
```
python train.py
```
```
losses=9.61300 loss_classifier=0.76776 loss_box_reg=0.01154 loss_keypoint=8.07662 loss_objectness=0.69867 loss_rpn_box_reg=0.05841:   0%|          | 0/42 [00:01<?, ?it/s]
...
losses=4.04951 loss_classifier=0.14994 loss_box_reg=0.24355 loss_keypoint=3.58709 loss_objectness=0.03577 loss_rpn_box_reg=0.03316:  17%|█▋        | 7/42 [14:41:34<68:04:01, 7001.19s/it]
```

#### ✅Validation
```
python validation.py
```
```
Evaluation for *bbox*:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.728
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.457
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.301
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.526
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.158
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.564
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.430
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.631
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.685
 
Evaluation for *keypoints*:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.516
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.776
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.551
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.499
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.849
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.655
```

#### 🎥Live Showcase with Webcam
```
python detect.py
```

#### 🎥Showcase Image
```
python detect.py --file image.png
```
![detected_image.png](docs%2Fassets%2Fdetected_image.png)

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
</details>

### 🐢References
- https://pytorch.org/vision/main/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn
- https://cocodataset.org