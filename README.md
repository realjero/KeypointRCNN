# Pose Estimation using Keypoint RCNN

**`Mert Özmeral & Jerome Habanz`**

### 📝 Description
This project implements Keypoint RCNN using PyTorch, trained on the COCO 2017 dataset. It includes a training script for model optimization, a test script for evaluation, and a user-friendly testing framework for real-time webcam analysis. The training script refines model parameters for accurate keypoint detection. The test script assesses model performance on validation data or custom images. The testing framework enables interactive webcam testing, showcasing the model's real-world capabilities. Together, these components empower users in pose estimation and related computer vision tasks.

### 🏚️Dataset

For training, we have used the [COCO2017-Dataset](https://cocodataset.org/)
- [train2017](http://images.cocodataset.org/zips/train2017.zip)
- [val2017](http://images.cocodataset.org/zips/val2017.zip)
- [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [Data Format](https://cocodataset.org/#format-data)

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
├───e42_b8_lr0.02_m0.9.pth
...
```

<details open>
  <summary><h3>🔰Usage </h3></summary>

Make sure the required [packages](#packages) are installed!

<details>
  <summary><h4>🏋️Training</h4></summary>

```
python train.py
```

###### Output:

```
losses=9.61300 loss_classifier=0.76776 loss_box_reg=0.01154 loss_keypoint=8.07662 loss_objectness=0.69867 loss_rpn_box_reg=0.05841:   0%|          | 0/42 [00:01<?, ?it/s]
...
losses=4.04951 loss_classifier=0.14994 loss_box_reg=0.24355 loss_keypoint=3.58709 loss_objectness=0.03577 loss_rpn_box_reg=0.03316:  17%|█▋        | 7/42 [14:41:34<68:04:01, 7001.19s/it]
```

</details>

<details>
  <summary><h4>✅Testing</h4></summary>

```
python test.py
```

###### Output:
36
```
Evaluation for *bbox*:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.811
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.582
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.620
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.187
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.548
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.636
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.701
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.781
 
Evaluation for *keypoints*:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.642
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.853
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.701
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.608
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.707
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.711
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.904
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.665
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.777
```

</details>

<details>
  <summary><h4>🎥Live Showcase with Webcam</h4></summary>

```
python detect.py
```

</details>

<details>
  <summary><h4>🎥Showcase Image</h4></summary>

```
python detect.py --file image.png
```

###### Output:

![detected_image.png](docs%2Fassets%2Fdetected_image.png)
</details>

<details>
  <summary id="packages"><h4>📚Packages</h4></summary>
The project is compatible with 🐍Python 3.10

You can find the packages used in _requirements.txt_ or down below⬇️
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

</details>

### 🔻Results

###### Google Drive:
- [e42_b8_lr0.02_m0.9.pth](https://drive.google.com/file/d/1p6MW5PYiz9c8BSYb0iLd4qx1w64Be3Ww/view?usp=sharing)


### 🐢References

- https://pytorch.org/vision/main/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn
- https://cocodataset.org