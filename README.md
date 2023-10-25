# **Proposal of an advanced deep learning model and its performance evaluation for real-time Vespa detection**

![123그림1](https://github.com/Kwon-YJ/temp123/assets/49362903/bf700e35-077b-4aad-a7e7-bfda4c70019a)

This paper introduces a new advanced YOLOX model for real-time Vespa monitoring and evaluates its detection performance compared to the vanilla YOLOX. The YOLO-based deep learning models have been widely utilized in real-time object detection. Despite the sustained progress of YOLO-based deep learning models, the detection of small objects such as Vespa is still one of the adventurous tasks. Now, the most effective object detection model for small objects is the YOLOX of an anchor-free model. Thus, this paper proposes a better YOLOX model in which a new feature extraction layer based on the shuffle layer is introduced to obtain advancement of detection performance in small objects such as Vespa reducing inference speed, simultaneously. To evaluate the detection performance of the proposed model, 5 kinds of experiments are conducted and compared to the vanilla YOLOX. First, the mAP@50 evaluation for 900 test images of the custom Vespa dataset is carried between the proposed and the vanilla YOLOX as a detection accuracy test. As a result, mAP was 93.8% and 89.6%, respectively. The proposed method showed better performance by a margin of 4.2%. Second, a data generalization test is evaluated for a dataset of forestry pests with 31 species. This test is for the verification of two models for expansion onto the detection of various insect species. Both models function well for the detection of 31 kinds of forest pests showing over 99% accuracy of mAP@50. Third, a detection accuracy test for small objects is done. For this test, we preprocessed all images of the custom Vespa dataset to obtain resized Vespa objects with a 0.3% size ratio for all 900 test images, in which the average pixel size ratio between each Vespa object’s pixel size to the corresponding whole image’s pixel size is set to approximately 0.3% to consider Vespa image capture of realistic monitoring environments. The proposed structure also showed better results by a 1.14% mAP@50 margin in experiment 3 for 0.3% test Vespa images with the same size of 900 and average 16M pixels, dedicated small object evaluations. The fourth and the fifth experiments are the inference time comparison for 0.3% test Vespa image dataset of experiment 3 and FHD-sized webcam images. In terms of inference speed, the proposed model was 1.34 ~ 1.35 times faster than the vanilla model due to optimized convolution operation in the backbone. Therefore, we can verify that our advanced model for YOLOX is more effective than the state-of-the-art YOLOX model in terms of inference accuracy and speed and can be widely applied in the fields of small object detection such as Vespa monitoring, pest control, and insect detection.

## Installation

- we only tested in torch 2.0.1, CUDA 12.1, cuDNN 8.9.2, Docker 24.0.2(cb74dfc)

- Download docker file : https://drive.google.com/file/d/1frf_khaO7FhViVYDkr4g4Er_6C9fLhgR/view?usp=drive_link

```bash
xz -d shuffle_yolox_lite.tar.xz
docker load -i shuffle_yolox_lite.tar
docker run -it <tag>
```

## Usage

```
# docker container 

# train vespa dataset
python3 tools/train.py configs/vespa/v4.py

# eval vespa dataset
python3 tools/test.py configs/vespa/v4.py work_dirs/v4/epoch_300.pth --eval bbox

# train pest dataset
python3 tools/train.py configs/yolox/yolox_x_8xb8-300e_coco.py 

# eval past dataset
python3 tools/test.py configs/yolox/yolox_x_8xb8-300e_coco.py work_dirs/yolox_x_8xb8-300e_coco/epoch_300.pth --eval bbox
```
