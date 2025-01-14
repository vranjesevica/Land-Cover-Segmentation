# Land Cover Segmentation Project

This project focuses on land cover segmentation using the U-Net model. **The idea** behind this project was to learn how image segmentation works, in order to later develop a model capable of classifying different types of land cover and based on that, recognizing planted crops or potential plant diseases, all using satellite imagery. **The goal** of the project is to classify and segment different types of land such as urban areas, agricultural land, forests, water, and others using satellite images. I used **DeepGlobe dataset**, which contains a variety of satellite images along with corresponding masks defining the different land types. The project covers all steps from downloading and preparing the data, image augmentation, to training and evaluating the model. Each of these steps is described below.

## Small introduction to Image Segmentation
**Image segmentation** is the process of dividing an image into parts or regions that are similar in certain characteristics, such as color, texture, or intensity. I choosed this tehnique because image segmentation is used to identify specific objects or regions within an image, it separates relevant parts of the image from the irrelevant ones, which allows precise labeling and analysis of specific objects or areas. For example, in a land segmentation project, the aim is to identify different types of land cover (such as urban areas, forests, water, etc.) based on satellite images - exatcly what I have here.

Segmentation can be performed using many methods, but after some research I choosed **U-net architecture** because it is very commonly used and gives very good results for this type of deep learning task.

## Loading dataset

The dataset is downloaded from Kaggle. It is the **DeepGlobe Land Cover Classification dataset**. The DeepGlobe dataset is used for automatic land cover classification and segmentation. This dataset is part of the DeepGlobe challenge for land cover classification, aimed at improving sustainable development, autonomous agriculture, and urban planning through automatic land cover classification.

#### Data Characteristics

* The training set contains 803 RGB satellite images.
* Each image has dimensions of 2448x2448 pixels.
* The pixel resolution is 50 cm.
* The dataset also includes 171 validation images and 172 test images, but test images do not contain masks.

#### Sources

* Images are collected by satellites from DigitalGlobe.

#### Labels and Masks

Each satellite image has a corresponding annotation mask, which is an RGB image with 7 classes of land cover. Each class has a specific color code:

*   RGB (0, 255, 255) -> Urban Land – urban settlements, man-made areas
*   RGB (255, 255, 0) ->Agricultural Land – farms, fields, orchards, vineyards
*   RGB (255, 0, 255) -> Grassland – non-forest, non-agricultural green land
*   RGB (0, 255, 0) -> Forest Land – areas with high tree canopy density
*   RGB (0, 0, 255) -> Water – rivers, lakes, oceans
*   RGB (255, 255, 255) -> Barren Land – mountains, deserts, beaches without vegetation
*   RGB (0, 0, 0) -> Unknown – clouds and unknown areas

## Data Preparation - Preprocessing

This cell sets the parameters and functions for loading, processing, and augmenting satellite images and their corresponding masks. Data loading is done from the metadata CSV file, while augmentation is performed using horizontal flipping and random rotations of images. Conversion between colors and classes is enabled using KDTree for fast mapping. Finally, the data is prepared for model training. A visualization of the prepared data is done on one image that is flippedn adn randomly rotated to verify that the masks are correctly loaded and processed.
<div align="middle">
	<img src="https://github.com/user-attachments/assets/b07a3d5b-a6d3-48f5-a02f-39c2bdbcba73"/>
	<img src="https://github.com/user-attachments/assets/9f6d34c8-b014-4752-a048-e7ff4771eff3"/>
	<img src="https://github.com/user-attachments/assets/7a252fd8-2349-4cd3-840a-9bef6f58c38a"/>
</div>

#### Data Preprocessing Results

After processing the original data, I obtain 2409 images for input into the model. After splitting into training, test, and validation sets, I have:

* 1734 images for model training
* 193 images for validation
* 482 images for testing model performance

#### Dataset Preparation

Firstly, there is the LandCoverDataset class, which inherits from the dataset and contains the image and mask data. This class also performs the conversion of images into the CHW format required for processing in PyTorch. I created datasets and data loaders for training, testing, and validation using the LandCoverDataset class and then passed through the DataLoader for batching and permutation.

## Defining the U-Net Model

U-Net is a deep learning architecture specifically designed for image segmentation and that is main reason I used it. U-Net is a convolutional neural network. The UNet class implements an encoder-decoder architecture with additional convolutional blocks. The architecture consists of two parts: the encoder, which extracts features from the input image by applying convolutional filters to detect various image characteristics, and the decoder, whose goal is to reconstruct the image to its original resolution using features from the encoder. Additionally, I implemented skip connections between corresponding encoder and decoder layers, allowing the decoder to use high-resolution features generated by the encoder, improving segmentation accuracy.
<div align="middle">
	<img src="https://github.com/user-attachments/assets/c02df0a6-d1b4-4db1-963f-2afbae61ae9d"/>
</div>

The U-Net model is set on the available device (CPU or GPU), the loss is computed using CrossEntropyLoss, and optimization is performed with the Adam optimizer.

#### Model Training and Validation

The loop goes through epochs, trains the model on the train set, computes loss and accuracy, and then evaluates the model on the validation set. After each epoch, the loss and accuracy results for training and validation are displayed.

## Model Testing

Testing was done on the test dataset. The testing results are printed, and based on them, it can be concluded that the model has solid performance. Here is some example:
<div align="middle">
	<img src="https://github.com/user-attachments/assets/05addf86-5afd-4cec-8955-a48111e97ee6"/>
	<img src="https://github.com/user-attachments/assets/08bd97f2-3535-4536-809d-e6450cbac901"/>
	<img src="https://github.com/user-attachments/assets/7bf66dc5-0a99-418b-bf04-e20c3c915108"/>
	<img src="https://github.com/user-attachments/assets/766bb5c7-ff75-4c95-9d7e-4819e3cf3c11"/>
	<img src="https://github.com/user-attachments/assets/e99a12a0-826e-4126-adf1-88ca393a3975"/>
</div>

As we can see in the last image, my model struggles to accurately recognize small areas, which may be influenced by several factors. One of them is the high resolution of these images, which requires powerful hardware capabilities. Due to this, I had to use relatively large batch sizes, as using smaller ones caused training to crash. Additionally, the architecture of the encoder and decoder could be made more complex, which would likely yield better results. However, in the other four images, it is evident that the model is performing well overall.

## Model Evaluation

#### IoU (Intersection over Union)
The IoU metric gives a result of 63%. This metric measures the average overlap between my predicted regions (masks) and the actual regions (target labels). This result means that, on average, 63% of my predicted regions overlap with the actual regions.
<div align="middle">
	<img height='300' src="https://github.com/user-attachments/assets/191c632a-bc29-43bb-ab43-4279289b3bc3"/>
</div>

#### F1 Score
The F1 score combines precision (how accurately the model found relevant instances) and recall (how many of the actual instances were found by the model). My result of 76% indicates the average precision and recall of my model.

## Conclusion

Based on the obtained outputs, I can see that my model performs quite well in finding the appropriate regions. There are errors, which occur more frequently when regions occupy a small area (few pixels), and the model fails to register them, as well as errors on the borders (where one region ends and another starts).

I also provided code for manual testing of the model. By entering the appropriate path to an image, it is possible to predict the mask for the given image and simultaneously open the corresponding real mask to see the similarity between them. I think this is good way of showing results for any image you want. 	
<div align="middle">
	<img height='450' src="https://github.com/user-attachments/assets/ed47acc5-6634-4a1d-a916-8359a6edb8f6"/>
</div>
