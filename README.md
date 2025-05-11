![art](./graphs/Pierre-Auguste_Renoir_3.jpg)

# ART STYLE CLASSIFICATION WITH DEEP LEARNING
___
## TABLE OF CONTENTS
___
1. [INTRODUCTION](#introduction)
2. [DATA](#data)
3. [DATA CLEANING/DATA WRANGLING](#data-cleaningdata-wrangling)
4. [EXPLORATORY DATA ANALYSIS](#exploratory-data-analysis)
5. [METHODOLOGY](#methodology)
6. [MODELING & NEURAL NETWORK ARCHITECTURE](#modeling-and-neural-network-architecture)
7. [RESULTS](#results)
8. [CONCLUSION & LESSONS LEARNED](#conclusion-and-lessons-learned)
9. [CREDITS](#credits)
10. [SOURCES](#sources)

## INTRODUCTION
___
*Museums are important hubs of recorded cultural history. However, they struggle with budget cuts
and rely on increased audience engagement to encourage economic support. 
Some museums, particularly smaller and lesser known ones, are not able to offer programs and activities 
that actively engage people in their local communities. 
New art enthusiasts can also be intimidated by art without background knowledge.* 

*An online poll from
art start-up Meural shows that from a group of 1,000 people, only 18% of all ages in the US could identify who painted
**The Girl with the Pearl Earring** (Hint: It's Johannes Vermeer!). However, 20% of people said they would appreciate art
if it was more accessible. With globalization and the internet, appreciating art has become a hugely popular experience shared 
by many people from more diverse backgrounds. There is an opportunity to reach and teach a broader group
of people. This can be made easier by being able to identify the style of a painting. 
For this project I will create a machine learning model that classifies the art style of a painting based on its image.* 

## DATA
___
To train the models, we will be using Icaros’ Best Artworks of All Time Dataset from Kaggle, a dataset with 8,446 images organized by 50 of the most renowned artists.

* [BEST ARTWORKS OF ALL TIME](https://www.kaggle.com/ikarus777/best-artworks-of-all-time)

We will be using the following packages:
* __keras__ for training and modeling data
* __skimage, maplotlib.pyplot,__ and __seaborn__ for visualizing the raw data, pre-processed images, and metrics
* __graphviz__ and __pydot__ for neural network visualization

## DATA CLEANING/DATA WRANGLING
___
To Best Artworks of All Time dataset contains 8,446 images organized by 50 of the most renowned artists. However, since we are interested in classifying by art style, I reclassified and relabeled the data accordingly using artists and genre then dropped all other columns. There are a total of 31 art styles noted in the data set as follows:

* Expressionism
* Expressionism/Abstractionism 
* Social Realism/Muralism 
* Impressionism
* Surrealism/Impressionism
* Surrealism
* Realism/Impressionism
* Byzantine Art
* Post-Impressionism
* Symbolism/Art Nouveau
* Northern Renaissance
* Suprematism
* Symbolism
* Cubism
* Baroque
* Romanticism
* Primitivism/Surrealism
* Mannerism
* Primitivism
* Proto Renaissance
* Early Renaissance
* High Renaissance
* Impressionism/Post-Impressionism
* High Renaissance/Mannerism'
* Realism
* Symbolism/Expressionism
* Expressionism/Abstractionism/Surrealism
* Neoplasticism
* Pop Art
* Symbolism/Post-Impressionism
* Abstract Expressionism

Since some artists are known for more than one art style, we will be excluding labels that categorize the style of paintings in multiple categories for better accuracy in the model. This reduces our dataset down to 6,669 images and 20 unique labels.

Furthermore, due to the computationally expensive nature of image classification, we will focus on 6 particular classes with varying quantities for a large spread of distribution:

* Impressionism
* Cubism
* Expressionism
* Pop Art
* Byzantine Art
* Abstract Expressionism

![art2](./graphs/sample_train.jpg)

This leaves us with a total of 2,306 images to test on over the 6 chosen classes.

We will be using ~60% or 1351 images to train the model in the chosen classes, which leaves 9% or 206 images to validate and 10% or 230 images to test and predict the accuracy of the models. A new sample of images is taken for the train and validation set everytime the model is run. The images in the test dataset do not change or are resampled again for the duration of the project to avoid the model ‘learning’ and ‘cheating’ when predicting. 


## EXPLORATORY DATA ANALYSIS
___
After visualizing the data, it’s very clear that the data is highly imbalanced. Impressionism has the most images while Abstract Expressionism has the least amount of images. The biggest concern in a heavily skewed distribution is that the model may predict each image as an Impressionist painting since it is the most common painting style that the model is learning from.

![art3](./graphs/dist.jpg)

To see if there are any patterns within each class, I looked at the pixel intensity of a random sample of images from the training dataset. Based on the results, some images are overexposed or underexposed and should be normalized during the pre-processing stage to better train the model.

![art4](./graphs/pixel_intensity.jpg)

## METHODOLOGY
___
The biggest challenges on this dataset will be its quantity of images as well as its highly skewed distribution across art styles. Thus, I will be evaluating the performance of two artificial neural networks-a Convolutional Neural Network (CNN) and a Residual Network with 50 layers (RESNET50) using the following methodologies in Keras:

1. __Data augmentation and pre-processing:__ Because the dataset is small to begin with, I use ImageDataGenerator from Keras to augment a sample of the data by applying distortions and saving them for classes lacking data such as “Abstract Expressionism”. This increases the robustness of the models and ability to generalize to unknown images. The data was augmented by being normalized and rotated between 0 to 40 degrees. We use the ‘nearest’ method to fill missing pixels after rotating or shifting the images. Since some images include people and landscapes, we only randomly flip the images horizontally instead of vertically to train the model properly. If the data only had abstract paintings it might make more sense to flip the images vertically as well. We zoom, stretch the images width and height, shear, and brighten the images at varying ranges. I randomly distort 75% of the images in each class sample to avoid overfitting and learning too many of the same images to still preserve the distribution of the data.

![art5](./graphs/sample_augmented.jpg)

2. __Weight Initialization:__ When fitting the model, I use class_weights from sklearn.utils to balance the distribution between art styles in the training dataset on both the CNN model and the RESNET50 model.


3. __Transfer Learning:__ There aren't enough images in the smaller classes to balance the data and learn from. Abstract Expressionism will be the hardest style to detect due to the small amount of images in this class. Having a very small dataset will make it very difficult for the base model to continue to learn so I will be using pre-trained weights from the ImageNet dataset pre-training on top of a RESNET50 model.

4. __Fine Tuning:__ From our transfer learning model, we will fine tune the last 20% of layers of the RESNET50 model, freeze all other layers, slow down the learning rate, and use a different optimizer to see if we can improve our model even further


## MODELING AND NEURAL NETWORK ARCHITECTURE
___
1. __Convolutional Neural Network:__ Due to the small size of our data set, we create a similar baseline convolutional neural network and reference [Francois Chollet’s](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) baseline convolutional model for classifying images in small datasets. The input shape of the structure is 200 x 200 x 3. Each stage of the structure is made of a convolutional layer followed by a max pooling layer for a total of three stages. The convolutional layers have filters that respectively generate 32, 64,128, and 128 feature maps. At the output stage, we flatten and dropout approximately 40% of the parameters to prevent overfitting after one fully connected layer. The last dense layer equals the number of classes for output and softmax activation in place of sigmoid for multi-classification.

   
[Visual of CNN](./graphs/cnn_model.jpg)

2. __Residual Neural Network model:__ I also built a ResNet50 model from scratch with the help of [Priya Dwivedi's Guide](https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33). The Residual Network has an input dimension of 200 × 200 × 3. The architecture contains two different blocks-an identity block and a convolutional block-that uses a ‘shortcut connection’ at each block. 
    * The identity block has a convolutional 2D layer at each stage followed by a batch normalization process for a total of three stages. The input value saved at the beginning of the first stage is added back to the main path in the last stage and passed through a Relu activation. 
    * The convolutional block has a convolutional 2D layer at each stage followed by a batch normalization process for a total of three stages. A skip connection path with a convolutional and batch normalization layer is created for the input saved at first stage. It is then added back into the main path and passed through a Relu activation. 
    * The full ResNet50 model has a convolutional 2D layer in the first stage followed by a batch normalization and max pooling layer. The next 4 stages are composed of a convolutional block and 2, 3, 5, 2 identity blocks respectively. 
    * The output layer uses Average Pooling followed by one fully connected layer with the number of classes and softmax activation for multi-classification. 
    
    In total, there are 53 convolutional layers (hence the name resnet50 for approximately 50 layers in the residual neural network).

[Visual of ResNet50](./graphs/resnet50_model.jpg)   

Again, to overcome the limitation of having a small dataset, we will be using weight regularization and pre-training for object recognition on ImageNet. 


## RESULTS
___
![metric1](./img_metrics/train_val_accuracy.jpg)

Based on the results for each model, it's very clear that the fine-tuned RESNET50 and pre-trained RESNET50 performs the best, followed by the simple CNN model with a validation accuracy of ~88, ~87% and ~76% at 25 epochs respectively. 

The RESNET50 and weighted RESNET50 we created performs significantly worse with a validation accuracy of at ~53% and ~12% respectively at 25 epochs. This could potentially be attributed to a number of factors such as lack of data and underfitting. 

This is especially noted for the weighted models. There aren't enough images in the smaller classes to balance the data and learn from. Abstract Expressionism is the hardest style to detect due to the small amount of images in each class.

Having a very small dataset made it very difficult for the base model to continue to learn so fine tuning on top of a RESNET50 model and using pre-trained weights from the ImageNet dataset helped improve the accuracy.

![metric4](./img_metrics/accuracy_df.jpg)

Here we can see how well each individual class is predicted. While, Abstract Expressionism isn't correctly predicted at all, the Pre-Trained and Fine-Tuned model still predicts 2/3rds of the classes above 70%.

### BEST MODEL: PRE-TRAINED RESNET50 ON IMAGENET v. FINE TUNED MODEL
___
![metric2](./img_metrics/metric_comparison1.jpg)

Based on the above confusion matrix from our pre-trained model, the individual class that performed the best were Pop Art, Cubism, and Impressionism for both of the best models.

![comparison1](./graphs/comparison1.jpg)

As expected, Abstract Expressionism had very low predictions due to having such a small quantity of images to learn from in our pre-trained confusion matrix. The true labels were misclassified as Cubism and Impressionism. Given the limited data, this intuitively makes sense since Impressionism has a very open composition that appears abstract up close and Cubism is a specific, yet abstract form of art. It would be interesting to see if increasing the dataset will continue to misclassify Abstract Expessionist paintings or if the model will learn to distinguish minute differences between art styles.

![comparison2](./graphs/comparison2.jpg)

Again, we see another example of an Impressionist painting appearing abstract and 'messy' up close, similar to the Abstract Expressionism painting.


![metric3](./img_metrics/metric_comparison2.jpg)

Due to such an uneven class distribution, we will mostly be looking at the recall score because we want to identify distinct features not easily obvious between paintings. Cubism and Impressionism have the highest metrics across the board, potentially due to the fact that they make up majority of the dataset. Surprisingly, despite its small quantity, Pop Art has a higher precision, recall, and f1-score than Impressionism. Perhaps this is due to the unique nature of the art style, which repeats various images in different colors and allows the data to train on repeating patterns despite the pixel intensity of each image. This suggests that the model seems to detect styles with significant repeating patterns and struggles with styles closer to one another much like a person would probably do.

![comparison3](./graphs/comparison3.jpg)


## CONCLUSION AND LESSONS LEARNED
___

In conclusion, based on all our results and metrics, the best model for classifying paintings is the Pre-Trained ResNet model. I chose the Pre-Trained ResNet Model over the Fine-Tuned model due to overfitting. The training set for the Fine Tuned model remains close to 100% while the validation set only reaches 88% accuracy. We successfully applied a deep learning approach to achieve overall 87% accuracy on a pre-trained ResNet50 with the ImageNet database.


To improve the model, I would like to do some of the following in the future:


* __Pre-train Other Models:__ In the future I would like to pre-train on other models such as the VGG-16, Inceptionv3, or EfficientNet and compare which model performs the best. 


* __Methodology:__ Because the data had trouble identifying images with similar features or images with very little data to begin with, it might be beneficial to combine certain classes together. Implementing Bagging and K-Fold Cross-Validation to improve accuracy would be another methodology I would be interested in  implementing.


* __Diversity and Larger Dataset:__ I would also be interested in expanding my dataset to more diverse forms of art and include more images. The 50 most famous artists are mostly of European descent and it would be beneficial for the model to learn more cultural and diverse artworks from other countries and backgrounds.


## CREDITS
___

Thank you to my patient and wonderful mentor Nik Skhirtladze, Francois Chollet and Priya Dwivedi on their helpful tutorials, and Stack Overflow for all my troubleshooting needs.


## SOURCES
___
* [Image Classification Using Very Little Data by Francois Chollet](https://blog.keras.io/building-powerful-image-classification-models-using-very-littledata.html)
* [Kaggle Best Artworks of All Time](https://www.kaggle.com/ikarus777/best-artworks-ofall-time)
* [Keras API](https://keras.io/api/)
* [MOMA Contemporary Challenges](https://www.britannica.com/art/museum-of-modern-art-institution/Contemporary-challenges)
* [Stack Overflow](https://stackoverflow.com/)
* [Understanding ResNet in Keras by Priya Dwivedi](https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras446d7ff84d33)

