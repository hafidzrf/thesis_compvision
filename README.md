# thesis_compvision
Detailed code of my graduate thesis in Bandung Institute of Technology in 2022

# Abstract
Various studies have been carried out to produce a map of the area's vulnerability due to earthquakes. Indonesia, with its high earthquake risk, already has an earthquake risk map based on the type of building for each region. However, this map still assumes the same building types for a large area (about 1000 km2).

Several methods can be used to identify the building vulnerabilities due to earthquakes. Starting from analytical methods such as NLTHA and pushover analysis, to empirical methods such as rapid visual assessment. An analytical method is ideal to identify building vulnerabilities that have known detailed model and morphology.

However, this method are difficult to implement on a wide scale. Consequently, empirical methods, although have lower accuracy, allows for rapid execution, and therefore often implemented to map regional vulnerabilities.

The European Commission in 2006 released the EMS-98 macroseismic model to estimate earthquake vulnerability and damage to existing buildings based on visual features. The application of this visual classification of building typology, if carried out on a massive scale, will provide information on the vulnerability of the area quite well, and able to provide an overview to stakeholders regarding the risk of economic impacts that will occur in the next potential earthquake. In practice, the EMS-98 method is carried out by visiting each building and making visual observations by surveyors. However, if this implementation is carried out on a large scale, the cost will be exorbitant.

There are various options to reduce the cost of this assessment. For the classification of building typologies, deep learning techniques based on convolutional neural network (CNN) can be used to identify visual features from photos hierarchically. 

Several building typologies from EMS-98 were considered in this study, namely confined masonry, RC infilled masonry, timber structure, and unconfined masonry. Data augmentation is used to increase the generalization ability of the CNN model and to balance the unbalanced class distribution, by oversampling the minority typology classes such as timber structure and unconfined masonry.

In this study, the CNN deep learning algorithm will be applied to datasets with various data sources, number of samples, and CNN architecture. The CNN architectures considered in this study are InceptionV3, Xception, MobileNet V3L, and EfficientNet B0. Model performance is measured by the f1-score. Domain transfer is carried out between models with different data sources to test the robustness of the model.

These CNN models use weights from ImageNet. The models were trained with a train-validation-test ratio of 70:15:15, and the Adam optimizer is searched for  its optimal hyperparameters using grid search. Efforts to regularize the model are carried out by adding L2 regularization, adding dropout layers, transforming photos of buildings with augmentation, and label smoothing techniques. Fine-tuning is carried out progressively starting from the top convolution layer to the stem convolution layer with a continuously decreasing learning rate.

Comparing different model architectures, the test results show that the EfficientNetB0 model provides the highest f1-score with an average performance of 78.12% from 5 datasets and 4 building typologies.

Comparing the number of samples for the training process, for the dataset of the Google Street View (GSV) virtual tour, doubling the sample size in the dataset led to an increase in f1-score performance by up to 10% for the minority typology class. For datasets from data mining using the GSV API, quadrupling the sample size leads to an increase in f1-score up to 45% for minority typology classes and 12% in weighted average. In the domain transfer process, the dataset from the GSV virtual tour has the best f1-score when tested with other domains, reaching 70% when tested on the GSV data mining dataset, and 61% when tested on the camera photo dataset.

Comparing the dataset sources, photos taken from data mining using the GSV API have an average f1-score of 87.61%, 17% higher than the dataset by camera photos with relatively similar data retrieval effort.

Evaluation of decision making by CNN model is visualized with GradCAM algorithm, and has shown quite good results, where the model in the top convolution layer has highlighted the desired building objects and ignored non-relevant objects.

Testing of this CNN system in Bandung city using data mining shows that of 1538 sample photos of buildings taken randomly, 58% are confined masonry, 39% are RC infilled masonry, 2% are timber structures, and 1% are unconfined masonry. Manual inspection are carried out in some of the predicted sample photos and demonstrated results as expected. Several things to note are that data mining from GSV are unable to reach remote areas that are inaccessible using a GSV vehicle, thus making images of unconfined masonry buildings and timber structures, which are often found in suburban and rural areas, still difficult to obtain.

Keywords : building vulnerabilities, building typology, Google Street View, data mining, data augmentation, convolutional neural network, domain transfer

