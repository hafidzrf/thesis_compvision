# The general methodology of this study, represented in codes available in this folder are:

1. Design of building-nonbuilding binary classification CNN model to aid the subsequent data mining process with **Building - Non Building GSV Binary Classification.ipynb**
2. Execution of data mining process (specifically, for Random Sampled Dataset) using Google Street View API with **Image_Generator_StreetView API.ipynb**
3. Dataset compilation (manual and/or automated), and organizing the train-test-split for each dataset using **Dataset Creation.ipynb**
4. Perform augmentation process *only in training set* to enhance generalization capability of CNN model using **Data Augmentation Generator.ipynb**
5. Perform Hyperparameter optimization for Adam learning rate and batch size for each dataset, using random search method. **Hyperparameter Optimization.ipynb**
6. Perform main task : training, validation, and testing for each models, using **Typology Classifier.ipynb**
7. Evaluation of model performance, tested on other domains with **Domain Transfer.ipynb** , and evaluation of model outputs in recognizing building objects and its detailed features differentiating each typology class with **GradCAM Visualization.ipynb**
