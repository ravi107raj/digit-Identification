## Digit Identification using Random Forest
This repository contains code for training and evaluating a random forest classifier on the MNIST dataset for digit identification. The code utilizes scikit-learn, a popular machine learning library in Python, and demonstrates the use of grid search for hyperparameter tuning.

# Overview
The Digit Identification using Random Forest repository provides an implementation of a random forest classifier to classify digits in the MNIST dataset. The goal is to accurately identify handwritten digits (0-9) based on their grayscale images.

The code uses scikit-learn's RandomForestClassifier for training the model and performs grid search to find the optimal hyperparameters. It splits the dataset into training and testing sets, trains the model on the training data, evaluates its performance on the testing data, and displays the confusion matrix and accuracy score.

This repository serves as a learning resource and provides a starting point for building and evaluating machine learning models on the MNIST dataset. It showcases best practices for data preprocessing, model training, hyperparameter tuning, and evaluation. By exploring this code, you can gain insights into the application of random forest classifiers and grid search in solving classification tasks.

The code is well-documented and includes comments explaining the purpose and functionality of each section. It can serve as a reference for understanding the implementation details and can be easily modified or extended for experimentation and further research.

Feel free to explore the code, run it on your own machine, and customize it to fit your specific requirements. Enjoy learning and experimenting with the MNIST dataset and random forest classifiers!

# Results
The output of the script will include the following information:

![image](https://github.com/abhigyan02/Digit-Identification/assets/75851981/968d9890-d7cf-47a8-9ea1-9f0a6bc11469)

![image](https://github.com/abhigyan02/Digit-Identification/assets/75851981/30d97430-8354-43c1-b744-bdcc6d8c930c)

![image](https://github.com/abhigyan02/Digit-Identification/assets/75851981/9ec138ae-db15-40dc-ac13-0eacc1331bf2)

![image](https://github.com/abhigyan02/Digit-Identification/assets/75851981/7b3216bf-5c6a-451c-885b-cc458cb3fe66)

![image](https://github.com/abhigyan02/Digit-Identification/assets/75851981/87f7d956-bb50-424e-910b-fb01f9b2383f)

Optimal hyperparameters selected by grid search (number of estimators, maximum depth, and minimum samples leaf).
Confusion matrix: A matrix showing the number of true positive, true negative, false positive, and false negative predictions for each class.
Accuracy score: The accuracy of the model on the test set.
Additionally, the script will display the predicted class for the last image in the dataset along with the image itself.
