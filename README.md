This project is dedicated to automatically detect od COVID-19 in the X-ray image dataset with the help of Keras,Tensorflow and deep learning. 

Since the COVID-19 became the world-wide problem I thought it could be the good idea to design the model for detecting the coronavirus. 

First thing first, we need to use the X-ray images and for this purpose I used the repo of  Dr. Cohen(He started to collect X-ray images and he is pushing these images in his github repo) https://github.com/ieee8023/covid-chestxray-dataset

In order to create the COVID-19 X-ray image dataset for this project, I:

    Parsed the metadata.csv
    file found in Dr. Cohen’s repository.
    Selected all rows that are:
        Positive for COVID-19 (i.e., ignoring MERS, SARS, and ARDS cases).
        Posterioranterior (PA) view of the lungs. I used the PA view as, to my knowledge, that was the view used for my “healthy” cases, as discussed below; however, I’m sure that a medical professional will be able clarify and correct me if I am incorrect (which I very well may be, this is just an example).

In total, that left me with 25 X-ray images of positive COVID-19 cases (Figure 2, left).

The next step was to sample X-ray images of healthy patients.

To do so, I used Kaggle’s Chest X-Ray Images (Pneumonia) dataset (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and sampled 25 X-ray images from healthy patients (Figure 2, right). There are a number of problems with Kaggle’s Chest X-Ray dataset, namely noisy/incorrect labels, but it served as a good enough starting point for this proof of concept COVID-19 detector.

After gathering my dataset, I was left with 50 total images, equally split with 25 images of COVID-19 positive X-rays and 25 images of healthy patient X-rays.

I’ve included my sample dataset in the “Downloads” section of this tutorial, so you do not have to recreate it.

The main python code for this project is train_covid19.py. and for executing this code you need also mention the dataset for the execution, therefore it will be python3 train_covid19.py --dataset dataset, besides here is two other command line arguments which could be used: 
    --plot
    : An optional path to an output training history plot. By default the plot is named plot.png
    unless otherwise specified via the command line.
    --model
    : The optional path to our output COVID-19 model; by default it will be named covid19.model
    .
For this project I split the dataset in 80/20 which is 80% of the data for training and 20% for the testing. In order to ensure that our model generalizes, we perform data augmentation by setting the random image rotation setting to 15 degrees clockwise or counterclockwise.


**Training our COVID-19 detector with Keras and TensorFlow**
With our train_covid19.py cript implemented, we are now ready to train our automatic COVID-19 detector over the dataset that we have. 
Once the train is done we could see the following results below: 
[INFO] evaluating network...
              precision    recall  f1-score   support
       covid       0.83      1.00      0.91         5
      normal       1.00      0.80      0.89         5
    accuracy                           0.90        10
   macro avg       0.92      0.90      0.90        10
weighted avg       0.92      0.90      0.90        10

acc: 0.9000
sensitivity: 1.0000
specificity: 0.8000


As you can see from the results above, our automatic COVID-19 detector is obtaining ~90-92% accuracy on our sample dataset based solely on X-ray images — no other data, including geographical location, population density, etc. was used to train this model.

We are also obtaining 100% sensitivity and 80% specificity implying that:
* of pients that do have COVID-19 (i.e., true positives), we could accurately identify them as “COVID-19 positive” 100% of the time using our model.
* Of patients that do not have COVID-19 (i.e., true negatives), we could accurately identify them as “COVID-19 negative” only 80% of the time using our model.

As our training history plot shows, our network is not overfitting, despite having very limited training data:
![plot](https://user-images.githubusercontent.com/23243761/77067526-5b651580-69e5-11ea-87e4-e0437b689d5f.png)

And last by not least here is the Final result from the x-ray which shows either the x-ray image has the potential of COVID-19 or not. 
![covid19_keras_dataset](https://user-images.githubusercontent.com/23243761/77067671-a5e69200-69e5-11ea-9d7d-3a841807255b.png)
