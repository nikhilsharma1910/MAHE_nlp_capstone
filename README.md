# Flu Detection from Social Media Interactions

## **Project Overview**
My capstone assignment from coursera asked me to identify flu cases from a social media so that it can be flagged to health department and authorities. This is the code to implement that use-case using NLP and machine learning techniques.

## 🚀 **Features**
- ✅ Text classification of social media-like posts  
- ✅ Logistic Regression and Random Forest models  
- ✅ Cross-validation and regularization for model optimization  
- ✅ Model persistence using `joblib` for real-time predictions  

## 📊 **Dataset**
The dataset consists of **2000** synthetic social media-like interactions with three columns:
- **Date:** The date when the post was made
- **Text:** The content of the social media post
- **Label:** `flu` or `non-flu` (classification target)

### Example:
```csv
Date,Text,Label
2024-11-12,"Feeling super tired with a sore throat and body aches.",flu
2024-11-14,"Had an amazing time at the party last night!",non-flu
```

## **Tech Stack**
- Python3.11.10
- pandas
- scikit-learn
- nltk
- joblib
- Machine Learning Models: Logistic Regression, Random Forest

## **Installation**
```bash
git clone https://github.com/nikhilsharma1910/MAHE_nlp_capstone.git
cd MAHE_nlp_capstone/
python3 -m venv env_project
source env_project/bin/activate
pip install -r requirements.txt
```

## **Generating the dataset**
```bash
python generate_dataset.py
```

## **Training the model**
```bash
python training_model.py
```

## **Testing the model**
This step will prompt you for a message/text entry and will provide an output
```bash
python test_model.py
```

## **Results of Model Training**
Due to synthetic data creation model seems to be biased towards some words.
```text
Cross-Validation Accuracy (Logistic Regression): 0.9985

Logistic Regression Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       204
           1       1.00      0.99      1.00       196

    accuracy                           1.00       400
   macro avg       1.00      1.00      1.00       400
weighted avg       1.00      1.00      1.00       400

Confusion Matrix:
 [[204   0]
 [  1 195]]
Accuracy: 0.9975
ROC-AUC Score: 1.0

Random Forest Classification Report:
              precision    recall  f1-score   support

           0       0.95      1.00      0.97       204
           1       1.00      0.94      0.97       196

    accuracy                           0.97       400
   macro avg       0.97      0.97      0.97       400
weighted avg       0.97      0.97      0.97       400

Confusion Matrix:
 [[204   0]
 [ 11 185]]
Accuracy: 0.9725
ROC-AUC Score: 0.9999749899959984
```

## **Results of the model**
It predicts and differentiates between flu and non flu symptoms well
![Alt text](running_example.png?raw=true "Title")

## **Future Improvements**
- Incorporate real-time social media data
- Use deep learning models (like LSTM) for advanced text analysis
- Handle ambiguous text data with more robust preprocessing

## **END NOTE**
This project is done as a project report for my college assignment. This is one of the getting started approaches of how to classify text and the real world usecases it represnets. With a good quality dataset this model can be intergated into several dashboards which can help authorities in real-time. Any suggestions and help is greatly appreciated.
