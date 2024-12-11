# Sentiment Analysis with Fine-Tuned RoBERTa

This project demonstrates how to fine-tune a pre-trained RoBERTa model on a custom sentiment analysis dataset using the Hugging Face Transformers library. The goal is to classify text into six emotion categories: sadness, joy, fear, anger, surprise, and love.

Project Structure

Dataset Files:

train.txt: Training data with text and labels.

val.txt: Validation data with text and labels.

test.txt: Test data with text and labels.

Outputs:


Evaluation metrics and visualizations.


The labels are mapped to integers as follows:

Label

Integer

sadness

0

joy

1

fear

2

anger

3

surprise

4

love

5



Results Interpretation

Accuracy: Overall correctness of the model.

F1 Score: Balance between precision and recall.

Precision: Percentage of correct positive predictions.

Recall: Percentage of actual positives identified correctly.

Confusion Matrix: Visual representation of classification errors.

Link to the trained model on Google file: https://drive.google.com/drive/folders/1eWB3KeA26sXCcDg-VemX5n7YJ2cAcHYV?usp=sharing
Dataset used: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?resource=download
