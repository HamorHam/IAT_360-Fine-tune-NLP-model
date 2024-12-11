# Sentiment Analysis with Fine-Tuned RoBERTa

This project demonstrates how to fine-tune a pretrained RoBERTa model on a custom sentiment analysis dataset using the Hugging Face Transformers library. The goal is to classify text into six emotion categories: sadness, joy, fear, anger, surprise, and love.

Project Structure

Dataset Files:

train.txt: Training data with text and labels.

val.txt: Validation data with text and labels.

test.txt: Test data with text and labels.

Code Files:

train_model.py: Script to fine-tune the RoBERTa model.

evaluate_model.py: Script to evaluate the model and display metrics, including a confusion matrix.

utils.py: Helper functions for data processing and metric computation.

Outputs:

Trained model files in ./tweet_eval_finetuned.

Evaluation metrics and visualizations.
Dataset Format

Each dataset file (train.txt, val.txt, test.txt) contains lines in the following format:

<text>;<label>

Example:

I feel so happy today;joy
I am feeling anxious about the future;fear
This is the worst day ever;sadness

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

Training the Model

Run the training script to fine-tune the RoBERTa model:

python train_model.py

Key Hyperparameters:

Batch Size: Default is 16.

Learning Rate: Default is 2e-5.

Number of Epochs: Default is 3.

These parameters can be modified in the train_model.py file.

Evaluating the Model

To evaluate the model and print metrics, including accuracy, F1 score, precision, recall, and confusion matrix, run:

python evaluate_model.py

Example Metrics Output:

Evaluation Metrics:
Accuracy: 0.8942
F1 Score: 0.8924
Precision: 0.9001
Recall: 0.8900

A confusion matrix visualization will also be displayed, highlighting the classification performance for each label.

Making Predictions

Use the trained model to make predictions on new text data:

from transformers import pipeline

model_path = "./tweet_eval_finetuned"
sentiment_pipeline = pipeline("text-classification", model=model_path)

texts = ["I am so excited!", "This is terrible."]
predictions = sentiment_pipeline(texts)
for text, pred in zip(texts, predictions):
    print(f"Text: {text}\nPrediction: {pred}\n")

Results Interpretation

Accuracy: Overall correctness of the model.

F1 Score: Balance between precision and recall.

Precision: Percentage of correct positive predictions.

Recall: Percentage of actual positives identified correctly.

Confusion Matrix: Visual representation of classification errors.
