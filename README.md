# News Article Categorization with Deep Learning

## Introduction
This project aims to develop a model that can predict the category of a news article. The goal is to assist online news companies in categorizing their articles, enhancing user experience, and improving content discoverability.

## Dataset
The dataset used in this project is the public BBC News dataset, available on Kaggle as part of the BBC News Classification Competitions. It comprises 2225 news articles labeled into five categories: business, entertainment, politics, sport, and tech.

## Exploratory Data Analysis
An initial exploration of the dataset revealed that the 'sports' category has the highest number of articles, while 'tech' has the least. However, the number of articles across all categories is relatively balanced. The word count in each article varies significantly, indicating a diverse dataset.

## Model Development
Three initial models with similar architectures were developed to address this problem. However, these models were found to be overfitting, with the LSTM-based model achieving the highest accuracy.

## Model Improvement
The LSTM model was further refined by implementing several techniques, resulting in faster training times, higher accuracy, and more stability. However, with an accuracy of 67%, there is still room for improvement.

## Future Improvements
Here are some potential strategies for enhancing the model's performance:

1. **Increase Dataset Size**: Expanding the training dataset could lead to improvements in model performance, as more data can help the model learn more patterns.
2. **Data Augmentation**: As an alternative to increasing the dataset size, data augmentation techniques can be applied to create new training data from existing data. For text data, this could involve techniques like sentence shuffling.

[HuggingFace Deployment](https://huggingface.co/spaces/FarizFirdaus/NewsCategorization)
