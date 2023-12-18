---
layout: post
title:  "Machine Learning vs. Deep Learning on Tabular Data"
author: Jake Whitworth
description: "Comparison of different ML methods against a Neural Network with Uncertainty Calibration on tabular data."
image: /assets/images/ml_dl.png
---

# Modern Machine Learning

In this age of AI, transformers and other neural network architectures are all the rage. ChatGPT acted as a catalyst to the hype around large language models (LLMs), image generation models, and other complex systems. When it comes to these complex AI systems that we see today, almost all of them are built using complex deep learning architectures. Deep learning is a sub-category of machine learning, and uses multiple layers of neural networks to learn from data.

![Figure](https://raw.githubusercontent.com/jdubindaclub/my386blog/main/assets/images/ai_ml_dl.png)

Deep learning has proved to be a powerful tool in AI, but it has its limitations. One such limitation is that deep learning has historically performed worse on tabular data when compared to traditional machine learning models. I aim to address that limitation in this project.

# My Background with Tabular Data

This last summer, I spent my time working at a quantitative hedge fund. We used deep learning models to recommend trades to our portfolio manager. There was a team for multimodal deep learning (models that use multiple modalilties of data. For example, a model that uses text, images, and tabular data), but I was assigned to the tabular models. Our team's philosophy was that deep learning could outperform traditional machine learning models no matter what modality of data we were using. The stock market is the closest thing that I know of to a chaotic system, and their belief in deep learning came from the idea that we needed a more complex model to capture the complexity of the market.

# Deep Learning on Tabular Data

Our tabular data that we used at the hedge fund had thousands of variables and millions of rows. This is ideal when trying to capture complex relationships using deep learning. This, however, is rarely the case with tabular data. Deep learning models are data hungry, and are often too complex for small datasets. I aim to test how well deep learning performs on tabular data when compared to traditional machine learning models.

# Uncertainty Calibration

One of the advantages of deep learning models is that they can output uncertainty. This is useful in many applications, such as self-driving cars. If a self-driving car is unsure about what it is seeing, it can output a high uncertainty score. This uncertainty score can then be used to determine if the car should stop or not. This is a very useful feature, but it is not unique to deep learning models. There are many traditional machine learning models that can output uncertainty scores. At the hedge fund, uncertainty calibration was the method that I saw had the biggest impact on the models. I aim to use this method to improve the performance of a simple neural network on tabular data.



# About the Data

![Figure](https://raw.githubusercontent.com/jdubindaclub/my386blog/main/assets/images/smoking_variables.png)



# Data Exploration

I started my exploration by using one of my favorite libraries: YData Profiling. This library quickly and automatically generates descriptive statistics, data type information, missing value analysis, and distribution visualizations for each column in your dataset. This library also offers advanced features like correlation analysis, unique value counts, and data quality assessments. YData was able to generate the dashboard in 1 minute and 11 seconds. Using the YData report I was able to see that there were 26 rows of duplicates, and no missing values. 

Instead of having to set up a loop to show me heatmaps, correlation plots between variables, and other visualizations, YData has all of those plots readily available. This is a great tool for data exploration, and I highly recommend it. To find out more about this library, you can visit their <a href="https://docs.profiling.ydata.ai/latest/"> website. </a> 

We can see in the YData report that multiple columns such as waistline, HDL_chole, LDL_chole, and triglyceride have significant outliers, and are heavily skewed. 




# Model Selection

I am going to use two traditional machine learning models, and two neural network-based models. The traditional machine learning models that I will be using are XGBoost (tree model) and K Nearest Neighbors. The neural network models that I will be using are a simple neural network, and a neural network with uncertainty calibration.

## ML Models

- XGBoost:
XGBoost is a gradient boosted tree model. XGBoost has become very popular over the last few years for its performance on tabular data. XGBoost goes one step beyond gradient boosting, as it gets its nickname comes from being an "extreme" gradient boosted tree model. Because XGBoost is not our focus in this project, I will not go into detail about how it works. If you would like to learn more about XGBoost, you can read the <a href="https://xgboost.readthedocs.io/en/latest/"> XGBoost documentation. </a>
To perform the hyperparameter tuning for the traditional ML models, I first created a pipeline for each model, then used GridSearchCV to find the best hyperparameters. The code I used to create the pipelines and perform the hyperparameter tuning is as follows:
