# Application-of-machine-learning-deep-learning-algorithms-in-the-detection-of-depressive-tendencies

- [Research Background](#research-background)
- [Data Source](#data-source)
- [Research Approach](#research-approach)
- [Data Preprocessing](#data-preprocessing)
- [Traditional Machine Learning Modeling](#traditional-machine-learning-modeling)
- [Deep Learning Modeling](#deep-learning-modeling)
- [Feature Importance Exploration](#feature-importance-exploration)
- [Ablation Experiment](#ablation-experiment)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Summary and Reflection](#summary-and-reflection)

# Research Background

With the widespread use of social media, users' online behavior and posted content contain a wealth of information, providing new perspectives for mental health research. This study aims to explore the use of social media data, particularly user behavior characteristics (such as the number of posts and follow relationships) and text content (through word embedding representations) to build machine learning models for the early identification and auxiliary diagnosis of depressive tendencies in users. This will help improve the efficiency and accessibility of mental health screening, provide technical support for early intervention, and thereby improve individual well-being and public health.

# Data Source

The data is sourced from the Weibo user depression detection dataset collected and processed by Li Chenghao, Zhang Yilin, and others on GitHub
(WU3D). Data URL: https://github.com/aidenwang9867/Weibo-User-Depression-Detection-Dataset. The original files include two independent datasets stored in JSON format, where “depressed” represents users with depression and “normal” represents ordinary users. The user detail fields in the original dataset include, but are not limited to, nickname, gender, self-description, number of posts, follow relationships, and all posted tweets, which are relatively complex. During the data preprocessing stage, we will simplify the data.

# Research approach

This study is divided into six sections: Section 1 covers data preprocessing; Section 2 employs traditional machine learning methods, including ensemble learning, for modeling and analysis; Section 3 uses deep learning methods to construct a Transformer model for modeling and analysis, and compares it with traditional models; Section 4 explores the importance of variables; the fifth part is an ablation experiment to explore the contribution of variables in different sections; the sixth part is hyperparameter tuning to obtain a better-performing model.

# Data Preprocessing

Considering modeling convenience and runtime factors, the data was simplified, and only gender(gender), profile (profile), number of followers (num_of_follower), number of follows (num_of_following), total number of tweets (all_tweet_count), original tweet count (original_tweet_count), repost count (repost_tweet_count), and the content of the top 10 tweets (all_tweets_content). Then, label depressed users as 1 and normal users as 0, and merge and randomize the two datasets.
For numerical variables, after plotting histograms and box plots to observe the distribution of the data, it was found that all numerical variables show severe right skewness, so these variables were log-transformed. After transformation, the data distribution became normal and could be directly used for modeling;
For the binary gender variable, one-hot encoding was performed to generate two new columns: gender_female and gender_male;
For self-descriptive and promotional texts, first replace “⽆” and empty strings with “PAD,” then perform basic processing such as word segmentation, regularization, and stopword removal.
Traditional machine learning modeling requires text vectorization, which involves vectorizing the text. word embedding is chosen over TF-IDF. This is because even after truncating the number of posts, there are still many small units after word segmentation.Using TF-IDF would not only lead to a dimension explosion, but no word embedding method can capture the advantages of semantic relationships. The word embedding model selected is the official embedding model from Wikipedia, set to 300 dimensions.
The following is an overview of the processed data.


# Traditional machine learning modeling

This section divides the processed data into training and testing sets and defines a model evaluation and visualization function used to evaluate and select a suitable classification model to determine which model performs better in the classification task of predicting whether Weibo users are depressed. First, four baseline models were selected: logistic regression, naive Bayes classifier, decision tree, and support vector machine. Considering the potential impact of sample imbalance, after modeling each model using the original data, three resampling methods—oversampling, undersampling, and SMOTE—were also used for modeling. The performance of the four baseline models is summarized as follows (in fact, the resampling methods did not significantly improve model performance, demonstrating the modeling results of the original dataset):
