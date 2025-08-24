# Application-of-machine-learning-deep-learning-algorithms-in-the-detection-of-depressive-tendencies
Research Background

With the widespread use of social media, users' online behavior and posted content contain a wealth of information, providing new perspectives for mental health research. This study aims to explore the use of social media data, particularly user behavior characteristics (such as the number of posts and follow relationships) and text content (through word embedding representations) to build machine learning models for the early identification and auxiliary diagnosis of depressive tendencies in users. This will help improve the efficiency and accessibility of mental health screening, provide technical support for early intervention, and thereby improve individual well-being and public health.

Data Source

The data is sourced from the Weibo user depression detection dataset collected and processed by Li Chenghao, Zhang Yilin, and others on GitHub
(WU3D). Data URL: https://github.com/aidenwang9867/Weibo-User-Depression-Detection-Dataset. The original files include two independent datasets stored in JSON format, where “depressed” represents users with depression and “normal” represents ordinary users. The user detail fields in the original dataset include, but are not limited to, nickname, gender, self-description, number of posts, follow relationships, and all posted tweets, which are relatively complex. During the data preprocessing stage, we will simplify the data.

Research approach

This study is divided into six sections: Section 1 covers data preprocessing; Section 2 employs traditional machine learning methods, including ensemble learning, for modeling and analysis; Section 3 uses deep learning methods to construct a Transformer model for modeling and analysis, and compares it with traditional models; Section 4 explores the importance of variables; the fifth part is an ablation experiment to explore the contribution of variables in different sections; the sixth part is hyperparameter tuning to obtain a better-performing model.
