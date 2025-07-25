# Fake News Detection Using ML

# Introduction
In the modern digital era, fake news poses a significant threat, influencing public opinion and decision-making. The rapid spread of misinformation, fueled by social media and online platforms, necessitates robust tools for its detection. This project employs machine learning techniques to classify news articles as real or fake. Using algorithms like Support Vector Machine (SVM), Logistic Regression, Decision Tree, Random Forest, and Naive Bayes, the project evaluates their effectiveness and provides a comparative analysis based on performance metrics.

# Features
Automated detection of fake news using machine learning.
Comparison of multiple classifiers based on accuracy, precision, recall, and F1-score.
Visualizations of key trends using WordClouds and confusion matrices.

# Dataset
The dataset used for this project is sourced from Kaggle and consists of news articles labeled as real or fake. The dataset includes various features, such as:

Title: The title of the news article.
Text: The content of the article.
Label: A binary indicator (1 = Fake, 0 = Real).
Figure 6 illustrates the distribution of real and fake articles in the dataset.

# Libraries and Tools Used
The project uses the following Python libraries and tools:

nltk: Text preprocessing and tokenization.
matplotlib: Data visualization.
wordcloud: Generate word cloud visualizations.
seaborn: Advanced data visualization.
sklearn.model_selection: Train-test split and cross-validation.
sklearn.ensemble: Random Forest classifier.

Install the required dependencies using:
pip install nltk matplotlib wordcloud seaborn scikit-learn

# Machine Learning Models
The following algorithms are implemented and compared:

# Support Vector Machine (SVM)
Finds the hyperplane that best separates classes.
Visualized with its confusion matrix.

 ![Confusion Matrix for SVM](Image/SVM.png)
 
# Naive Bayes
Based on Bayes' theorem for probabilistic classification.

![Confusion Matrix for Navie Bayes](Image/Navie_Bayes.png)

# Logistic Regression
Models binary outcomes with a logistic function.

![Confusion Matrix for Logistic Regression](Image/Logistic_Regression.png)

# Decision Tree
Uses tree structures for decisions and classifications.

![Confusion Matrix for Decision tree](Image/decision_tree.png)

# Random Forest
An ensemble method combining multiple decision_trees.

![Confusion Matrix for Random Forest](Image/random_forest.png)

# Evaluation Metrics
Accuracy: Percentage of correctly classified instances.

Precision: Proportion of true positives among predicted positives.

Recall: Proportion of true positives among actual positives.

F1-Score: Harmonic mean of precision and recall.

# Results
The following table summarizes the performance of the classifiers:

![Results Table](Image/Table.png)

![Comparision of Machine Learning Algorithms](Image/Comparision.png)


