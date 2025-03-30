# Employee_Data_Analysis
1. Problem Statement: Enhancing Employee Promotion Prediction for Strategic Human Resource Management
In contemporary organizations, effective Human Resource Management plays a pivotal role in fostering a dynamic and thriving work environment. One critical aspect of HR strategy is the judicious identification of employees deserving of promotions. However, the manual assessment of promotion potential is often time-consuming and subject to biases. To address these challenges, there is an imperative need to leverage advanced data analytics and machine learning techniques to develop a predictive model that enhances the accuracy and objectivity of employee promotion predictions.
Challenges:
1.	Subjectivity and Bias: Traditional promotion decision-making processes may be influenced by subjective judgments, potentially introducing bias and hindering diversity and inclusion efforts.
2.	Resource Intensiveness: Manual evaluation of numerous factors, including performance ratings, training history, and length of service, poses a resource-intensive challenge for HR departments.
3.	Data-Driven Insights: The current absence of a robust, data-driven approach limits the ability to uncover nuanced patterns and correlations that contribute to promotion decisions.




1.1 Objective:
The primary objective of this project is to develop a predictive model that accurately identifies employees with a high likelihood of promotion. By harnessing the power of machine learning algorithms, we aim to create a tool that not only enhances the efficiency of the promotion process but also ensures fairness and transparency.
1.2 Proposed Solution:
Utilizing historical employee data, encompassing features such as performance ratings, training records, and demographic information, we will design and train machine learning models. Logistic Regression, Decision Tree Classifier, and Random Forest Classifier will be explored to identify the most effective algorithm for predicting promotions.
1.3 Expected Outcomes:
1.	Increased Accuracy: The implementation of a machine learning model is anticipated to outperform traditional methods, yielding a higher accuracy rate in predicting employee promotions.
2.	Transparency and Fairness: By relying on data-driven insights, the model is expected to reduce subjective biases, promoting transparency and fairness in promotion decisions.
3.	Resource Efficiency: The automated prediction process will streamline HR workflows, enabling more efficient allocation of resources to strategic HR initiatives.





1.4 Significance:
This project aligns with the organization's commitment to fostering a data-driven culture, promoting employee development, and ensuring equitable career advancement opportunities. The successful implementation of an advanced predictive model for employee promotions holds the potential to revolutionize HR practices and contribute to the overall success and resilience of the organization in an increasingly competitive landscape.














2.Detailed Description of the project
2.1 Proposed work with tools and datasets used 
Dataset: Kaggle
1. Data Overview
The dataset consists of the following columns:
•	employee_id: Unique identifier for each employee.
•	department: The department in which the employee works (e.g., Sales & Marketing, Operations, Technology).
•	region: The region in which the employee is located.
•	education: Educational qualification of the employee.
•	gender: Gender of the employee (e.g., Male, Female).
•	recruitment_channel: The channel through which the employee was recruited (e.g., sourcing, other).
•	no_of_trainings: Number of training programs attended by the employee.
•	age: Age of the employee.
•	previous_year_rating: Rating received by the employee in the previous year.
•	length_of_service: Number of years the employee has been in service.
•	awards_won: Whether the employee has won any awards (1 for Yes, 0 for No).
•	avg_training_score: Average training score of the employee.
•	is_promoted: Promotion status (1 for Yes, 0 for No).


2.2 Methodology And Work Flow
•	Data Preprocessing
•	Handle missing values.
•	Encode categorical variables.
•	Standardize or normalize numerical features.
•	Feature Importance
•	Identify key features influencing promotion using feature importance analysis.
•	Model Selection and Training
•	Select a suitable machine learning model for promotion prediction (e.g., logistic regression, decision tree, random forest).
•	Split the data into training and testing sets.
•	Train the model on the training set.
•	Model Evaluation
•	Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1 score.
•	Visualize the ROC curve and calculate the AUC.






2.3 Distribution of Categorical Variables
To gain insights into the distribution of categorical variables, pie charts were employed to visualize the composition of each feature. The code snippet below generates a 3x3 grid of pie charts, each representing the distribution of values for a specific feature.
The generated pie charts offer a visual representation of the proportion of each category within the selected features. This aids in understanding the diversity and spread of categorical variables, contributing to a more comprehensive analysis of the dataset. Adjustments to the layout and spacing have been made to ensure clarity in visualization.
To gain insights into the distribution of numerical variables, histograms with kernel density estimates (KDE) were utilized. The code snippet below creates a grid of subplots, each containing a histogram for a specific numerical feature.
Histograms provide a visual representation of the frequency distribution of numerical variables, offering insights into the central tendency and spread of the data. Kernel density estimates overlaid on the histograms provide a smoothed representation of the probability density function. This aids in identifying patterns and understanding the shape of the distribution for each numerical feature. The layout is organized in a grid for ease of comparison between different features.

To assess the normality of numerical variables, probability plots were generated using the provided code snippet. This process involves two visualizations: a histogram and a probability plot.


The normality check is crucial for statistical analyses that assume a normal distribution. Observations from the probability plot can guide decisions on whether transformations or non-parametric methods might be necessary for variables that deviate significantly from normality.

Log Transformation of Numerical Features
To address potential skewness in numerical features, a log transformation was applied to selected variables. The transformed features include 'age_log', 'length_of_service_log', and 'avg_training_score_log'.
The log transformation is a technique to stabilize variance and make the data more closely approximate a normal distribution. This is particularly beneficial when working with statistical models that assume normality. The histograms illustrate the impact of the log transformation on the distribution of each feature.

Correlation Analysis of Numerical Features
To understand the relationships between numerical variables, a correlation heatmap was constructed. The heatmap visualizes the correlation coefficients between different numerical features.
The heatmap provides the following insights:

Color Gradient: The color gradient represents the strength and direction of the correlation. Darker shades indicate stronger correlations, and the color varies based on the direction (positive or negative).

Annotation Values: Numerical values within each cell denote the correlation coefficient. Positive values imply a positive correlation, negative values suggest a negative correlation, and values close to zero indicate weak or no correlation.

Interpretation: Understanding the correlation between variables is crucial for identifying potential multicollinearity or dependencies in the dataset. High positive/negative correlations may suggest relationships that could impact model performance or require further investigation.

This heatmap aids in identifying patterns and dependencies within the numerical features, contributing to a comprehensive understanding of the dataset's structure. It is a valuable tool for feature selection, model building, and decision-making in subsequent analytical stages.

2.4 Predictive Modeling and Evaluation
1.Decision Tree Classifier
To predict employee promotions, a Decision Tree Classifier model was trained on the provided dataset.
 
2. Logistic Regression
Model Overview:
Logistic Regression is a linear model that is widely used for binary classification tasks. It estimates the probability that a given instance belongs to a particular category. In the context of predicting employee promotions, Logistic Regression can provide interpretable coefficients and is computationally efficient.

Model Performance:
After training and evaluating the Logistic Regression model on the provided dataset, it achieved the highest accuracy among the models considered in this analysis.
 

3 Random Forest Classifier
Model Overview:
The Random Forest Classifier is an ensemble learning method that builds multiple decision trees and merges their predictions. It is known for its robustness and ability to handle complex relationships in data. Random Forests can capture non-linear patterns and are less prone to overfitting.

Model Performance:
The Random Forest Classifier was also trained and evaluated on the dataset. While it demonstrated competitive performance, the accuracy achieved by Logistic Regression was marginally superior.
 
4.K-means Clustering 
The K-means Clustering was also done  and evaluated on the dataset.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class K_Means:
    def __init__(self, k=5, tolerance=0.001, max_iter=500):
        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tolerance

    def euclidean_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def fit(self, data):
        self.centroids = {}

        self.centroids[0] = np.array([25])   # Cluster for average training score less than 50
        self.centroids[1] = np.array([55])   # Cluster for average training score less than 60
        self.centroids[2] = np.array([65])   # Cluster for average training score less than 70
        self.centroids[3] = np.array([75])   # Cluster for average training score less than 80
        self.centroids[4] = np.array([85])   # Cluster for average training score less than 90

        for i in range(self.max_iterations):
            self.classes = {}
            for j in range(self.k):
                self.classes[j] = []

            for point in data:
                distances = []
                for index in self.centroids:
                    distances.append(self.euclidean_distance(point, self.centroids[index]))
                cluster_index = distances.index(min(distances))
                self.classes[cluster_index].append(point)

            previous = dict(self.centroids)
            for cluster_index in self.classes:
                if len(self.classes[cluster_index]) > 0:
                    self.centroids[cluster_index] = np.average(self.classes[cluster_index], axis=0)
                else:

                    pass

            isOptimal = True

            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            if isOptimal:
                break

k_means = K_Means(k=5)

k_means.fit(df["avg_training_score"].values.reshape(-1, 1))
df['cluster_avg_training_score'] = [k_means.predict(np.array([[avg_training_score]])) for avg_training_score in df["avg_training_score"]]

cmap = ListedColormap(['blue', 'green', 'orange', 'red', 'purple'])

plt.scatter(df["avg_training_score"], df['cluster_avg_training_score'], c=df['cluster_avg_training_score'], cmap=cmap)
plt.xlabel('Average Training Score')
plt.ylabel('Cluster')
plt.title('Clustering of Employees by Average Training Score')
plt.yticks(np.arange(5), ['< 50', '< 60', '< 70', '< 80', '< 90'])
plt.ylim(-0.5, 4.5)
plt.colorbar(ticks=[0, 1, 2, 3, 4], label='Average Training Score Clusters', format=plt.FuncFormatter(lambda val, loc: ['< 50', '< 60', '< 70', '< 80', '< 90'][val]))
plt.show()


2.5 Comparative Analysis
The performance of each model was assessed using a combination of traditional classification metrics, such as accuracy, precision, recall, and F1 score. Additionally, precision-recall curves were analyzed to understand the trade-off between precision and recall.





3.IMPLEMENTATION
3.1 Program Code:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
%matplotlib inline

df=pd.read_csv("employee.csv")
df.head()
df.info()
df.isnull().sum()
df.dropna(axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)
df.drop(columns=['employee_id'],inplace=True)
num_features=[i for i in df.columns if df.dtypes[i]!='object']
num_features
cat_features=[i for i in df.columns if df.dtypes[i]=='object' ]
print(cat_features)
df[cat_features].nunique()
for i in num_features:
    plt.figure(figsize=(10, 5))

    sns.boxplot(y=df[i])
    print(i)
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd

# Create subplots with a 3x3 grid and larger figsize
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))

# Loop through the first 9 columns of your DataFrame
for i, feature in enumerate(df.columns[:9]):
    row = i // 3
    col = i % 3

    # Plot a pie chart for the distribution of values for each feature
    pd.value_counts(df[feature]).plot.pie(autopct="%.1f%%", ax=axs[row][col],
                                          radius=0.9)  # Adjust the radius
    axs[row][col].set_title(feature)  # Set title for the subplot

# Set a super title for the entire figure
plt.suptitle("Distribution of features", fontsize=20)



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a subplot grid
num_rows = (len(num_features) - 1) // 4 + 1
fig, axs = plt.subplots(num_rows, 4, figsize=(16, 4 * num_rows))

for i, column in enumerate(num_features):
    row = i // 4
    col = i % 4
    sns.histplot(df[column], kde=True, ax=axs[row][col])

# Set a super title for the entire figure
plt.suptitle("Distribution of features")


plt.show()


import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab

def plot_curve(df, feature):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df[feature].hist()

    plt.subplot(1, 2, 2)
    if df[feature].dtype in [int, float]:
        stats.probplot(df[feature], dist='norm', plot=pylab)
    else:
        print(f"Probability plot cannot be generated for non-numerical data in feature: {feature}")

    plt.show()




age_log = np.log1p(df['age'])
service_log = np.log1p(df['length_of_service'])
score_log = np.log1p(df['avg_training_score'])

df.insert(6, 'age_log', age_log)
df.insert(9, 'length_of_service_log', service_log)
df.insert(12, 'avg_training_score_log', score_log)

df.head()
log_columns = ['age_log', 'length_of_service_log', 'avg_training_score_log']

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

sns.histplot(df['age_log'], ax=ax1,kde="False")
ax1.set_title('Distribution of age_log')
sns.histplot(df['length_of_service_log'], ax=ax2,kde="False")
ax2.set_title('Distribution of length_of_service_log')
sns.histplot(df['avg_training_score_log'], ax=ax3,kde="False")
ax3.set_title('Distribution of avg_training_score_log')

plt.suptitle('Distribution of log converted features', fontweight='bold')
plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt


numeric_columns = df.select_dtypes(include=['number'])

# Creating a correlation heatmap
plt.figure(figsize=(10, 8))
plt.title('Correlation of features')
sns.heatmap(numeric_columns.corr(), annot=True, linewidths=.5, cmap="YlGnBu")
plt.show()



from sklearn.preprocessing import StandardScaler
features = np.array(df_encoded.columns).reshape(-1, 1)

for feature in features:
    scaler = StandardScaler()
    scaler.fit(df_encoded[feature])
    df_encoded[feature] = scaler.transform(df_encoded[feature])

df_encoded.head()



x = df_encoded.drop(columns=['is_promoted'], inplace=False)
y = df_encoded['is_promoted'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)#80:20 ratio
print('Shape of X_train: ', X_train.shape)
print('Shape of X_test: ', X_test.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of y_test: ', y_test.shape)


def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="macro")
    recall = recall_score(y_test, pred, average="macro")
    f1 = f1_score(y_test, pred, average="macro")

    print('Confusion Matrix')
    print(confusion)
    print('Accuracy: {0:.4f}, Precision: {1:.4f}, Recall {2:.4f}, F1: {3:.4f}'.format(accuracy, precision, recall, f1))


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
dt_pred_proba = dt_clf.predict_proba(X_test)[:, 1]


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Assuming you have a trained classifier 'dt_clf' and test data 'X_test', 'y_test'
y_scores = dt_clf.predict_proba(X_test)[:, 1]  # Get the probability scores for the positive class

# Specify the positive label if it's not 1
precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=3)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()


lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
lr_pred_proba = lr_clf.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, lr_pred, lr_pred_proba)
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Assuming you have a trained classifier 'dt_clf' and test data 'X_test', 'y_test'
y_scores = lr_clf.predict_proba(X_test)[:, 1]  # Get the probability scores for the positive class

# Specify the positive label if it's not 1
precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=3)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Assuming you have a trained classifier 'dt_clf' and test data 'X_test', 'y_test'
y_scores = rf_clf.predict_proba(X_test)[:, 1]  # Get the probability scores for the positive class

# Specify the positive label if it's not 1
precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=3)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()


3.2 RESULT ANALYSIS
3.2.1 Graphs:
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 

4. Conclusion
While both Logistic Regression and the Random Forest Classifier displayed commendable performance, Logistic Regression emerged as the model with the highest accuracy in predicting employee promotions on this dataset. The choice between models may depend on specific business requirements, interpretability, and computational considerations.

The comprehensive evaluation and comparison of these models provide valuable insights into their strengths and weaknesses. The findings guide decision-makers in selecting an appropriate model for deployment based on the organization's goals and constraints.



