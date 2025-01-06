# Supervised Machine Learning: Classification & Regression Analysis  

## *Project Overview*  
This project explores the application of supervised machine learning algorithms to classify trade-related data. The primary objective is to assess the performance of different classification models, identify key features, and generate actionable business insights. Models like Logistic Regression, Decision Trees, and others are evaluated and compared using comprehensive performance metrics.  

---

## *Problem Statement*  
The project aims to classify trade transactions into specific categories based on various trade-related attributes. This classification seeks to reveal patterns and trends that can inform strategic business decisions. Key objectives include:  

- Leveraging supervised learning models for trade transaction classification.  
- Identifying significant features and establishing thresholds for effective classification.  
- Comparing model performances to select the most suitable approach for the dataset.  

---

## *Key Features*  

### *Classification Models*  
- Logistic Regression (LR)  
- Support Vector Machines (SVM)  
- Stochastic Gradient Descent (SGD)  
- Decision Trees (DT)  
- K-Nearest Neighbors (KNN)  
- Random Forest (RF)  
- Naive Bayes (NB)  
- XGBoost (Extreme Gradient Boosting)  

### *Performance Evaluation*  
- Accuracy, Precision, Recall, F1-Score, AUC  
- Cross-Validation (K-Fold) for robust evaluation  

### *Run Statistics*  
- Training time and memory usage across all models  

### *Feature Importance*  
- Identification of key features such as *Country, **Product, and **Value*  
- Analysis of thresholds and business implications for critical attributes  

---

## *Data Preprocessing*  
- *Data Cleaning:* The dataset contains no missing values, so no treatment is required.  
- *Encoding:* Categorical variables are encoded using ordinal encoding.  
- *Scaling:* Min-Max Scaling is applied to normalize numerical data.  

---

## *Dataset*  

- *Source:* Kaggle Import-Export Dataset  
- *Sample Size:* 5,001 rows sampled from 15,000 records  

### *Key Variables*  
- *Categorical:* Country, Product, Import_Export, Category, Port, Shipping_Method, Supplier, Customer, Payment_Terms  
- *Numerical:* Quantity, Value, Weight  
- *Index Variables:* Transaction_ID, Invoice_Number  

---

## *Technologies Used*  
- *Programming Language:* Python  
- *Libraries:* Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- *Environment:* Google Colab  
- *Version Control:* GitHub for tracking and collaboration  

---

## *Methodology*  

### *Data Preprocessing*  
- *Data Cleaning:* No missing data identified; no further action required.  
- *Encoding:* Categorical variables encoded using ordinal encoding.  
- *Scaling:* Numerical data scaled using Min-Max Scaling.  

### *Classification Models*  
- *Training and Testing Split:* 70% of the data is used for training, and 30% is reserved for testing.  
- *Comparison:* Models are compared based on performance on the test set and cross-validation results.  

### *Performance Metrics*  
- *Confusion Matrix*  
- *Precision, Recall, F1-Score, AUC*  
- *Training Time and Memory Usage* for runtime analysis  

### *Feature Analysis*  
- Significant features are identified using *Random Forest* and *Logistic Regression*.  
- Thresholds are derived for key attributes like *Value, **Weight, and **Shipping_Method*.  

---

## *Insights and Applications*  

### *Model Insights*  
- *Logistic Regression:* Demonstrates strong runtime efficiency and interpretability but faces challenges with class imbalance.  
- *Random Forest:* Offers robust insights into feature importance but is resource-intensive.  
- *SVM and SGD:* Perform well for dominant classes but require more balanced datasets for broader applicability.  

### *Business Applications*  
- *Targeted Marketing:* Utilize insights from key features and clusters to design personalized marketing campaigns.  
- *Inventory Management:* Optimize trade-specific strategies by distinguishing between bulk and lightweight goods.  
- *Supplier-Customer Relations:* Refine payment terms and shipping methods to improve operational efficiency.  

---

## *Contributors*  
- **Sejal Raj**
