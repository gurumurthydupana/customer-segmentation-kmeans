# Customer Segmentation using K-Means Clustering

## Project Overview
This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to segment mall customers into distinct groups based on their demographic and spending behavior. The objective is to identify meaningful customer segments that can help businesses design targeted marketing strategies, improve customer retention, and drive revenue growth.

The analysis uses three features:
- Age
- Annual Income (k$)
- Spending Score (1–100)

The optimal number of clusters is determined using the **Elbow Method**, and the resulting segments are visualized through 3D scatter plots and cluster distribution charts.

---

## Business Objective
Businesses often serve customers with different purchasing behaviors. By grouping customers with similar characteristics, organizations can:

- Identify high-value customers
- Design personalized promotions
- Improve retention strategies
- Reduce churn
- Optimize marketing spend

---

## Dataset Information
The project uses the **Mall Customers Dataset**, which contains 200 customer records with the following fields:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

### Feature Selection
The following numerical features are used for clustering:
- Age
- Annual Income (k$)
- Spending Score (1–100)

---

## Tools and Libraries Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

---

## Methodology

### 1. Data Loading
The dataset is loaded using Pandas.

### 2. Feature Selection
Selected the most relevant numerical features.

### 3. Data Standardization
Applied `StandardScaler` to normalize features before clustering.

### 4. Elbow Method
Calculated average distortion for values of K from 1 to 10 to determine the optimal number of clusters.

### 5. K-Means Clustering
Applied K-Means with `k = 5`.

### 6. Visualization
Generated:
- Elbow Method plot
- 3D cluster visualization
- Bar chart showing customers per cluster

---

## Project Structure

```text
customer-segmentation-kmeans/
│── README.md
│── requirements.txt
│── MBSA_FINAL_PROJECT.py
│── MBSA_DATASET_Mall_Customers.csv
│── MBSA_GROUP_14_REPORT.pdf
│
└── images/
    ├── elbow_method.png
    ├── customer_segments_3d.png
    └── customers_per_cluster.png
