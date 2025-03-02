# Late Delivery Risk Analysis 

## 1. Project Overview
In the world of logistics and e-commerce, on-time delivery is crucial for maintaining customer satisfaction and operational efficiency. This project explores the factors contributing to late deliveries, using statistical analysis and machine learning to uncover key insights.

By leveraging SOTA data science techniques, we identify the most influential factors affecting late deliveries, helping businesses optimize their supply chain and mitigate risks.

## 2. Key Objectives

- Identify critical factors influencing late deliveries through group comparison analysis using data visualizations, and statistical tests.
- Apply tree-based feature importance and selection to identify the more predictive variables.
- Use causal inference to examine how factors can influence lateness.​
- Provide actionable insights for improving delivery performance

## 3. Why This Matters?

Timely delivery is a competitive advantage. By understanding the root causes of late shipments, companies can reduce costs, improve efficiency, and enhance customer trust.

## In the initial phase of our project, we focused on data cleaning and preprocessing to ensure the dataset's quality and suitability for analysis. The key steps undertaken include:​

## 4. Data Science Workflow 

### Data Cleaning and Preprocessing
  
In this initial phase of the project, we first started by loading the dataset, specifying the correct format (ISO-8859-1). Then we checked for missing values, drop the columns with highest missing values and verified if the rest of the variables with missing values were significant. Then we removed the columns that were not necessary for the model like Customer name, Customer email, Customer password, the IDs from Customer, Orders and Products and others.

Since we have columns with date information but their data type is 'object', we needed to convert it fo datetime data type.

### EDA

We decided to divide our EDA in two phases General and Group Comparison (Late Delivery Risk).

#### General

While doing the Correlation Matrix for Numerical Features​, we removed “Product Status” since it has no values, all inputs are zeros. Also, we identify highly correlated features with a threshold of 0.8, and extract feature pairs with high correlation.​ To avoid multicollinearity, remove one of the features from the pairs and keep the features that are relevant to the use case.​ We removed features these  “Order Item Product Price”, “Order Year”, “Sales”, “Order Month”, “Sales per Customer”, and “Benefit per Order”.

Additionally, we were working with analyzing the causes of delay, where shipping information is prioritized when compared to the order information. In real life, we assume that the products are being ordered and shipped in the same year, unless it is ordered in the last week of the year, where it will shipped at the beginning of the following year. Therefore, to avoid multicollinearity, we removed the 'order year' column. The same applies for month, we will remove the 'order month' column.

Moreover, the 'sales' and 'order item total' columns were correlated and both has the same values. Therefore, we removed the 'sales' column. It also applies for 'sales per customer' and 'order item total'. Since we were not analyzing from the business side, we removed the 'sales per customer' column. The same applies for 'order item profit ratio' and 'benefit per order', we removed the 'benefit per order' column. 

By doing the above we finalized with a cleaned version of a correlation matrix.

About the outliers, we detect the numerical features in the dataset containing any outliers by using boxplots to visualize and applying the IQR method.​ Boxplots shown are results after outlier removal. The majority of the outliers were removed, and further removal will be performed when the model performances are not meeting expectations.




