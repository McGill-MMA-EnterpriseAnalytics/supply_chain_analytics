# Analyzing Product Delays & ​Lateness Risk in ​Supply Chain​

Team Members: 
Yina Liang Li – Business Analyst – yina6202​

Liliana Garcia Caraballo – Data Analyst – lilianapgarciac​

Mirza Abubacker – Machine Learning Engineer - ThisIsMirk​

Kevin Wang – Data Scientist - Kevin-Wang-McGill​

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

In the initial phase of our project, we focused on data cleaning and preprocessing to ensure the dataset's quality and suitability for analysis. The key steps undertaken include:​

## 4. Data Science Workflow 

### Data Cleaning and Preprocessing
#### (See Data Cleaning-Preprocessing.ipynb file)
  
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

#### Group Comparison (Late Delivery Risk)

We focused on understanding the distribution of key features across the two groups: on-time deliveries (Late Delivery Risk = 0) vs. late deliveries (Late Delivery Risk = 1). By analyzing categorical and numerical variables separately, we aimed to identify patterns and potential factors influencing delivery performance. For categorical variables, we conducted chi-square tests to evaluate whether there were significant differences in distributions between the two groups. 

The results showed strong associations between shipping mode, order status, and delivery status with late deliveries, suggesting that these factors play a crucial role in predicting delivery delays. Some variables, like customer country and market, showed no statistically significant relationship, indicating that delivery risk might be more dependent on logistics factors rather than geographic location.

For numerical variables, we used violin plots and box plots to compare distributions between the two groups. This helped us visualize whether specific features, such as order quantity, shipping cost, or product weight, differed significantly between late and on-time deliveries. Additionally, we analyzed correlations between numerical variables and delivery risk, identifying features that might contribute to late deliveries. The insights from this comparison provide an initial idea for feature selection and predictive modeling, ensuring that our analysis is data-driven and backed by statistical evidence.

After doing the EDA, we concluded the following hypothesis:

Hypothesis 1: Does the shipping mode (vs. first class or same day) increases the probability of late delivery? 
Hypothesis 2: Does the payment type affects late delivery risk because of differences in order processing time?
Hypothesis 3: Are orders from certain regions are causally more likely to experience late delivery due to logistical or infrastructural constraints?

With these, the next step was to apply Causal Inference to determine whether these factors actually cause late deliveries or not. 

### Causal Inference 
#### (See Causal Inference.ipynb. Please note that the Casual Inference is based on the EDA part, so you can see that the analysis made on the DataCleaning-Preprocessing.ipynb file it is also included in the Causal Inference.ipynb file)

This project applies causal inference methods to investigate key factors affecting order delivery delays, with a special focus on the causal impact of shipping modes on delay risk. Through rigorous causal analysis, we discovered results that contradicted our initial hypotheses, providing data-driven support for logistics decision-making.

#### Main Research Hypotheses

We investigated three main hypotheses: first, exploring how shipping mode (standard vs. non-standard) affects the probability of late delivery; second, analyzing whether payment method influences delay risk; and finally, examining whether orders from specific regions (United States) are more likely to experience delivery delays.

#### Key Findings

**Impact of Shipping Mode on Delays**

Our analysis shows that the delay risk for the standard shipping group is only 38.0%, while the non-standard shipping group has a delay risk as high as 79.6%, with a risk difference of -41.7 percentage points. This indicates that standard shipping actually significantly reduces delay risk, contrary to the direction of our initial hypothesis. The Individual Treatment Effects (ITE) distribution graph further confirms this finding, with most orders showing negative treatment effects, demonstrating that standard shipping generally reduces delay risk.

**Feature Importance Analysis**

Feature importance analysis using the permutation method revealed the key factors influencing the relationship between shipping mode and delay risk. Scheduled shipping days is the most significant factor, with importance approaching 1.0; economic factors (such as order profit, order total, and discount) have minimal influence, with importance between 0.1-0.2; order quantity has almost no impact on delay risk, with importance of only 0.05.

**SHAP Value Analysis Insights**

SHAP value analysis allowed us to understand more deeply how each feature influences model predictions. Scheduled shipping days showed an interesting pattern: standard shipping works best for orders with moderate delivery times, while it may actually increase risk for very short delivery times. Economic and quantity features are distributed relatively evenly on both sides of the Y-axis, indicating their positive and negative influences cancel each other out, with minimal impact on the final result. This aligns with our previous feature importance analysis.

**Other Hypothesis Validation**

Our validation of payment method and regional factors indicates that these factors have no significant causal impact on delay risk. Most effect values cluster around zero, indicating that even though the United States represents our largest order volume country, U.S. orders show no significant differences in delay risk. This further reinforces our previous finding: scheduled delivery time is the main factor affecting delay risk, not payment method or regional differences.

#### Causal Inference Conclusion
Our research found that scheduled shipping time is the main factor affecting delay risk, rather than payment method or regional differences. This suggests we should rethink our shipping strategy, focusing on the time requirements of orders rather than other factors. This finding not only challenges our initial hypothesis but also provides valuable data support for optimizing logistics decisions.

### Modeling

**Logistic Regression and Random Forest**

Before getting started on the modelling, a feature importance was run using random forest to determine the most important features in our dataset in relation to late delivery risk. This will significantly help us improve model performance as without it, we were looking at around 1,500 features. 
Since we were running a logistic regression model, the numerical columns had to be scaled and StandardScaler() was used to this job. For the categorical columns OneHotEncoding() was used as there were no ordinal columns in the dataset. Now, the data was fit for model training, testing, and tuning. 

_LogisticRegression_RandomForest.ipynb_

Logistic regression and Random Forest models were tested on a cross validation set using both top 50 features and top 20 features (20 because it was a good elbow cut-off range when looking at the feature importance plot). Top 20 features provided better results, with random forest providing great results in CV. 

From here, the random forest model was chosen to go through hyperparameter tuning. Since the dataset was so large (180,000 observations), GridSearchCV took a long time to run (~30 min). So we decided to use RandomizedSearchCV to conduct our hyperparameter tuning and got the parameters we were looking for and it took about 8 minutes to run. 

The model was then trained on the whole training set and then tested on the test set, resulting in 89% accuracy, precision, and recall. The ROC curve and the confusion matrix further confirms excellent model performance. 

**ANN**

We deployed an Artificial Neural Network (ANN) with a three-layer architecture (128-64-32 neurons) to analyze delivery risk patterns. The model used a dropout rate of 0.2 and batch size of 64, converging quickly during training. The final test accuracy of 70.42%. Our feature selection process highlighted that "Days for shipment (scheduled)" and shipping mode were consistently the most influential variables across modeling approaches, aligning with our causal inference findings.

## 5. Conclusion

We can conclude that Standard shipping reduces lateness risk (by 41.7 percentage points) compared to non-standard options.​ Also, First-Class and Second-Class shipping have higher lateness risk than Standard or Same-Day options.​ The most influential factor is the scheduled shipping time, indicating that delivery timing policies have a stronger impact on delays than other logistical or economic factors. This suggests that Standard shipping should be promoted to customers to balance efficiency and cost.​

Additionally, very short delivery times increase lateness risk due to unrealistic scheduling and medium-length delivery windows (3-4 days) offer the lowest risk of delays.​ With this, internal order scheduling can be improved to align with realistic delivery lead times.​

While initially hypothesized as potential risk factors, causal inference showed that payment type and customer region had no meaningful impact on late deliveries. 

Through Random Forest feature selection and predictive modeling, we achieved 89% accuracy in predicting late deliveries, confirming that key features like shipping mode, scheduled delivery time, and order status are critical predictors. The ANN model also validated these insights, reinforcing the idea that decisions related to logistics are more relevant to mitigate lateness risk on deliveries.
