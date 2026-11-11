# 🏦 Bank Marketing Response Prediction

### Machine Learning | Customer Subscription Prediction | Streamlit

An end-to-end Machine Learning project that predicts whether a bank customer is likely to subscribe to a term deposit based on customer demographics, previous campaign interactions and socio-economic indicators.

The project combines data analysis, feature engineering, classification algorithms and an interactive Streamlit application to support data-driven bank marketing decisions.

## 🌐 Web Application

🚀 **[Launch Bank Marketing Prediction App](http://localhost:8501/)**

*Local application: Run `streamlit run app.py` to access the application at http://localhost:8501/.*

## 📌 Project Overview

Bank marketing campaigns involve contacting customers to promote term deposit subscriptions. Identifying customers who are more likely to subscribe can help financial institutions improve campaign targeting and resource allocation.

This project uses historical bank marketing data to build and evaluate machine learning classification models that predict customer subscription behaviour.

## 🎯 Objectives

- Analyse customer demographics and banking information.
- Explore campaign performance and economic indicators.
- Perform exploratory data analysis and feature engineering.
- Build and compare multiple classification algorithms.
- Evaluate models using classification metrics.
- Optimize model hyperparameters.
- Develop an interactive prediction application using Streamlit.

## 📂 Dataset

**Dataset:** Bank Marketing with Social and Economic Context

**Source:** UCI Machine Learning Repository

**Dataset link:** [https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

- Records: 41,188
- Features: 20 input attributes
- Target variable: `y`
- Prediction classes: Yes / No

The target variable indicates whether a customer subscribed to a bank term deposit.

The `duration` feature was excluded from predictive modelling because it is only known after a customer interaction and would cause data leakage for pre-call prediction.

## 🛠️ Technologies Used

| Category          | Technologies        |
| ----------------- | ------------------- |
| Programming       | Python              |
| Data Analysis     | Pandas, NumPy       |
| Visualization     | Matplotlib, Seaborn |
| Machine Learning  | Scikit-learn        |
| Model Persistence | Joblib              |
| Web Application   | Streamlit           |
| Development       | Jupyter Notebook    |
| Version Control   | Git, GitHub         |

## 🔍 Project Workflow

1. Data Collection and Loading
2. Data Cleaning and Inspection
3. Exploratory Data Analysis
4. Customer Demographic Analysis
5. Campaign Performance Analysis
6. Economic Indicator Analysis
7. Correlation Analysis
8. Feature Engineering
9. Data Preprocessing
10. Train-Test Split
11. Classification Model Development
12. Model Evaluation
13. Confusion Matrix and ROC Analysis
14. Feature Importance Analysis
15. Hyperparameter Tuning
16. Final Model Selection
17. Prediction Testing
18. Streamlit Application Development

## 🤖 Machine Learning Algorithms

The following algorithms were implemented and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

Random Forest was further optimized using GridSearchCV.

The final model was selected based on its F1-score.

## 📊 Model Performance

The selected model achieved the following evaluation results:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 88.52% |
| Precision | 49.17% |
| Recall    | 57.76% |
| F1-Score  | 53.12% |
| ROC-AUC   | 0.8084 |

**Selected Model:** Optimized Random Forest

These results are based on the held-out test dataset. The model demonstrates useful discrimination ability, although precision and recall indicate opportunities for further improvement.

## 🖥️ Streamlit Application

The interactive web application allows users to:

- Enter customer demographic information.
- Provide previous campaign details.
- Enter economic indicators.
- Generate subscription predictions.
- View estimated subscription probability.
- Explore model performance metrics.



## ⚙️ Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/LubnaShireenR/bank-marketing-response-prediction.git
```

**2. Navigate to the project directory**

```bash
cd bank-marketing-response-prediction
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

**5. Open the application**

```text
http://localhost:8501/
```

## 🔮 Future Improvements

- Improve minority-class prediction performance.
- Experiment with SMOTE and other imbalance-handling techniques.
- Perform probability threshold optimization.
- Add explainable AI using SHAP.
- Develop additional customer segmentation features.
- Deploy the application for public access.

## 👩‍💻 Author

**Lubna Shireen R**

B.Tech – Electronics and Communication Engineering  
Specialization: Data Science

GitHub: [https://github.com/LubnaShireenR](https://github.com/LubnaShireenR)

LinkedIn: [https://www.linkedin.com/in/lubnashireenr](https://www.linkedin.com/in/lubnashireenr)

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

Dataset provided by the UCI Machine Learning Repository.

Research reference:

Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*.

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback!
