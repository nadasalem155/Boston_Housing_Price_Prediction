# Boston Housing Price Prediction 🏠

This project focuses on **predicting the median value of owner-occupied homes in Boston** using various housing and demographic features such as number of rooms, crime rate, and distance to employment centers.  

🔗 **Live App:** [Boston Housing Price Prediction](https://bostonhousing-price-prediction.streamlit.app/)

---

## Dataset 📊  
- **Source:** [Boston Housing Dataset on Kaggle](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices) — a classic dataset widely used in regression problems.  
- **Target Variable:** `MEDV` – Median value of owner-occupied homes in $1000's.  

---

## Dataset Features 📝

| Column   | Description |
|----------|-------------|
| CRIM     | Per capita crime rate by town |
| ZN       | Proportion of residential land zoned for large lots |
| INDUS    | Proportion of non-retail business acres |
| CHAS     | Charles River dummy variable (1 if tract bounds river) |
| NOX      | Nitric oxide concentration (parts per 10 million) |
| RM       | Average number of rooms per dwelling |
| AGE      | % of owner-occupied units built before 1940 |
| DIS      | Weighted distance to employment centers |
| RAD      | Accessibility to radial highways |
| TAX      | Property tax rate per $10,000 |
| PTRATIO  | Pupil-teacher ratio by town |
| B        | 1000(Bk - 0.63)^2 (Bk = % Black population) |
| LSTAT    | % lower status of the population |
| MEDV     | Median value of owner-occupied homes (Target) |

---

## Data Preprocessing 🧹  
- Dropped weakly correlated features (**CHAS**, **B**).  
- Verified no missing values or duplicate rows.  
- Detected and removed outliers using the **IQR method**.  
- Applied **Min-Max scaling** to numerical features for normalization.  

---

## Models Applied & Performance 🤖

| Model                      | Features Used                | Reason for Selection                     | MSE   |
|----------------------------|------------------------------|------------------------------------------|-------|
| Simple Linear Regression   | RM                           | Strongest positive correlation with MEDV | 28.92 |
| Multiple Linear Regression | RM, PTRATIO, LSTAT           | Top 3 most correlated features           | 13.82 |
| Polynomial Regression      | RM, PTRATIO, LSTAT           | Capture non-linear relationships         | **9.29** |

**Evaluation Metric:** Mean Squared Error (MSE) — *Lower values indicate better performance.*  

---

## Conclusion 🏁  

- **Polynomial Regression achieved the lowest MSE (9.29)** and provided the most accurate predictions.  
- The relationship between housing features and prices is **not strictly linear** and benefits from **non-linear modeling**.  

---

💡 **Note:**  
`MEDV` represents the **median** (not mean) value of home prices, which is more robust in the presence of outliers.  

---

## Project Report 📄  
For a detailed explanation of the project steps, analysis, and results, see the [Full presentation](project-presentation.pdf).
