# Boston Housing Price Prediction

This project focuses on predicting the median value of owner-occupied homes in Boston using various housing and demographic features such as number of rooms, crime rate, and distance to employment centers.

## Dataset

The dataset used is the classic Boston Housing Dataset, commonly used in regression problems.

- Target Variable: `MEDV` – Median value of owner-occupied homes in $1000's.

---

## Models Applied

### 1. Simple Linear Regression  
- Features used: `RM` (Average number of rooms per dwelling)  
- Reason: `RM` had the strongest correlation with the target variable.  
- MSE: 28.92

### 2. Multiple Linear Regression  
- Features used: Top 3 features most correlated with `MEDV`:  
  - `RM`, `PTRATIO`, `LSTAT`  
- Reason: These features showed the strongest linear relationships with the target.  
- MSE: 13.82

### 3. Polynomial Regression  
- Features used: Same as in multiple linear regression (`RM`, `PTRATIO`, `LSTAT`)  
- Reason: Applied polynomial transformation to capture non-linear relationships between features and the target.  
- MSE: 9.29 (Best performance)

**Evaluation Metric:** Mean Squared Error (MSE)  
Lower MSE = better performance

---

## Data Preprocessing

- Dropped weakly correlated features  
- Handled missing values and duplicate rows  
- Treated outliers using the IQR method  
- Scaled numerical features using MinMaxScaler

---

## Dataset Features

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

## Conclusion

Among the three applied models, Polynomial Regression achieved the lowest MSE and provided the most accurate predictions. This suggests that the relationship between housing features and prices is not strictly linear and benefits from more complex, non-linear modeling.

---

## Note

The `MEDV` column represents the median (not mean) value of home prices, which is a more robust measure in the presence of outliers.