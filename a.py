import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score

# -------------------------------
# LOAD & INITIAL CLEANING
# -------------------------------
# Using ';' and ',' as per European CSV format
df = pd.read_csv("AirQualityUCI.csv", sep=";", decimal=",")

# Drop empty unnamed columns
df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], errors='ignore')
df = df.dropna(how='all')

# Dataset uses -200 for missing values
df.replace(-200, np.nan, inplace=True)

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Drop NA to keep the structure similar to your reference code
df = df.dropna()

# Selecting important numerical features
features = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'NOx(GT)']
df = df[['Date', 'Time'] + features]

# -------------------------------
# SKEWNESS & HISTOGRAMS
# -------------------------------
for col in features:
    plt.figure()
    sns.histplot(df[col], kde=True, color='skyblue', edgecolor='black')
    
    plt.title(f"{col} Distribution (Skewness: {round(df[col].skew(), 2)})")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# -------------------------------
# BOXPLOTS & OUTLIERS (IQR)
# -------------------------------
df_original = df.copy()

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

df_clean = df.copy()
for col in features:
    temp = remove_outliers(df, col)
    df_clean = df_clean[df_clean.index.isin(temp.index)]

for col in features:
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_original[col], color='skyblue')
    plt.title(f"Before Outlier Removal\n{col}")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_clean[col], color='lightgreen')
    plt.title(f"After Outlier Removal\n{col}")
    
    plt.tight_layout()
    plt.show()

df = df_clean

# -------------------------------
# HEATMAP
# -------------------------------
numeric_df = df[features]
plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# -------------------------------
# OBJECTIVE 1: REGRESSION (CO vs Benzene)
# (In laptop it was RAM vs Price)
# -------------------------------
sns.regplot(x='CO(GT)', y='C6H6(GT)', data=df)
plt.title("Carbon Monoxide (CO) vs Benzene (C6H6)")
plt.show()

print("\nCorrelation:", df['CO(GT)'].corr(df['C6H6(GT)']))

X = df[['CO(GT)']]
y = df['C6H6(GT)']

slr_model = LinearRegression()
slr_model.fit(X, y)
slope = slr_model.coef_[0]
intercept = slr_model.intercept_

print(f"Equation: Benzene = {slope:.2f} * CO + {intercept:.2f}")

y_pred = slr_model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² Score: {r2:.4f}")

# Prediction Example (2.5 CO Level)
new_co = pd.DataFrame([[2.5]], columns=['CO(GT)'])
predicted_benzene = slr_model.predict(new_co)
print(f"Predicted Benzene for 2.5 CO: {predicted_benzene[0]:.2f}")

# Plotting Regression with Predicted point
sns.scatterplot(x=X['CO(GT)'], y=y, alpha=0.6, label="Actual Data")
plt.plot(X, slr_model.predict(X), color='red', label="Regression Line")
plt.scatter(new_co, predicted_benzene, color='green', s=100, label="Predicted Point")
plt.text(2.5, predicted_benzene[0], f"({2.5}, {predicted_benzene[0]:.1f})")
plt.xlabel("CO(GT)")
plt.ylabel("Benzene C6H6(GT)")
plt.title("Simple Linear Regression with Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -------------------------------
# OBJECTIVE 2: PIE CHART (CO Levels)
# (In laptop it was Types of Laptops)
# -------------------------------
conditions = [
    (df['CO(GT)'] <= 1.0),
    (df['CO(GT)'] > 1.0) & (df['CO(GT)'] <= 2.5),
    (df['CO(GT)'] > 2.5)
]
choices = ['Low (<1)', 'Medium (1-2.5)', 'High (>2.5)']
df['CO_Category'] = np.select(conditions, choices, default='Unknown')

type_counts = df['CO_Category'].value_counts()
plt.figure(figsize=(10, 7))

plt.pie(
    type_counts,
    labels=type_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops={'edgecolor': 'black'},
    pctdistance=0.8
)
plt.title("Distribution of CO Levels", fontsize=15, pad=20)
plt.tight_layout()
plt.show()

# -------------------------------
# OBJECTIVE 3: TOP COUNTS (Top Temperatures)
# (In laptop it was RAM configs)
# -------------------------------
df['T_rounded'] = df['T'].round().astype(int).astype(str) + " C"
df2 = df['T_rounded'].value_counts().head().reset_index()
df2.columns = ['Temperature', 'Count']

plt.figure(figsize=(8,5))
sns.barplot(data=df2, x='Count', y='Temperature', palette='magma')
plt.xlabel("Number of Occurrences")
plt.ylabel("Temperature")
plt.title("Top 5 Most Frequent Temperatures")

for i, v in enumerate(df2['Count']):
    plt.text(v + 2, i, str(v), va='center')

plt.subplots_adjust(left=0.2, right=0.9)
plt.show()

# -------------------------------
# OBJECTIVE 4: BAR GRAPH (Avg CO by Hour)
# (In laptop it was Laptop by Company)
# -------------------------------
df['Hour'] = df['Time'].str.split('.').str[0].astype(int)
hour_counts = df.groupby('Hour')['CO(GT)'].mean()

plt.figure(figsize=(12,6))
hour_counts.plot(kind='bar', color='teal')
plt.xlabel("Hour of Day")
plt.ylabel("Average CO Level")
plt.title("Average CO(GT) Level by Hour of Day")
plt.xticks(rotation=0)
plt.subplots_adjust(left=0.1, right=0.9)
plt.show()

# -------------------------------
# OBJECTIVE 5: HYPOTHESIS TESTING (T-TEST)
# (In laptop it was Low vs High RAM price. Here we do Day vs Night Temperature)
# -------------------------------
day_temps = df[(df['Hour'] >= 8) & (df['Hour'] <= 18)]['T']
night_temps = df[(df['Hour'] < 8) | (df['Hour'] > 18)]['T']
t_stat, p_value = ttest_ind(day_temps, night_temps, equal_var=False)

print("\n===== T-TEST RESULT =====")
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.4f}")
alpha = 0.05

if p_value < alpha:
    print("Conclusion: Reject Null Hypothesis (Significant Difference in Day vs Night Temperatures)")
else:
    print("Conclusion: Fail to Reject Null Hypothesis (No Significant Difference)")