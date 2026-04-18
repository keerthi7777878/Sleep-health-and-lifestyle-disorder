

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# LOAD DATA
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

df = pd.read_excel("C:/Users/keerthi/Downloads/Expanded_Sleep_Dataset (1).xlsx")

print("Dataset Loaded Successfully\n")
print(df.head())

# BASIC INFO
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

# DATA CLEANING
df = df.dropna(how='all')
df.fillna(method='ffill', inplace=True)

# NUMERIC CONVERSION
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

numeric_cols = df.select_dtypes(include=np.number).columns

if len(numeric_cols) == 0:
    print("No numeric columns found!")
    exit()

if 'Sleep Duration' in numeric_cols:
    num_col = 'Sleep Duration'
else:
    num_col = numeric_cols[0]

print("\nUsing column:", num_col)

# HISTOGRAM
plt.figure()
sns.histplot(df[num_col], bins=15, kde=True, color='skyblue')
plt.title(f"Distribution of {num_col}", fontsize=16, fontweight='bold')
plt.xlabel(num_col)
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# BAR CHART
top10 = df[num_col].value_counts().head(10)

plt.figure()
sns.barplot(x=top10.index, y=top10.values, palette='viridis')
plt.title(f"Top Values of {num_col}", fontsize=16, fontweight='bold')
plt.xlabel(num_col)
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# PIE CHART
plt.figure()
colors = sns.color_palette('pastel')
plt.pie(top10.values, labels=top10.index, autopct='%1.1f%%', colors=colors)
plt.title("Distribution Share", fontsize=16, fontweight='bold')
plt.show()

# BOXPLOT
plt.figure()
sns.boxplot(x=df[num_col], color='lightgreen')
plt.title(f"Boxplot of {num_col}", fontsize=16, fontweight='bold')
plt.show()

# HEATMAP
numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] > 1:
    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix", fontsize=16, fontweight='bold')
    plt.show()

# SCATTER PLOT
plt.figure()
sns.scatterplot(x=df.index, y=df[num_col], color='purple')
plt.title("Scatter Plot", fontsize=16, fontweight='bold')
plt.xlabel("Index")
plt.ylabel(num_col)
plt.show()

# REGRESSION LINE
plt.figure()
sns.regplot(x=df.index, y=df[num_col],
            scatter_kws={'color': 'violet'},
            line_kws={'color': 'orange'})
plt.title("Regression Line", fontsize=16, fontweight='bold',color='purple')
plt.xlabel("Index")
plt.ylabel(num_col)
plt.show()

# CATEGORIZATION
def categorize(val):
    if val < df[num_col].quantile(0.33):
        return "Low"
    elif val < df[num_col].quantile(0.66):
        return "Medium"
    else:
        return "High"

df['Category'] = df[num_col].apply(categorize)

category_count = df['Category'].value_counts()

plt.figure()
sns.barplot(x=category_count.index, y=category_count.values, palette='Set2')
plt.title("Category Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# MACHINE LEARNING
X = df.index.values.reshape(-1, 1)
y = df[num_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))