import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For better visualization
plt.style.use("ggplot")

# ----------------------------------------------------------
# 2. Load Dataset
# ----------------------------------------------------------

# Dataset file (change path if required)
df = pd.read_csv("Dataset.xls") # File is actually CSV format

print("First 5 Records")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------------------------------------
# 3. Data Cleaning
# ----------------------------------------------------------

# Fill missing salary with median salary
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

# Fill missing college with 'Unknown'
df["College"] = df["College"].fillna("Unknown")

# Remove rows with missing Age, Height, Weight
df.dropna(subset=["Age", "Height", "Weight"], inplace=True)

# Convert Height from feet-inches format (e.g., 6-2) to inches

def convert_height(height):
    try:
        feet, inches = height.split('-')
        return int(feet) * 12 + int(inches)
    except:
        return np.nan

df["Height_inches"] = df["Height"].apply(convert_height)

# Ensure Salary is numeric
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

# Remove remaining missing values
df.dropna(inplace=True)

print("\nData Types After Cleaning:")
print(df.dtypes)

# ----------------------------------------------------------
# 4. Height & Weight Analysis
# ----------------------------------------------------------

print("\n========== HEIGHT & WEIGHT ANALYSIS ==========")

avg_height = df["Height_inches"].mean()
avg_weight = df["Weight"].mean()

print("Average Height (inches):", round(avg_height, 2))
print("Average Weight:", round(avg_weight, 2))

# Tallest Player
tallest = df.loc[df["Height_inches"].idxmax()]
print("\nTallest Player:")
print(tallest[["Name", "Team", "Position", "Height"]])

# Heaviest Player
heaviest = df.loc[df["Weight"].idxmax()]
print("\nHeaviest Player:")
print(heaviest[["Name", "Team", "Position", "Weight"]])

# Position-wise comparison
position_stats = df.groupby("Position")[["Height_inches", "Weight"]].mean()

print("\nAverage Height & Weight by Position:")
print(position_stats)

# ----------------------------------------------------------
# 5. Salary Correlation Analysis
# ----------------------------------------------------------

print("\n========== CORRELATION ANALYSIS ==========")

corr_age = df["Salary"].corr(df["Age"])
corr_height = df["Salary"].corr(df["Height_inches"])
corr_weight = df["Salary"].corr(df["Weight"])

print("Salary vs Age Correlation:", round(corr_age, 3))
print("Salary vs Height Correlation:", round(corr_height, 3))
print("Salary vs Weight Correlation:", round(corr_weight, 3))

# Correlation Matrix
corr_matrix = df[["Salary", "Age", "Height_inches", "Weight"]].corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

# ----------------------------------------------------------
# 6. Outlier Detection (Top 1.5 IQR Rule)
# ----------------------------------------------------------

print("\n========== OUTLIER ANALYSIS ==========")

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR

salary_outliers = df[df["Salary"] > upper_limit]

print("\nPlayers with Extremely High Salaries:")
print(salary_outliers[["Name", "Team", "Position", "Age", "Salary"]])

print("\nOutliers by Team:")
print(salary_outliers["Team"].value_counts())

print("\nOutliers by Position:")
print(salary_outliers["Position"].value_counts())

# ----------------------------------------------------------
# 7. Position-wise Physical Comparison
# ----------------------------------------------------------

print("\n========== POSITION-WISE COMPARISON ==========")

position_physical = df.groupby("Position")[["Height_inches", "Weight"]].mean()

print(position_physical)

# ----------------------------------------------------------
# 8. Top 10 Highest Paid Players
# ----------------------------------------------------------

print("\n========== TOP 10 HIGHEST PAID PLAYERS ==========")

top10 = df.nlargest(10, "Salary")

print(top10[["Name", "Team", "Position", "Age",
             "Height", "Weight", "Salary"]])

print("\nMost Common Teams:")
print(top10["Team"].value_counts())

print("\nMost Common Positions:")
print(top10["Position"].value_counts())

print("\nAverage Age of Top Earners:",
      round(top10["Age"].mean(), 2))

# ==========================================================
# VISUALIZATIONS
# ==========================================================

# ----------------------------------------------------------
# Scatter Plot : Salary vs Age
# ----------------------------------------------------------
plt.figure(figsize=(8,5))
plt.scatter(df["Age"], df["Salary"])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Salary vs Age")
plt.show()

# ----------------------------------------------------------
# Scatter Plot : Salary vs Height
# ----------------------------------------------------------
plt.figure(figsize=(8,5))
plt.scatter(df["Height_inches"], df["Salary"])
plt.xlabel("Height (Inches)")
plt.ylabel("Salary")
plt.title("Salary vs Height")
plt.show()

# ----------------------------------------------------------
# Box Plot : Salary by Position
# ----------------------------------------------------------
plt.figure(figsize=(8,6))

positions = sorted(df["Position"].unique())
salary_data = [df[df["Position"] == pos]["Salary"]
               for pos in positions]

plt.boxplot(salary_data, labels=positions)

plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Salary Distribution by Position")
plt.show()

# ----------------------------------------------------------
# Heatmap : Correlation Matrix
# ----------------------------------------------------------
plt.figure(figsize=(6,5))

plt.imshow(corr_matrix, cmap='coolwarm')

plt.colorbar()

plt.xticks(range(len(corr_matrix.columns)),
           corr_matrix.columns,
           rotation=45)

plt.yticks(range(len(corr_matrix.columns)),
           corr_matrix.columns)

# Add correlation values on heatmap
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i,
                 round(corr_matrix.iloc[i, j], 2),
                 ha='center',
                 va='center',
                 color='black')

plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

print("\nAnalysis Completed Successfully!")
