import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

st.title("Data Cleaning & Visualization Dashboard")

df = pd.read_csv("titanic.csv")

st.subheader("Original Dataset")
st.write(df.head())

st.subheader("Dataset Information")

st.write("Rows and Columns:",df.shape)

st.write("Column Names:")
st.write(df.columns)

st.subheader("Handling Missing Values")

missing_values = df.isnull().sum()

st.write("Missing Values Before Cleaning:")
st.write(missing_values)

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col].fillna(df[col].mean(),inplace=True)

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0],inplace=True)

st.success("Missing values handled successfully!")

st.subheader("Duplicate Records")

duplicate_rows = df[df.duplicated()]

duplicate_count = duplicate_rows.shape[0]

st.write("Total Duplicate Rows Found:",duplicate_count)

if duplicate_count > 0:
    st.write("Duplicate Records:")
    st.write(duplicate_rows)
df = df.drop_duplicates()

st.success("Duplicate records removed successfully!")

st.subheader("Outlier Detection and Removal")

numeric_columns = df.select_dtypes(include=np.number).columns

outlier_count = 0

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    outlier_count += outliers.shape[0]

    if outliers.shape[0] > 0:

        st.write("Outliers detected in column:", col)

        st.write(outliers[[col]])

    df = df[
        (df[col] >= lower_limit) &
        (df[col] <= upper_limit)
    ]

st.write("Total Outliers Removed:",outlier_count)

st.success("Outliers removed successfully!")

st.subheader("Cleaned Dataset")

st.write(df.head())

st.write("New Dataset Shape:",df.shape)

st.subheader("Passenger Class Distribution")

fig1, ax1 = plt.subplots(figsize=(8,5))

class_counts = df['Pclass'].value_counts()

class_counts.plot(kind='bar', ax=ax1)

ax1.set_title("Number of Passengers in Each Class")

ax1.set_xlabel("Passenger Class")

ax1.set_ylabel("Number of Passengers")

plt.xticks(rotation=0)

st.pyplot(fig1)
st.subheader("Age Distribution of Passengers")

fig2, ax2 = plt.subplots(figsize=(8,5))

df['Age'].plot(kind='hist', bins=20, ax=ax2)

ax2.set_title("Distribution of Passenger Ages")

ax2.set_xlabel("Age")

ax2.set_ylabel("Frequency")

st.pyplot(fig2)

st.subheader("Boxplot for Age Outliers")

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.boxplot(x=df['Age'],ax=ax3)

ax3.set_title("Boxplot of Passenger Ages")

ax3.set_xlabel("Age")

st.pyplot(fig3)

st.subheader("Correlation Heatmap")

correlation = df.corr(numeric_only=True)

fig4, ax4 = plt.subplots(figsize=(10,6))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    ax=ax4
)

ax4.set_title("Correlation Between Numerical Features")

st.pyplot(fig4)

st.subheader("Gender Distribution")

gender_counts = df['Sex'].value_counts()

fig5, ax5 = plt.subplots(figsize=(7,7))

ax5.pie(
    gender_counts,
    labels=gender_counts.index,
    autopct='%1.1f%%'
)

ax5.set_title("Male vs Female Passengers")

st.pyplot(fig5)

st.success("Data Cleaning & Visualization Project Completed Successfully!")