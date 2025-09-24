import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import kagglehub

# Download latest version
'''path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")

print("Path to dataset files:", path)
'''


file_path = "C:/Users/Admin/PycharmProjects/fake news/datasett.csv"

df1 = pd.read_csv(file_path)


print(df1.head(7))
print(df1.tail(7))

print("info",df1.info())
print("shape",df1.shape)
print("describe",df1.describe())

print("null val",df1.isnull().sum())

print("data at specified index",df1.iloc[[1,15,20]])

print("dtyes",df1.dtypes)

print("cls",df1.columns)

print("mem_usage",df1.memory_usage().sum())

print(df1['text'])

sns.countplot(x='label', data=df1)
plt.show()

sns.kdeplot(df1['label'])
plt.show()