## EXNO-3-DS
NAME: V mythili 212223040123
DEPT: CSE
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
       ``
       import pandas as pd
df=pd.read_csv('Encoding Data (1).csv')
df
```

![image](https://github.com/user-attachments/assets/d927d4df-6da9-4854-ba6e-4c7bf9e13c81)

```
#ordinal encoder
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```


![image](https://github.com/user-attachments/assets/f3a70234-6282-41af-96f0-258cfc7fb395)


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```


![image](https://github.com/user-attachments/assets/573d9fec-93cd-4564-ab6b-347859598920)


```

#label encoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```


![image](https://github.com/user-attachments/assets/29dc6599-26a7-464c-a8e3-879e2ada2fb8)


```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)  # Use sparse_output instead of sparse
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2 = pd.concat([df2, enc], axis=1)
df2
```


![image](https://github.com/user-attachments/assets/47d343f7-e956-4440-b16e-c405079a137a)


```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)  # Use sparse_output instead of sparse
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2 = pd.concat([df2, enc], axis=1)
pd.get_dummies(df2,columns=["nom_0"])

```



![image](https://github.com/user-attachments/assets/6195e559-d82b-48b1-ba2e-8e7dc097b625)


```
pip install --upgrade category_encoders

```



![image](https://github.com/user-attachments/assets/7f781c26-afd5-4a63-bc6f-e6e40bc24c4a)


```
df=pd.read_csv("data.csv")
df
```


![image](https://github.com/user-attachments/assets/d17476cf-3d97-4c4a-862d-532340e9243a)



```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df["Ord_2"])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```


![image](https://github.com/user-attachments/assets/a7262015-ae74-4570-ad49-ed8ff53c986b)


```
from category_encoders import TargetEncoder
import pandas as pd
te = TargetEncoder()
cc = df.copy()
cc = pd.concat([cc, new], axis=1)
cc
```


![image](https://github.com/user-attachments/assets/80890c51-5bb9-4957-a5a1-dbc0513fc2fc)


```
# Feature transformation
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv("Data_to_Transform.csv")

df
```


![image](https://github.com/user-attachments/assets/b74949ce-63f7-4585-a132-cc3941674c84)


```
np.log(df["Highly Positive Skew"])
```



![image](https://github.com/user-attachments/assets/8b966f1f-0452-41a0-9f32-86d97251841a)


```
np.reciprocal(df["Moderate Positive Skew"])
```



![image](https://github.com/user-attachments/assets/c7a4270e-a132-431e-b760-0e82add5cdf1)


```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/e37881b5-8fb8-4db8-b8a7-37e83ae0acd6)


```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```


![image](https://github.com/user-attachments/assets/a4097680-9517-4ecc-a647-7d68e282c9e9)


```
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```


![image](https://github.com/user-attachments/assets/32ac5b39-1e01-4a15-a0f8-9abbefd5e267)


```
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```


![image](https://github.com/user-attachments/assets/6f24353e-23db-4fda-94f3-7670f4dea4a2)



```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal")
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```


![image](https://github.com/user-attachments/assets/2e351ecf-c989-42cb-913c-d9903463e2f5)



```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```

```
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```


![image](https://github.com/user-attachments/assets/ff5f74e1-44ab-4a13-9128-b16a6c50fa3b)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```


![image](https://github.com/user-attachments/assets/141f1c4c-e534-4adf-8245-8e72e7494938)


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```


![image](https://github.com/user-attachments/assets/9ca0eb47-2fde-489a-8123-7eee8b1841d0)



```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```



![image](https://github.com/user-attachments/assets/5bf8b9e5-9b7a-4b25-a436-f5ca16a5daf3)



```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```


![image](https://github.com/user-attachments/assets/26a1eb85-30d1-4b48-b146-39f57b875082)

```
#titanic dataset
dt=pd.read_csv("titanic_dataset.csv")
dt
```


![image](https://github.com/user-attachments/assets/0bbb48da-3bcd-47b8-9891-25f82bbdac47)


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt["Age"],line='45')
plt.show()
```



![image](https://github.com/user-attachments/assets/63317475-be95-4c02-994a-c5137c99cfb0)



```
sm.qqplot(dt["Age_1"],line='45')
plt.show()
```


![image](https://github.com/user-attachments/assets/48ef8e9e-6409-4c83-9619-f8d828c6b8de)



# RESULT:
       The feature encoding and transformation process been studied and excuted for the given data successfully.
       
