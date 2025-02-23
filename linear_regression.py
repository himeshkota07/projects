#!/usr/bin/env python
# coding: utf-8

# In[8]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import matplotlib.pyplot as plt

cars=pd.read_csv("C:/Users/himes/Machine Learning/Datasets/car_price/CarPrice_Assignment.csv")

display(cars)


# In[10]:


duplicate_count=sum(cars.duplicated())
#print(duplicate_count)
if duplicate_count>0:
    cars.drop_duplicates(inplace=True)
display(cars)


# In[11]:


unique_val_per_attri= cars.nunique()
single_value_columns=[i for i,value_count in enumerate(unique_val_per_attri) if value_count==1]
#print(single_value_columns)
if len(single_value_columns)>0:
    cars.drop(single_value_columns,axis=1,inplace=True)

print(cars.shape)


# In[12]:


train_set,test_set=train_test_split(cars,test_size=0.2,random_state=42)

cars_train=train_set.drop("price",axis=1)
cars_train_labels=train_set["price"].copy()


# In[17]:


imputer=SimpleImputer(strategy="median")
cars_train.head()

#cars_train_num=cars_train.drop(columns=["price"])
#imputer.fit(cars_train_num)
#cars_train_num_imputed=imputer.transform(cars_train_num)
display(cars_train_num_imputed)


# In[ ]:




