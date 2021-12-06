#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
A=np.array([1,0.2,0.5,0.2,1,0.8,0.5,0.8,1])
matrix=A.reshape(3,3)
print(matrix)


# In[10]:


#TRANSPOSE
print(A.transpose())


# In[97]:


#A=np.eye(5,5)
#print(A)


# In[82]:


#DETERMINANT OF A
det=np.linalg.det(A)
print('The determinant of A is:',det)
print('\n')


# In[89]:


#QN2
#DATAFRAME
import pandas as pd
df =pd.DataFrame({'A':[1,0.2,0.5],
                 'B':[0.2,1,0.8],
                 'C':[0.5,0.8,1]})
df.head()


# In[14]:


#QN3
C=np.array([[1,3,1,2,9,4,5,6,10,4]])
print(C)


# In[15]:


#standard deviation
C.std()


# In[18]:


#QN4
x=3.14
f_string="The value of x is 3.14"
print(f_string)


# In[54]:


#QN5
df=pd.read_csv(r'C:\Users\mbabazi\Desktop\DataScience1\COVID-19 Cases.csv')
df.head()


# In[45]:


#setting Date as an index
df.set_index('Date', inplace=True)
df.head()


# In[69]:


#QN6
df_results = df[(df.Difference >0) & (df.Case_Type == 'Confirmed') & (df.Country_Region == 'Italy')]
df_results


# In[98]:


#QN7
df_results['Difference']
df_results.hist()


# In[99]:


import seaborn as sns
df_results.Difference.value_counts().plot(kind = 'hist')


# In[100]:


#QN8
df_results.describe()


# In[65]:


#QN9
df.boxplot(by='Country_Region', column=['Difference'], grid=False)


# In[117]:


df_results['Difference'].plot(kind="box");


# In[66]:


#QN10
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[86]:


df_results_1[(df_results_1.Country_Region=='Italy')&(df_results_1.Case_Type=='Deaths')]


# In[85]:


df_results_2[(df_results_2.Country_Region=='Germany')&(df_results_2.Case_Type=='Confirmed')]


# In[109]:


#QN12
df_results_2.plot(kind='scatter', x='Country_Region', y='Cases')
plt.xlabel('Country_Region')
plt.ylabel('Cases')
plt.title('A scatter plot for Cases vs Country_Region')
plt.show()


# In[ ]:




