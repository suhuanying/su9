import pandas as pd
pd.set_option('display.unicode.east_asian_width',True)
penguin_df =pd.read_csv('penguins-chinese.csv',encoding='gbk')
print(penguin_df.head())

import pandas as pd
pd.set_option('display.unicode.east_asian_width',True)
penguin_df= pd.read_csv('penguins-chinese.csv',encoding='gbk')
penguin_df.dropna(inplace=True)
output = penguin_df['企鹅的种类']
features= penguin_df[['企鹅栖息的岛屿','喙的长度','喙的深度','翅膀的长度','身体质量','性别']]
features=pd.get_dummies(features)
output_codes,output_uniques=pd.factorize(output)

print('下面是去重后，目标输出变量的数据：')
print(output_uniques)
print('下面是独热编码后，特征列的数据：')
print(features.head())

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle


penguin_df =pd.read_csv('penguins-chinese.csv',encoding='gbk')
penguin_df.dropna(inplace=True)
output = penguin_df['企鹅的种类']
features= penguin_df[['企鹅栖息的岛屿','喙的长度','喙的深度','翅膀的长度','身体质量','性别']]
features =pd.get_dummies(features)
output_codes, output_uniques =pd.factorize(output)


x_train,x_test,y_train,y_test =train_test_split(features,output_codes,train_size=0.8)
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)

score= accuracy_score(y_test,y_pred)

with open('rfc_model.pkl','wb') as f:
    pickle.dump(rfc,f)

with open('output_uniques.pkl','wb') as f:
    pickle.dump(output_uniques,f)
    
print('保存成功，已生成相关文件。')



import streamlit as st
import pickle
with open('output_uniques.pkl','rb') as f:
    output_uniques_maap=pickle.load(f)
st.subheader('随机森林模型')
st.write(rfc_model)

st.subheader('映射关系实例')
st.write(output_uniques_maps[1])


