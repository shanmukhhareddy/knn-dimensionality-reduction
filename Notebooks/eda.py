# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder


# df=pd.read_csv("data/Breast_cancer.csv")
# print(df.shape)
# print(df.dtypes)

# # obj_cols = df.select_dtypes(include='object').columns
# # print(obj_cols)

# #print(df['Race'].nunique())
# #print(df.value_counts())



# obj=LabelEncoder()
# df['Status']=obj.fit_transform(df["Status"])
# df['A Stage']=obj.fit_transform(df["A Stage"])
# df['Estrogen Status']=obj.fit_transform(df["Estrogen Status"])
# df['Progesterone Status']=obj.fit_transform(df["Progesterone Status"])



# cols = ['Race', 'Marital Status','6th Stage', 'differentiate','Grade','T Stage ','N Stage']

# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# encoded = encoder.fit_transform(df[cols])
# encoded_df = pd.DataFrame(
#     encoded,
#     columns=encoder.get_feature_names_out(cols),
#     index=df.index
# )
# df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

# # for col in df.select_dtypes(include='object').columns:
# #     print(df[col].value_counts())

# target_col = 'Status'   # replace with your target column name

# cols = [c for c in df.columns if c != target_col] + [target_col]
# df = df[cols]


# print(df.info())

# # save DataFrame to CSV
# df.to_csv("data/processed_data.csv", index=False)

import pandas as pd 
from sklearn.datasets import make_classification


print(make_classification)

df=pd.DataFrame(make_classification)
print(df)