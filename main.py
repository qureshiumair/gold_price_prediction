import numpy as np
import pandas as pd

df = pd.read_csv("gld_price_data.csv")
###data preprocessing###
df["day"] = df["Date"].apply(lambda x:x.split("/")[0]).astype(int)
df["month"] = df["Date"].apply(lambda x:x.split("/")[1]).astype(int)
df["year"] = df["Date"].apply(lambda x:x.split("/")[2]).astype(int)
df.drop("Date",axis=1,inplace=True)
###train_test_split###
labels = df["EUR/USD"]
df.drop("EUR/USD",inplace=True,axis=1)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(df,labels , test_size=0.2,random_state=42)
###model selection and training###
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
###model accuracy ###
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_test,y_pred)))
