from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
data = {
    "Experience" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary" : [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000]
}

df = pd.DataFrame(data)

x = df[["Experience"]]
y = df[["Salary"]]

model = LinearRegression()
model.fit(x,y)
joblib.dump(model, "model.pkl")

print("Model Saved Successfully")