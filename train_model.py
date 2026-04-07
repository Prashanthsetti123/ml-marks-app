import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset
data = {
    "hours": [1, 2, 3, 4, 5],
    "marks": [30, 40, 50, 60, 70]
}

df = pd.DataFrame(data)

# Train model
model = LinearRegression()
model.fit(df[["hours"]], df["marks"])

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")