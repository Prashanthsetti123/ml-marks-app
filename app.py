from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    prediction = model.predict([[hours]])

    return render_template("index.html",
                           prediction_text=f"Predicted Marks: {prediction[0]:.2f}",
                           hours=hours)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)