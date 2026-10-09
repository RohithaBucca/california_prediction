from flask import Flask, request, render_template
import joblib


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

obj = joblib.load("California.joblib")

model = obj["Model"]
columns = obj["columns"]


# --------------------------------------------------
# Create Flask application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        columns=columns
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.route("/predict")
def predict():

    input_data = []

    # Get values from HTML form
    for column in columns:

        value = request.args.get(
            column,
            type=float
        )

        input_data.append(value)


    # Make prediction
    prediction = model.predict([input_data])


    # Send prediction back to HTML
    return render_template(
        "index.html",
        columns=columns,
        prediction=round(prediction[0], 2)
    )


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )
