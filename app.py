import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


from Earthquake_prediction.pipelines.Prediction_Pipeline import CustomData, PredictPipeline
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, static_folder="static")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Support both form and JSON
    if request.is_json:
        req_data = request.get_json()
        latitude = float(req_data.get('latitude'))
        longitude = float(req_data.get('longitude'))
        depth = float(req_data.get('depth'))
        mag = float(req_data.get('mag'))
        hour = float(req_data.get('hour'))
    else:
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))
        depth = float(request.form.get('depth'))
        mag = float(request.form.get('mag'))
        hour = float(request.form.get('hour'))

    data = CustomData(
        latitude=latitude,
        longitude=longitude,
        depth=depth,
        mag=mag,
        hour=hour
    )
    final_result = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(final_result)
    result = float(round(pred[0], 2))

    # If AJAX/JS fetch, return JSON; else render result.html
    if request.is_json:
        return jsonify({"prediction": result})
    else:
        return render_template("result.html", final_result=result)

# execution begin
if __name__ == "__main__":
    print("Open your browser and go to: http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=True)