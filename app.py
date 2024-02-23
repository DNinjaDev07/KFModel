from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=['GET'])
def main():
	return {'message': 'Kidney Failure Prediction'}

# Reserved method incase we need to use JSON inputs with JSON tags
# def validate_json(data):
#     # Check if request contains JSON data
#     if not data:
#         return jsonify({"error": "Request must contain JSON data"}), 400

#     # Check if 'age', 'female', 'egfr', and 'upcr' are present in the JSON data
#     if 'age' not in data or 'female' not in data or 'egfr' not in data or 'upcr' not in data:
#         return jsonify({"error": "JSON data must contain 'age', 'female', 'egfr', and 'upcr'"}), 400

#     # Check if 'age' is not null or empty
#     if not data['age']:
#         return jsonify({"error": "'age' cannot be null or empty"}), 400

#     # Check if 'female' is not null or empty
#     if 'female' in data and data['female'] is None:
#         return jsonify({"error": "'female' cannot be null"}), 400

#     # Check if 'egfr' is not null or empty
#     if not data['egfr']:
#         return jsonify({"error": "'egfr' cannot be null or empty"}), 400

#     # Check if 'upcr' is not null or empty
#     if not data['upcr']:
#         return jsonify({"error": "'upcr' cannot be null or empty"}), 400

@app.route('/predict', methods=['POST'])
def predict_kidney_failure():
    
    # Get JSON data and do basic checks on input request 
    data = request.get_json()

  #  validation_result = validate_json(data)

    # # If validation fails, return the error message
    # if validation_result:
    #     return validation_result
    
    # Check if 'input' key exists in JSON data
    if 'input' not in data:
        return jsonify({"error": "JSON data must contain 'input' array"}), 400

    # Get the 'input' array from the JSON data
    input_array = data['input']

    age = input_array[0]
    female = input_array[1]
    egfr = input_array[2] # estimated glomerular filtration rate
    upcr = input_array[3] #urine protein creatinine ratio 

    mean_age = 6.598546
    mean_log_acr = 5.776576
    mean_egfr = 3.144011

    #Transform age from request before input to model:
    age = (age / 10) - mean_age

    # Transform egfr from request before input to model:
    egfr = (egfr / 5) - mean_egfr

    # Transform log_acr from request before input to model:
    acr = (5.3920 +
               0.3072 * np.log(min(upcr/ 50, 1)) +
               1.5793 * np.log(max(min(upcr/ 500, 1), 0.1)) +
               1.1266 * np.log(max(upcr/ 500, 1)))
    
    log_acr = np.log(acr)  - mean_log_acr
    
    prob_1year = 1 - (0.81210 ** (np.exp(((-0.22598) * age) + ((-0.18364) * female) + ((0.32050) * log_acr) + ((-0.46961) * egfr))))

    prob_1year = prob_1year *100
    #prob_1year = 1 - {0.81210 ^ [((-0.22598 * age) + (-0.18364 * female) + (0.32050 * log_acr) + (-0.46961 * egfr))]}
    return jsonify({'message': 'The probability of kidney failure in 1 year is '+str(prob_1year)}), 200

if __name__ == '__main__':
    app.run()

