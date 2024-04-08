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
    egfr = input_array[1]               # estimated glomerular filtration rate
    phosphate = input_array[2]          # phsophate (mg/dL)
    is_male = input_array[3]            # male sex
    upcr = input_array[4]               # urine protein creatinine ratio (mg/g)

    # Means
    mean_age_per_10 = 6.598546
    mean_egfr_per_5 = 3.144011
    mean_phosphate = 4.024117
    mean_uacr_log = 5.776576

    # Transform age from request before input to model:
    age_per_10 = (age / 10) - mean_age_per_10

    # Transform egfr from request before input to model:
    egfr_per_5 = (egfr / 5) - mean_egfr_per_5

    # Transform phosphate from request before input to model:
    phosphate = phosphate - mean_phosphate

    # uPCR (more commonly collected in the clinic), needs to be transformed into uACR (used by our model)
    # Convert uACR to log-scale (makes coefficient easier to interpret)
    uacr_log = (5.3920 +
        0.3072 * np.log(min(upcr / 50, 1)) +
        1.5793 * np.log(max(min(upcr / 500, 1), 0.1)) +
        1.1266 * np.log(max(upcr / 500, 1)))

    # Transform uACR before input to model:
    uacr_log = uacr_log - mean_uacr_log

    # Predicting dialysis and death at 1- and 2-year timeframes
    # Confidence intervals (using standard errors on the coefficient estimates because who can tell)
    risk_equation = lambda coefs, vars: np.exp(np.dot(coefs, vars))
    Fij  = lambda ri, F0: int(round((1 - (1 - F0)**(ri)) * 100, 0))

    dialysis_coefs = np.asarray([-0.3188, -0.3461, 0.1717, 0.1280, 0.2100])              # The order is assumed here!
    death_coefs = np.asarray([0.76670, 0.21410, -0.05462, 0.03436, -0.00040])            # The order is assumed here!

    dialysis_coefs_se = np.asarray([0.03677, 0.08201, 0.06202, 0.11640, 0.03814])        # The order is assumed here!
    death_coefs_se = np.asarray([0.06954, 0.08965, 0.09512, 0.14110, 0.03887])           # The order is assumed here!

    vars = np.asarray([age_per_10, egfr_per_5, phosphate, is_male, uacr_log])            # The order is assumed here!

    # Dialysis prediction (1- and 2-year)
    prob_dialysis1 = sorted([
        Fij(risk_equation(dialysis_coefs + dialysis_coefs_se, vars), 0.1548428), \
        Fij(risk_equation(dialysis_coefs, vars), 0.1548428), \
        Fij(risk_equation(dialysis_coefs - dialysis_coefs_se, vars), 0.1548428)
    ])

    prob_dialysis2 = sorted([
        Fij(risk_equation(dialysis_coefs + dialysis_coefs_se, vars), 0.2928511), \
        Fij(risk_equation(dialysis_coefs, vars), 0.2928511), \
        Fij(risk_equation(dialysis_coefs - dialysis_coefs_se, vars), 0.2928511)
    ])

    # Death prediction
    prob_death1 = sorted([
        Fij(risk_equation(death_coefs + death_coefs_se, vars), 0.047547503), \
        Fij(risk_equation(death_coefs, vars), 0.047547503), \
        Fij(risk_equation(death_coefs - death_coefs_se, vars), 0.047547503)
    ])

    prob_death2 = sorted([
        Fij(risk_equation(death_coefs + death_coefs_se, vars), 0.08898508), \
        Fij(risk_equation(death_coefs, vars), 0.08898508), \
        Fij(risk_equation(death_coefs - death_coefs_se, vars), 0.08898508)
    ])
        

    # Return it in the response however frontend wants it
    # Each prediction (prob_dialysis1, prob_dialysis2, prob_death1, prob_death2) has the format
    #   (lower_CI, main_prediction, upper_CI).
    # This should be displayed on the frontend as
    #   {{patient}}'s {{timeframe}}-year probability of {{death | dialysis}} is: {{main_prediction}} (95%CI: {{lower_CI}}, {{upper_CI}}) % 
    print(prob_dialysis1, prob_dialysis2)
    print(prob_death1, prob_death2)
    
    finalresponse = f"""
    Patient's 1-year probability of dialysis is: {prob_dialysis1[1]} (95%CI: {prob_dialysis1[0]}, {prob_dialysis1[2]})
    Patient's 2-year probability of dialysis is: {prob_dialysis2[1]} (95%CI: {prob_dialysis2[0]}, {prob_dialysis2[2]})
    Patient's 1-year probability of death is: {prob_death1[1]} (95%CI: {prob_death1[0]}, {prob_death1[2]})
    Patient's 2-year probability of death is: {prob_death2[1]} (95%CI: {prob_death2[0]}, {prob_death2[2]})
    """
    return jsonify({'message': finalresponse}), 200

if __name__ == '__main__':
    app.run()

