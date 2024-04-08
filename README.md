Kidney Failure Prediction Model.
It predicts the patient's risk of kidney failure for 1 and 2 years.
It also predicts the patient's risk of death in 1 and 2 years.

The code should be compiled and run on an application and the endpoint "/predict" should be called.

http verb: POST
{"input":[90, 15, 3.4, 0, 300]}- Sample 1

where:

age = input[0] (in Sample 1,it is 90)
egfr = input[1] (in Sample 1,it is 15)
phosphate = input[2] (in Sample 1,it is 3.4)
is_male = input[3] (in Sample 1,it is 0 – 0 means female, 1 means male)
upcr = input[4] (in Sample 1, it is 300)

Sample Response:

{
    "message": "\n    Patient's 1-year probability of dialysis is: 16 (95%CI: 13, 20)\n    Patient's 2-year probability of dialysis is: 31 (95%CI: 26, 36)\n    Patient's 1-year probability of death is: 20 (95%CI: 15, 28)\n    Patient's 2-year probability of death is: 35 (95%CI: 26, 46)\n    "
}
