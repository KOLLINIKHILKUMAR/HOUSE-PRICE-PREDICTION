from flask import Flask, render_template, request
import numpy as np
import pickle


model = pickle.load(open(r'rf_reg.pkl','rb'))

app = Flask(__name__)

@app.route("/", methods=['POST','GET'])
def home():
    ans=""
    if request.method=='POST':
        input_dict = request.form.to_dict()
        input_values = input_dict.values()
        input_values = list(map(float, list(input_values)))
        input_values = np.array(input_values)
        input_values = input_values.reshape(1,-1)
        prediction = model.predict(input_values)[0]
        ans="THE PREDICTED PRICE IS "+str("{:.2f}".format((prediction)))+"K dollars($)"
    return render_template('index.html', answer=ans)


if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0')