from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

app = Flask(__name__)
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

# @app.route("/home")
# def home():
#     return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/result",methods=["POST","GET"])  
def result():
    list_col=['item_weight','item_fat_content','item_visibility','item_type','item_mrp',
    'outlet_establishment_year','outlet_size','outlet_location_type',
    'outlet_type'] 
    # print('start')
    item_weight = float(request.form['item_weight'])
    item_fat_content = int(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = int(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = int(request.form['outlet_establishment_year'])
    outlet_size = int(request.form['outlet_size'])
    outlet_location_type = int(request.form['outlet_location_type'])
    outlet_type = int(request.form['outlet_type'])
     
    #label encoding
    
    # print('before encode')

    # le= joblib.load(r'/home/elangoshunmugaraj/storesalesprediction/models/le.sav')
    # print('inside encode')

    # item_fat_content = np.array([item_fat_content])
    # print(item_fat_content)
    # item_fat_content=le.transform(item_fat_content)
    # item_type=le.transform(item_type)
    # outlet_size=le.transform(outlet_size)
    # outlet_location_type=le.transform(outlet_location_type)
    # outlet_type=le.transform(outlet_type)
    


     #Put all in the list

    inputs= np.array([item_weight,item_fat_content,item_visibility,item_type,item_mrp,
    outlet_establishment_year,outlet_size,outlet_location_type,
    outlet_type] ).reshape(1,-1)

    print(inputs)




    #Lets apply standered scalar
    
    sc= joblib.load(r'/home/elangoshunmugaraj/storesalesprediction/models/sc.sav')

    print(inputs)

    inputs_std= sc.transform(inputs)
    
    print(inputs_std)
    #Lets apply prediction

    model= joblib.load(r'/home/elangoshunmugaraj/storesalesprediction/models/random_forest_grid.sav')
    
    prediction = model.predict(inputs_std)
    print(prediction)    
    output = round(prediction[0], 2)
    print(output)    

    # return "good"

    return render_template('result.html', prediction_text="Predicted amount of the selected details: {}".format(output)) #jsonify({"prediction":output})

if __name__ == '__main__':
    app.run(debug=True)
