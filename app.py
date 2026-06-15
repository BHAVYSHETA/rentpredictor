from flask import Flask, url_for, render_template, redirect, request
from models.rentpredicter import rentprediction

app = Flask(__name__)

print("Initializing Model......")
rent_app = rentprediction("data/house_rent_dataset.csv")

rent_app.load_data()
rent_app.preprocess_data()
rent_app.split_data()
rent_app.load_model()
rent_app.y_pred = rent_app.model.predict(rent_app.X_test)
rent_app.evaluate()

print("Model:", rent_app.model)

@app.route('/')
def home():
    '''Main Route For Display The House Form'''
    return render_template('index.html', result=None)

@app.route('/predict', methods=['post'])
def predict():
    '''Handle Prediction Request'''
    try:
        area = float(request.form.get('area'))
        bedrooms = int(request.form.get('bedrooms'))
        bathrooms = int(request.form.get('bathrooms'))
        floor =  int(request.form.get('floor'))
        ageofproperty = int(request.form.get('ageofproperty'))
        furnishing = str(request.form.get('furnishing'))
        parking = str(request.form.get('parking'))
        city = str(request.form.get('city'))
        
        result = rent_app.predict_rent(area, bedrooms, bathrooms, floor, ageofproperty, furnishing, parking, city)

        return render_template('index.html', result=result)
    
    except Exception as e:
        error_result = {'error:', str(e)}
        return render_template('index.html', result=error_result)
    
@app.route('/model-stats')
def model_stats():
    '''Display model performance matrics'''
    rent_app.evaluate()
    return 'Model Metrics Printed In Console'

if __name__ == '__main__':
    app.run(debug=True)