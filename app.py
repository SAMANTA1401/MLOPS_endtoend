from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pipeline.prediction_pipeline import CustomData, PredictionPipeline
from exception.exception import CustomException
from logger.logging import logging
import sys


app = Flask(__name__)

cut_category = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_category = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_category = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

@app.route('/', methods=['GET', 'POST'])
def index():
    logging.info('enter user interface')
    return render_template('index.html', cut_category=cut_category, color_category=color_category, clarity_category=clarity_category)


@app.route('/predict', methods=['GET','POST'])
def predict():
    logging.info('received prediction request')
    try:
        features = request.form.to_dict()
        logging.info(f'received raw features: \n{features}')
        obj = PredictionPipeline()
        data = CustomData(carat=features['carat'], depth=features['depth'], table=features['table'], x=features['x'], y=features['y'], z=features['z'], cut=features['cut'], color=features['color'], clarity=features['clarity'])
        df = data.get_data_as_dataframe()
        logging.info('created dataframe from raw features')
        output_price = obj.predict(df)
        logging.info('predicted price successfully')
        print(output_price)
        price = str(output_price[0])
        # return jsonify(data)
        # return jsonify({'price': str(output_price[0])})
        return render_template('index.html',cut_category=cut_category, color_category=color_category, clarity_category=clarity_category, price=price)
    
    except Exception as e:
        error = CustomException(e, sys)
        raise logging.info(error)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)