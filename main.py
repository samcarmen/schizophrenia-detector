#Flask libraries
from flask import Flask, render_template,request,json
import numpy as np
import tensorflow as tf
import logging

#Feature Extraction libraries
from math import sqrt
import numpy
from statsmodels.tsa.vector_ar.var_model import VAR
import scot.connectivity as cn

from preprocessing import preprocess_user_raw_data
from VectorAutoregression import extractCoeff
from PartialDirectedCoherence import average,get_PDC,combined_16x16x5_pdc,preprocess_and_extract_var_for_PDC

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
'''
To run the file:

error1: scripts disabled on this system > open windows powershell and run as admin
        -type " Set-ExecutionPolicy RemoteSigned " without the ""
        -enter "Y" to allow scripts


1.open terminal on top > new terminal > select power shell (beside "+" sign)
2.type " env\Scripts\activate " without the ""
3.type " set FLASK_APP=app1.py " without the "" //OPTIONAL
4.type " $env:FLASK_APP='app1.py' "without the "" //OPTIONAL
5.type "python main.py"
'''

global signal_var_coefficients 

def model_setup():
    global CNN_model_1

    # To change to model, change the arguments to ('saved_model\\{{your model file name}}')
    CNN_model_1 = tf.keras.models.load_model('saved_model\\final_model.hdf5')

model_setup() 
'''
Loads the model into the CNN_model variable when the website is initialized
So that the user doesn't have to wait for the CNN model to be loaded while making a prediction.
'''

@app.route('/')
def home():
    '''
    This function routes the user to the home page, when the website is first loaded.
    '''
    app.logger.info('MESSAGE INFO HERE:Processing default request')
    return render_template('main.html')


@app.route('/uploadEEG',methods=['POST'])
def uploadEEG():
    '''
    This function is triggered when user uploads the EEG data.
    It will preprocess the EEG data into 7680x16 and store it as a
    global variable.
    '''
    app.logger.info('POST REQUEST RECEIVED')
    eeg_data=request.json['data'].split("\r\n")

    #store the post request's data in a list.
    for i in range(len(eeg_data)):
        if eeg_data[i]=="":
            eeg_data.pop(i)
            app.logger.info('Removed item:' +str(i)+', it is an empty string. Cannot be converted to float')
        else:
            eeg_data[i]=float(eeg_data[i])
    app.logger.info("\nfirst 10 values:"+str(eeg_data[0:10]))
    app.logger.info("\nlast 10 values:"+str(eeg_data[len(eeg_data)-10:len(eeg_data)]))

    #Start preprocessing the raw data
    global preprocessed_signals
    preprocessed_signals = preprocess_user_raw_data(eeg_data) #x7680x16
    return json.dumps(preprocessed_signals)

@app.route('/VAR')
def getVAR():
    '''
    This function is triggered when user clicks on the
    Feature Extraction (VAR) button, this function will then
    feature extract the 7680x16 raw data using a VAR model.
    '''
    global signal_var_coefficients 
    signal_var_coefficients = extractCoeff(preprocessed_signals,5)
    return json.dumps(signal_var_coefficients.tolist())

@app.route('/PDC')
def getPDC():
    '''
    This function is triggered when user clicks on the
    Feature Extraction (PDC) button, this function will then 
    apply the get_PDC with the VAR coefficients obtained previously. 
    '''
    #16x80 matrix
    global signal_PDC_coefficients 
    signal_PDC_coefficients = preprocess_and_extract_var_for_PDC(extractCoeff(preprocessed_signals,5))

    #5x16x16 matrix
    global feature_extracted_pdc_matrix
    feature_extracted_pdc_matrix = get_PDC(signal_PDC_coefficients)

    #16x16x5 matrix
    global predict_input_PDC
    predict_input_PDC = combined_16x16x5_pdc(feature_extracted_pdc_matrix)
                        #16x80                  #5x16x16                       #16x16x5
    return json.dumps([signal_PDC_coefficients,feature_extracted_pdc_matrix,predict_input_PDC])

@app.route('/predict')
def makePrediction():
    '''
    This function will feed the feature extracted data into the
    CNN Model for prediction
    '''
    results_list=[]
    results_list.append(CNN_model_1.predict([predict_input_PDC]).tolist())
    return json.dumps(results_list[0])

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
