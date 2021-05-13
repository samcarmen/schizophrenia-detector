#Flask libraries
from flask import Flask, render_template,request,json
import numpy as np
import tensorflow as tf
import logging

#Feature Extraction libraries
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy #installed
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
import scot
import scot.connectivity as cn

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
'''
to run the file:
## pip install keras
## pip install tensorflow

##error1: unable to import go view>command palette>python interpreter>select env interpreter

1.open terminal and type "env\Scripts\activate"
2.set FLASK_APP=app1.py
3.$env:FLASK_APP="app1.py"
4.type "flask run --reload"
'''

testData = [
    {
        "item1": "placeholder1",
        "item2": "placeholder2",
        "item3": "placeholder3"
    },
    {
        "item1": "placeholder4",
        "item2": "placeholder5",
        "item3": "placeholder6"
    }
]


global signal_var_coefficients 

def model_setup():
    global CNN_model
    CNN_model = tf.keras.models.load_model('saved_model\\model_1.hdf5')
    #set model = load_model(schizo_cnn.h5)


def preprocess_user_raw_data(signal_array):
    '''
    TODO: add code here
    '''
    arr=numpy.array(signal_array)
    arr=arr.reshape(16,7680) #16 channels, each channels 7680 timeseries data
    arr= arr.tolist() #converts back to normal list

    data = []
    for column in range(len(arr[0])): #len(arr[0]) is 7680
        channel_column=[]
        for row in range(len(arr)): #len(arr) is 16
            channel_column.append(arr[row][column])
        data.append(channel_column)
    #print(numpy.array(data))
    return data

def extractCoeff(timeseries_data,lag_order):
    #note: arr is a numpy array
    model = VAR(timeseries_data)
    model_fit = model.fit(lag_order,trend='nc')
    coefs = model_fit.coefs #the ith lag coeffs
    for matrix in coefs:
        for i in range(16):
            matrix[i][i]=0
    return coefs

def combined_16x16x5_VAR(var_coeff_matrices): #takes 5x16x16 as input and transform to 16x16x5    
    combined=[]
    for i in range(16): #loop through rows
        row=[]
        for j in range(16):
            five_bands_value=[]
            for x in range(len(var_coeff_matrices)): #loop through each coef matrix
                five_bands_value.append(var_coeff_matrices[x][i][j])
            row.append(five_bands_value)
        combined.append(row)
    return combined



"""
Functions below are for extracting PDC data
"""

def preprocess_and_extract_var_for_PDC(patient_coef): #performs concat on array 16x80
    combined=[]
    for i in range(16): #loop through rows
        channel=[]
        for x in range(len(patient_coef)): #loop through each coef matrix, if lag is 5, then got 5 matrix
            channel= channel + patient_coef[x][i].tolist()
        combined.append(channel)       
    return combined #16*80 matrix

def preprocess_and_extract_var_for_PDC_NEWNEW(patient_coef): #performs concat on array 16x16x5
    new_16x80_mat=[]
    for row in range(16): #loop through rows
        combined=[]
        for column in range(16): #loop through columns
            combined = combined + patient_coef[row][column]
        new_16x80_mat.append(combined)            
    return new_16x80_mat

def average(l):
    avg = sum(l) / len(l) 
    return avg
    
def get_PDC(var_coef): #the var_coef has a shape of 16 x 16*lag order
    no_of_bands=5
    patient_pdc=[]
    for x in range(no_of_bands):
        PDC_output = [None]*16
        for i in range(len(PDC_output)):
            PDC_output[i]=[None]*16
        patient_pdc.append(PDC_output)
    
    c = cn.connectivity(['PDC'], var_coef, nfft=64)
    patient=c['PDC']
    for row in range(16):
        for column in range(16):
            bandwidths=[(0,4),(4,8),(8,14),(14,31),(31,64)]            
            for b in range(len(bandwidths)): #get the average value of the bandwidths
                lower_band=bandwidths[b][0]
                upper_band=bandwidths[b][1]
                patient_pdc[b][row][column]=average(patient[row][column][lower_band:upper_band])
    
    for mat in patient_pdc: ##################################################
        for i in range(16):
            mat[i][i]=0
    return patient_pdc #returns 5x16x16

def combined_16x16x5_pdc(pdc_bands): #takes 5x16x16 as input and transform to 16x16x5
    combined=[]
    for i in range(16): #loop through rows
        row=[]
        for j in range(16):
            five_bands_value=[]
            for x in range(len(pdc_bands)): #loop through each coef matrix
                five_bands_value.append(pdc_bands[x][i][j])
            row.append(five_bands_value)
        combined.append(row)
    return combined

def norm_data(X):
    # keepdims makes the result shape (1, 1, 3) instead of (3,). 
    # This doesn't matter here, but would matter if you wanted to
    #  normalize over a different axis.
    X_min = X.min(axis=(0, 1), keepdims=True)
    X_max = X.max(axis=(0, 1), keepdims=True)
    X = (X - X_min)/(X_max - X_min)
    return X

model_setup() 
'''
Loads the model into the CNN_model variable when the website is initialized
So that the user doesn't have to wait for presaved model to be loaded while making a prediction.
'''


@app.route('/')
def home():
    app.logger.info('MESSAGE INFO HERE:Processing default request')
    return render_template('main.html', postss=testData)


@app.route('/uploadEEG',methods=['POST'])
def uploadEEG():
    app.logger.info('POST REQUEST RECEIVED')
    eeg_data=request.json['data'].split("\r\n")
    for i in range(len(eeg_data)):
        if eeg_data[i]=="":
            eeg_data.pop(i)
            app.logger.info('Removed item:' +str(i)+', it is an empty string. Cannot be converted to float')
        else:
            eeg_data[i]=float(eeg_data[i])
    app.logger.info("\nfirst 10 values:"+str(eeg_data[0:10]))
    app.logger.info("\nlast 10 values:"+str(eeg_data[len(eeg_data)-10:len(eeg_data)]))

    global preprocessed_signals
    preprocessed_signals = preprocess_user_raw_data(eeg_data)
    return json.dumps(preprocessed_signals)

@app.route('/VAR')
def getVAR():
    global signal_var_coefficients 
    signal_var_coefficients = extractCoeff(preprocessed_signals,5)

    global predict_input_VAR
    predict_input_VAR = combined_16x16x5_VAR(signal_var_coefficients)
    return json.dumps(signal_var_coefficients.tolist())

@app.route('/PDC')
def getPDC():
    global signal_PDC_coefficients 
    #signal_PDC_coefficients = preprocess_and_extract_var_for_PDC_NEWNEW(combined_16x16x5_VAR(signal_var_coefficients ))
    signal_PDC_coefficients = preprocess_and_extract_var_for_PDC(extractCoeff(preprocessed_signals,5))

    global feature_extracted_pdc_matrix
    feature_extracted_pdc_matrix = get_PDC(signal_PDC_coefficients)

    global predict_input_PDC
    predict_input_PDC = combined_16x16x5_pdc(feature_extracted_pdc_matrix)
                        #16x80                  #5x16x16                       #16x16x5
    return json.dumps([signal_PDC_coefficients,feature_extracted_pdc_matrix,predict_input_PDC])

@app.route('/predict')
def makePrediction():
    #normalized_data=numpy.array( [norm_data( numpy.array(predict_input_PDC) ) ])#converts to ndarray and normalize it
    #do we really need to normalize it??
    #normalized_data = numpy.array( [norm_data( numpy.array(predict_input_VAR) ) ] )
    result = CNN_model.predict([predict_input_PDC])
    return json.dumps(result.tolist())

if __name__ == '__main__':
    app.run(debug=True)
