from statsmodels.tsa.vector_ar.var_model import VAR

def extractCoeff(timeseries_data,lag_order):
    '''
    Takes in a 7680x16 array to fit a VAR model and obtain the coefficients
    @return: 5x16x16 VAR coefficients array
    '''   
    model = VAR(timeseries_data)
    model_fit = model.fit(lag_order,trend='nc')
    coefs = model_fit.coefs #the lag coeffs
    return coefs

def combined_16x16x5_VAR(var_coeff_matrices): #takes 5x16x16 as input and transform to 16x16x5 
    '''
    Rearrange of the VAR coef to obtain a shape of 16x16x5
    @return: transformed 16x16x5 VAR
    '''   
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