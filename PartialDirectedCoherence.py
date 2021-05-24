import scot
import scot.connectivity as cn
def average(l):
    '''
    Obtain the average of all the values in a list
    '''
    avg = sum(l) / len(l) 
    return avg
    
def preprocess_and_extract_var_for_PDC(patient_coef):
    '''
    Takes in the 16x16x5 VAR coefficient array and 
    concatenate each rows to obtain 16x80 array
    @return: a concatenated VAR coefs of 16x80 array
    '''
    combined=[]
    for i in range(16): #loop through rows
        channel=[]
        for x in range(len(patient_coef)): #loop through each coef matrix, if lag is 5, then got 5 matrix
            channel= channel + patient_coef[x][i].tolist()
        combined.append(channel)       
    return combined #16*80 matrix
    
def get_PDC(var_coef): #the var_coef has a shape of 16 x 16*lag order
    '''
    Takes in the concatenated 16x80 VAR coefficient array and 
    apply the pdc function from SCoT library with a sampling rate of 128/2hz.
    @return: 5x16x16 array, where each of the 5 array represents a frequency band,
             eg: Alpha, Beta, Gamma, Theta, Delta 
    '''
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
    return patient_pdc #returns 5x16x16

def combined_16x16x5_pdc(pdc_bands): #takes 5x16x16 as input and transform to 16x16x5
    '''
    Takes in the 5x16x16 PDC data array and rearrange them to be 16x16x5
    @return: 16x16x5 PDC data array
    '''
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