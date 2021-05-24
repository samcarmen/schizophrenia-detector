import numpy
def preprocess_user_raw_data(signal_array):
    """
    Takes in a 122876 rows array and preprocess into 7680x16 array
    """
    arr=numpy.array(signal_array)
    arr=arr.reshape(16,7680) #16 channels, each channels 7680 timeseries data
    arr= arr.tolist() #converts back to normal list

    data = [] #transpose the rows to column and column to row
    for column in range(len(arr[0])): #len(arr[0]) is 7680
        channel_column=[]
        for row in range(len(arr)): #len(arr) is 16
            channel_column.append(arr[row][column])
        data.append(channel_column)
    #print(numpy.array(data))
    return data