import pandas as pd

def advancedStats(data, labels):
    '''Advanced stats should leverage pandas to calculate
    some relevant statistics on the data.

    data: numpy array of data
    labels: numpy array of labels
    '''
    # convert to dataframe
    df = pd.DataFrame(data)

    # print skew and kurtosis for every column
    for col in df:
        print('Column {} statistics:\n\tSkewness:{}\tKurtosis:{}'.format(col,df[col].skew(),df[col].kurtosis()))

    # assign in labels
    df["labels"] = labels

    print("\n\nDataframe statistics")


    # groupby labels into "benign" and "malignant"
    B = df[df['labels'] == 'B']
    M = df[df['labels'] == 'M']

    # collect means and standard deviations for columns,
    # grouped by label
    Bm=B.mean()
    Bs=B.std()
    Mm=M.mean()
    Ms=M.std()

    # Print mean and stddev for Benign
    print("Benign Stats: \nMean: \n{} \nStd: \n{}".format(Bm,Bs))

    print("\n")

    # Print mean and stddev for Malignant
    print("Malignant Stats: \nMean: \n{} \nStd: \n{}".format(Mm,Ms))
    

    
    
    '''#another method

    mean = df.groupby(['labels']).mean()
    std = df.groupby(['labels']).std()
    

    # Print mean and stddev for Benign
    print("Benign Stats: \nMean: \n{} \nStd: \n{}".format(mean.iloc[0],std.iloc[0]))

    print("\n")

    # Print mean and stddev for Malignant
    print("Malignant Stats: \nMean: \n{} \nStd: \n{}".format(mean.iloc[1],std.iloc[0]))
    '''
    
    


    




