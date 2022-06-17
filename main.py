# Rohan Ballapragada
# Run python3 main.py
# After running, it takes time to interpret but it prints accuracy

import string
import pandas as pd
import re
vocab_list = []

# function to process the txt files
def formatting_pre_processing_files(inputfile,outputfile):
    stripped = []
    stripped_nodup = [] # no stripped duplicate
    new_list = []
    with open(inputfile, "r") as f:
        text = f.read()
        words = text.split()
        write_list = []
        table = str.maketrans("", "", string.punctuation) # remove all punctuation
        stripped = [w.translate(table).lower() for w in words] # lowercase all words
        stripped_sort = sorted(stripped) # sort vocab
        [stripped_nodup.append(x) for x in stripped_sort if x not in stripped_nodup] # append vocab, skip if duplicate
        stripped_nodup = stripped_nodup[3:] # remove first threw words
        stripped_nodup.append("classlabel")
        write_list.append(stripped_nodup)
        if(inputfile == "trainingSet.txt"):         #this is being added so that we can get the vocab later on line 76 of this code.
            vocab_list.append(write_list[0])

        # iterate through stripped reviews
        #print("stripped: " + str(stripped))
        for x in stripped:
            # append to list when not 1 or 0
            if x != '1' and x != '0':
                new_list.append(x)
            else:
                # filter for list of words per review
                list_line = list(filter(lambda i: i in new_list, stripped_nodup))
                new_list = []
                list_indexes = []
                # Iterate through word
                for j in list_line:
                    # get index from no duplicate list
                    list_indexes.append(stripped_nodup.index(j))
                # add 0
                for j in range(0, len(stripped_nodup) - 1):
                    new_list.append('0')
                # wherever a word exists put 1
                for j in list_indexes:
                    new_list[j] = '1'
                new_list = new_list + [x]
                write_list.append(new_list)
                list_line = []
                new_list = []
    # open into txt files
    with open(outputfile, 'w') as d:
        d.writelines(','.join(str(j) for j in i) + '\n' for i in write_list)
        #d.write(str(write_list))

def train():
    # read from preprocessed file to train
    df = pd.read_csv("preprocessed_train.txt", sep=",", index_col=None)
    # create dataframes for positive and negative
    df_positive = df.loc[df['classlabel'] == 1]
    df_negative = df.loc[df['classlabel'] == 0]
    sum_positive = len(df_positive.index)
    sum_negative = len(df_negative.index)
    # Take size of vocab
    negative_columns = len(df_negative.columns)

    vocab = {}
    probability = ()
    # Iterate through the vocab words
    for i in range(0, negative_columns - 1):
        # probability calculation
        probability = ((len(df_positive[df_positive[df_positive.columns[i]] == 1].index) + 1)/(len(df_positive[df_positive.columns[i]].index) + 2), (len(df_negative[df_negative[df_negative.columns[i]] == 1].index) + 1)/(len(df_negative[df_negative.columns[i]].index + 2)))
        # put into vocab dictionary
        vocab[df_negative.columns[i]] = probability
    return vocab, sum_positive, sum_negative

def test(vocab, sum_positive, sum_negative):
    # open test file
    df_test = pd.read_csv("preprocessed_test.txt", sep=",", index_col=None)
    count = 0
    # Iterate through rows
    for row in df_test.index:
        #print(row)
        probability = 1
        # get all indexes for row that equal 1
        indices = [i for i, x in enumerate(df_test.iloc[row].values) if x == 1]
        # get probability for review being positive and negative
        p_positive = sum_positive/(sum_positive + sum_negative)
        p_negative = sum_negative/(sum_positive + sum_negative)
        # Iterate through indices
        for j in indices:
            # get word and check if it exists in vocab dictionary
            colname = df_test.columns[j]
            if (colname in vocab):
                # multuple probabilities for each word positive and negative
                p_positive *= vocab[colname][0]
                p_negative *= vocab[colname][1]
            else:
                # When coming across vocab not in dictionary value added to 1 is 0
                p_positive *= 1/(sum_positive + 2)
                p_negative *= 1/(sum_negative + 2)
        # check if positive is greater than negative
        # if it is iterate count by 1
        if (p_positive > p_negative):
            if (df_test.iloc[row]['classlabel'] == 1):
                count+=1
        # if negative greater than positive
        # iterate count by 1 if it is
        else:
            if (df_test.iloc[row]['classlabel'] == 0):
                count+=1
    return count, len(df_test)

def main():
    formatting_pre_processing_files('trainingSet.txt','preprocessed_train.txt')
    formatting_pre_processing_files('testSet.txt','preprocessed_test.txt')
    vocab, sum_positive, sum_negative = train()
    count, total = test(vocab, sum_positive, sum_negative)
    print(count/total)

main()







