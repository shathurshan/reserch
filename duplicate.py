import nltk as nltk
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os
import glob

nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer

# Datasets in duplicate questions
dataset1 = pd.read_csv("CSV files/duplicate_master_questions.csv")
dataset2 = pd.read_csv("CSV files/duplicate_duplicate_questions.csv")
dataset3 = pd.read_csv("CSV files/non_duplicate_1_question.csv")
dataset4 = pd.read_csv("CSV files/non_duplicate_2_question.csv")

# Titles of the dataset
title1 = dataset1['Title']
title2 = dataset2['Title']
title3 = dataset3['Title']
title4 = dataset4['Title']

# Bodies of the dataset
body1 = dataset1['Body']
body2 = dataset2['Body']
body3 = dataset3['Body']
body4 = dataset4['Body']

# Tags of the dataset
tags1 = dataset1['Tags']
tags2 = dataset2['Tags']
tags3 = dataset3['Tags']
tags4 = dataset4['Tags']


# Target variable
target_variable1 = dataset1['Type']
target_variable2 = dataset3['Type']


def getData(list1, list2):
    count = 0
    similarity = []
    for d in list1:
        data = [d, list2[count]]
        similarity.append(calculateSimlarity(data))
        count += 1
    return similarity


def getTags(list1, list2):
    count = 0
    similarity = []
    for d in list1:
        data = [d, list2[count]]
        similarity.append(calculateSimlarity2(data))
        count += 1
    return similarity


def calculateSimlarity(arr):
    Tfidf_vect = TfidfVectorizer(stop_words='english')
    vector_matrix = Tfidf_vect.fit_transform(arr)

    cosine_similarity_matrix = cosine_similarity(vector_matrix)

    return cosine_similarity_matrix[0][1]


def calculateSimlarity2(arr):
    Tfidf_vect = TfidfVectorizer(analyzer='char_wb')
    vector_matrix = Tfidf_vect.fit_transform(arr)

    cosine_similarity_matrix = cosine_similarity(vector_matrix)

    return cosine_similarity_matrix[0][1]


title_value = getData(title1, title2)
body_value = getData(body1, body2)
tag_value = getTags(tags1, tags2)

title_value_duplicate = getData(title3, title4)
body_value_duplicate = getData(body3, body4)
tag_value_duplicate = getTags(tags3, tags4)


def final_data(value1, value2, value3, value4):
    finalData = []
    count = 0
    for p in value1:
        finalData.append([p, value2[count], value3[count], value4[count]])
        count += 1
    return finalData


fina_lvalue = final_data(title_value, body_value, tag_value, target_variable1)
duplicate_final_value = final_data(title_value_duplicate, body_value_duplicate, tag_value_duplicate, target_variable2)
#
header = ['Title Sim', 'Body Sim', 'Tags Sim', 'isDuplicate']
# # data = [
# #     ['Albania', 28748, 'AL', 'ALB'],
# #     ['Algeria', 2381741, 'DZ', 'DZA'],
# #     ['American Samoa', 199, 'AS', 'ASM'],
# #     ['Andorra', 468, 'AD', 'AND'],
# #     ['Angola', 1246700, 'AO', 'AGO']
# # ]
data = fina_lvalue
duplicate_data = duplicate_final_value

# # creating duplicate result csv
# with open('duplicate_similarity.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#
#     # write the header
#     writer.writerow(header)
#
#     # write multiple rows
#     writer.writerows(data)
#
# # creating non duplicate result csv
# with open('non_duplicate_similarity.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#
#     # write the header
#     writer.writerow(header)
#
#     # write multiple rows
#     writer.writerows(duplicate_data)



# 1. defines path to csv files
os.chdir(r"C:\Users\Acer\Desktop\Research\Algorithms\Cosine Similarity\Research_cosine\Results")

# 2. Match the pattern (‘csv’) and save the list of file names in the ‘all_filenames’
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# 3. Combine all files in the list and export as CSV
# combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
# export to csv
combined_csv.to_csv("similarity_results.csv", index=False)
