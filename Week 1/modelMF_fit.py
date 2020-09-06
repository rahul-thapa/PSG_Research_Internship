import random
from scipy.sparse import csr_matrix, dok_matrix
from math import ceil
import numpy as np
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict


def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, true_r, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# Reading data
df = pd.read_csv('./rating.csv', sep="\t")
print(df.head())

# Creating reader object
reader = Reader(rating_scale=(1, 5))

# Specifying the columns for training
data = Dataset.load_from_df(df[['U_ID', 'P_ID', 'RATING']], reader)

trainset, testset = train_test_split(data, test_size=.20)

# Defining the algorithm
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)

predictions = algo.test(testset)

# print(predictions[1:5])

# Then compute Root Mean Square Error
accuracy.rmse(predictions)
#preds = algo.predict(uid="4", iid="5", r_ui=4, verbose=True)


# for ind in range(1000):
#     predictions[ind]
#     predictedValue = algo.predict(
#       uid=str(df['U_ID'][ind]), iid=str(df['P_ID'][ind]), r_ui=df['RATING'][ind], verbose=True)
#     print(predictedValue)


# Taking the ratio of ground truth and predicted values from the predictions array which has the
# predictions made from the testset


def calc_ndcg(user, predictions):

    averageValues = []

    for i in range(len(predictions)):
        gt = predictions[i][1]  # ground truth
        pt = predictions[i][2]  # predicted truth
        if(pt > 0 and gt > 0):
            ratio = gt/pt
            averageValues.append(ratio)

    ndcg = 0
    for x in averageValues:
        ndcg += x
    ndcg = ndcg/(len(averageValues))
    return {user: ndcg}
    #print("NDCG:", ndcg)


ndcg_values_final = []
#shuffledPredictions = np.random.shuffle(predictions)

top_n = get_top_n(predictions, n=10)

itr = 0
for uid, user_ratings in top_n.items():
    res = calc_ndcg(uid, user_ratings)
    ndcg_values_final.append(res[uid])
    itr += 1
    if itr > 20:
        break
    # print(user_ratings)
    #print(uid, [iid for (iid, _) in user_ratings])

print("Final ndcg value: ", sum(ndcg_values_final)/len(ndcg_values_final))
# print(top_n)
