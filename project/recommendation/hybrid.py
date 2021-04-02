import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from surprise.model_selection import train_test_split
from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy


warnings.simplefilter('ignore')


class RecommendationSystem:
    def __init__(self):
        self.inddict = None
        self.D = None
        self.products_n = pd.read_csv("/Users/priya/PycharmProject/flask/flask_auth_app/project/datasets/Dataset_merged2.csv")
        self.svd = SVD()
        self.indices_map = None
        print(">>>> Initialized")

    def train(self):
        ratings_dataset = pd.read_csv('/Users/priya/PycharmProject/flask/flask_auth_app/project/datasets/ratings_Beauty.csv')
        products_dataset = pd.read_csv('/Users/priya/PycharmProject/flask/flask_auth_app/project/datasets/dataset_final.csv')
        pr = ratings_dataset.drop_duplicates(['ProductId'])
        pdes = products_dataset[['title', 'image', 'asin', 'description']]
        products_n  = pd.merge(pr, pdes, left_on= 'ProductId', right_on= 'asin')


        products_wd = products_n[products_n['description'].notnull()].copy()
        products_wd = products_wd[products_wd['description'].map(len) >5]
        products_wd.reset_index(drop=True, inplace=True)
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        tfidf_des = tf.fit_transform(products_wd['description'])

        indices_n = pd.Series(products_wd['title'])
        self.inddict = indices_n.to_dict()
        self.inddict = dict((v,k) for k,v in self.inddict.items())

        self.D = euclidean_distances(tfidf_des)


        data = self.products_n[['UserId', 'ProductId', 'Rating']]
        reader = Reader(line_format='user item rating', sep=',')
        data = Dataset.load_from_df(data, reader=reader)


        trainset, testset = train_test_split(data, test_size=.2)
        self.svd.fit(trainset)
        self.svd.predict(1, 7535842801)
        predictions = self.svd.test(testset)
        accuracy.rmse(predictions)
        accuracy.mae(predictions)



        def convert_int(x):
            try:
                return int(x)
            except:
                return np.nan


        self.indices_map = self.products_n.set_index('id')
        self.products_n['Rating'] = self.products_n['Rating'].apply(convert_int)

        print(">>>> Training completed")

    def test(self, user_id, title):
        ind = self.inddict[title]
        distance = list(enumerate(self.D[ind]))
        distance = sorted(distance, key=lambda x: x[1])
        distance = distance[1:50]
        products_index = [i[0] for i in distance]
        products = self.products_n.iloc[products_index][['title', 'description', 'Rating', 'UserId', 'id']]
        products['est'] = products['id'].apply(lambda x: self.svd.predict(user_id, self.indices_map.loc[x]['ProductId']).est)
        products = products.sort_values('est', ascending=False)
        return products.head(10)

# print(system.test(1, 'Mary Kay Satin Hands Hand Cream Travel MINI Size Set of 6'))

system = RecommendationSystem()
# system.train()
# result = system.test(7535842801, 'Mary Kay Satin Hands Hand Cream Travel MINI Size Set of 6')

# print(result.to_json())