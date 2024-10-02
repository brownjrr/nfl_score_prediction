import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN


pd.set_option('display.width', 10000)

def get_pbp_dataframe():
    df = pd.read_csv("../data/play_by_play.csv").drop_duplicates(subset=['Detail'])

    return df

def get_player_pos_dict():
    df = pd.read_csv("../data/players.csv")
    df['position'] = df['position'].fillna("POSITION_NOT_FOUND")

    return df[['player_id', 'position']].set_index("player_id").to_dict()['position']

def prepocess_text(x, player_pos_dict):
    pattern = r'<a href.*?/a>'

    players_refs = re.findall(pattern, x)
    
    players = []
    for i in players_refs:
        players.append(re.search(r'".*?"', i).group().strip("\"").split('/')[-1].replace(".htm", ""))
    
    player_tag_dict = dict(zip(players_refs, players))

    new_text = x
    for i in players_refs:
        if 'href="/teams/' in i:
            new_text = new_text.replace(i, "")
        else:
            # print((type(i), i))
            # print(player_tag_dict[i])
            pid = player_tag_dict[i]

            if pid in player_pos_dict:
                new_text = new_text.replace(i, player_pos_dict[pid])
            else:
                new_text = new_text.replace(i, f"<PLAYER_NOT_FOUND:{pid}>")

    new_text = re.sub(r'[\|/|")|(|#]', r'', new_text) # special_char

    return new_text

def preprocess_play_text(df):
    player_pos_dict = get_player_pos_dict()

    df['play_text'] = df['Detail'].apply(prepocess_text, args=(player_pos_dict,),)

    print(df)

    df = df[~((df['play_text'].str.contains("POSITION_NOT_FOUND")) | (df['play_text'].str.contains("PLAYER_NOT_FOUND")))]

    print(df)

    tfidf_vect = TfidfVectorizer()
    tfidf = tfidf_vect.fit_transform(df['play_text'].values)

    # wcss = []
    # for i in range(1,10):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    #     kmeans.fit_predict(tfidf)
    #     wcss.append(kmeans.inertia_)

    # plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], wcss)
    # plt.xlabel('No of clusters')
    # plt.ylabel('Values')
    # plt.show()

    # n_clusters = 6
    # model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    # labels = model.fit_predict(tfidf)
    
    # print(f"labels: {labels}")

    # df['cluster_label'] = labels # assigning labels to text

    # print(df)

    # for cluster_num in range(n_clusters):
    #     print(f'cluster_num: {cluster_num}')

    #     temp_df = df[df['cluster_label']==cluster_num].reset_index(drop=True)

    #     print(f"Num Rows WIth Label: {temp_df.shape[0]}")

    #     for i, ind in enumerate(df.index):
    #         if not i < 5:
    #             break

    #         print(temp_df['play_text'][ind])
        
    #     print("_________________")
    #     print()
    
    # print("Top terms per cluster:")
    # order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    # for i in range(n_clusters):
    #     print("Cluster %d:" % i, end='')
    #     for ind in order_centroids[i, :n_clusters]:
    #         print(' %s' % tfidf_vect.get_feature_names_out()[ind], end='')
    #         print()

    dbscan = DBSCAN(eps=1.0, min_samples=5)
    dbscan.fit(tfidf)

    cluster_labels = dbscan.labels_
    coords = tfidf.toarray()

    print(f"coords: {coords}")

    no_clusters = len(np.unique(cluster_labels) )
    no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)

    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)

    colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', cluster_labels))
    plt.scatter(x, y, c=colors, marker="o", picker=True)
    plt.show()

if __name__ == "__main__":
    df = get_pbp_dataframe()

    preprocess_play_text(df)