import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


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
    df = df[~((df['play_text'].str.contains("POSITION_NOT_FOUND")) | (df['play_text'].str.contains("PLAYER_NOT_FOUND")))]

    return df

def show_cluster_examples(df, labels, n_clusters, num_examples=5, show_top_terms=False, model=None, tfidf_vect=None):
    print(f"labels: {labels}")

    df['cluster_label'] = labels # assigning labels to text

    print(df)

    for cluster_num in range(n_clusters):
        print(f'cluster_num: {cluster_num}')

        temp_df = df[df['cluster_label']==cluster_num].reset_index(drop=True)

        print(f"Num Rows WIth Label: {temp_df.shape[0]}")

        for i, ind in enumerate(df.index):
            if not i < num_examples:
                break

            print(temp_df['play_text'][ind])
        
        print("_________________")
        print()

    if show_top_terms:
        print("Top terms per cluster:")
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        for i in range(n_clusters):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :n_clusters]:
                print(' %s' % tfidf_vect.get_feature_names_out()[ind], end='')
                print()

def kmean_cluster_2d():
    df = get_pbp_dataframe().head(5000)

    df = preprocess_play_text(df)

    tfidf_vect = TfidfVectorizer()
    tfidf = tfidf_vect.fit_transform(df['play_text'].values)

    # Reduce dimensionality using LSA (Latent Semantic Analysis)
    lsa = TruncatedSVD(n_components=2)
    lsa_matrix = lsa.fit_transform(tfidf)

    # Perform K-Means clustering
    k = 4 # best
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(lsa_matrix)

    # Visualize the clusters
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    plt.figure(figsize=(10, 8))
    plt.scatter(lsa_matrix[:, 0], lsa_matrix[:, 1], c=labels, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1,], s=200, c='red')
    plt.title("Text Clustering with K-Means")
    plt.show()

def kmeans_cluster_3d(df, num_examples):
    print(df)

    tfidf_vect = TfidfVectorizer()
    tfidf = tfidf_vect.fit_transform(df['play_text'].values)

    # Reduce dimensionality using LSA (Latent Semantic Analysis)
    lsa = TruncatedSVD(n_components=4)
    lsa_matrix = lsa.fit_transform(tfidf)

    # Perform K-Means clustering
    k = 4 # best
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(lsa_matrix)

    # Visualize the clusters
    labels = kmeans.labels_

    print(f"Unique Labels: {set(labels)}")

    show_cluster_examples(df, labels, k, num_examples=num_examples, show_top_terms=False, model=None, tfidf_vect=None)

    cluster_centers = kmeans.cluster_centers_

    fig = plt.figure(figsize=(10, 8))
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')

    w = np.array(labels==0)
    x = np.array(labels==1)
    y = np.array(labels==2)
    z = np.array(labels==3)

    ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],c="black",s=150,label="Centers",alpha=1)
    ax.scatter(lsa_matrix[x,0],lsa_matrix[x,1],lsa_matrix[x,2],c="blue",s=40,label="C1")
    ax.scatter(lsa_matrix[y,0],lsa_matrix[y,1],lsa_matrix[y,2],c="yellow",s=40,label="C2")
    ax.scatter(lsa_matrix[z,0],lsa_matrix[z,1],lsa_matrix[z,2],c="red",s=40,label="C3")
    ax.scatter(lsa_matrix[w,0],lsa_matrix[w,1],lsa_matrix[w,2],c="green",s=40,label="C4")

    plt.show()

def transform_quarter_col(x):
    if x in ['1', '2', '3', '4']: return x
    elif x == "OT": return "5"
    elif x == "2OT": return "6"
    elif x == "3OT": return "7"
    elif x == "4OT": return "8"

    assert False, "Unknown quarter value"

def transform_time_col(x):
    return x.zfill(5)

def get_scoring_play_bool_col(group):
    group[['team_1_diff', 'team_2_diff']] = group[['team_1', 'team_2']].diff()

    group['scoring_play_team_1'] = False
    group['scoring_play_team_2'] = False

    group['scoring_play_team_1'] = group.apply(lambda row: True if row['team_1_diff']>0 else False, axis=1)
    group['scoring_play_team_2'] = group.apply(lambda row: True if row['team_2_diff']>0 else False, axis=1)

    group['scoring_play'] = group[
        ['scoring_play_team_1', 'scoring_play_team_2']
    ].apply(lambda row: True if row['scoring_play_team_1']==True or row['scoring_play_team_2']==True else False, axis=1)

    return group

def tsne_cluster(df, num_examples):
    print(f"Initial df size: {df.shape}")

    # unique id cols: boxscore_id
    df = df.groupby('boxscore_id').apply(get_scoring_play_bool_col).reset_index(drop=True)

    # print(df[['boxscore_id', 'Quarter', 'Time', 'team_1', 'team_2', 'scoring_play']])

    df = df.dropna(subset=['Quarter', 'Time'])

    df['Quarter'] = df['Quarter'].apply(transform_quarter_col)
    df['Time'] = df['Time'].apply(transform_time_col)
    
    df = df.sort_values(by=['Quarter', 'Time'], ascending=[True, False])

    print(df)

    tfidf_vect = TfidfVectorizer()
    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()
    
    print(tfidf)

    df = pd.DataFrame(tfidf, index=df['boxscore_id'], columns=tfidf_vect.get_feature_names_out())

    print(df)

    # init = "random"
    # learning_rate = 200.0

    # tsne = TSNE(init=init, learning_rate=learning_rate, random_state=0)

    # X_tsne = tsne.fit_transform(X_fruits_normalized)

    # plt.figure(figsize=(8, 6))
    # plt.grid(alpha=0.2)
    # plot_labelled_scatter(
    #     X_tsne,
    #     y_fruits,
    #     ["apple", "mandarin", "orange", "lemon"],
    #     title="Fruits dataset t-SNE",
    # )
    # plt.tight_layout()

def sandbox():
    pass
    ################################################################################
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

    ################################################################################
    # print("running DBSCAN")
    # dbscan = DBSCAN(eps=1.0, min_samples=5)

    # print("fitting data with DBSCAN")
    # dbscan.fit(tfidf)

    # cluster_labels = dbscan.labels_
    # coords = tfidf.toarray()

    # print(f"coords: {coords}")

    # no_clusters = len(np.unique(cluster_labels) )
    # no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)

    # print('Estimated no. of clusters: %d' % no_clusters)
    # print('Estimated no. of noise points: %d' % no_noise)

    # colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', cluster_labels))
    # plt.scatter(x, y, c=colors, marker="o", picker=True)
    # plt.show()

if __name__ == "__main__":
    df = get_pbp_dataframe().head(5000)
    df = preprocess_play_text(df)

    # kmean_cluster_2d()
    # kmeans_cluster_3d(df, num_examples=10)
    tsne_cluster(df, num_examples=10)