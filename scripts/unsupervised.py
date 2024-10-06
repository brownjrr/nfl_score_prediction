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
import plotly.express as px
import time
import datetime
from matplotlib import colormaps
import spacy
from sklearn import metrics


pd.set_option('display.width', 10000)

def get_pbp_dataframe():
    df = pd.read_csv("../data/play_by_play.csv").drop_duplicates(subset=['Detail']).dropna(subset=['Detail'])

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

def get_time_in_seconds(x):
    x = time.strptime(x, "%M:%S")
    x = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    return x

def preprocess_play_text(df):
    player_pos_dict = get_player_pos_dict()

    df['play_text'] = df['Detail'].apply(prepocess_text, args=(player_pos_dict,),)
    df = df[~((df['play_text'].str.contains("POSITION_NOT_FOUND")) | (df['play_text'].str.contains("PLAYER_NOT_FOUND")))]

    df = df.groupby('boxscore_id').apply(get_scoring_play_bool_col).reset_index(drop=True)

    df = df.dropna(subset=['Quarter', 'Time'])

    df['Quarter'] = df['Quarter'].apply(transform_quarter_col)
    df['Time'] = df['Time'].apply(transform_time_col)
    df['time_in_sec'] = df['Time'].apply(get_time_in_seconds)
    
    df = df.sort_values(by=['boxscore_id', 'Quarter', 'Time'], ascending=[True, True, False])

    return df

def show_cluster_examples(df, labels, n_clusters, num_examples=5, show_top_terms=False, model=None, tfidf_vect=None):
    print(f"labels: {labels}")

    df['cluster_label'] = labels # assigning labels to text

    print(df)

    print(f"Unique Labels: {df['cluster_label'].unique()}")

    for cluster_num in df['cluster_label'].unique():
        print(f'cluster_num: {cluster_num}')

        temp_df = df[df['cluster_label']==cluster_num].reset_index(drop=True)

        print(f"Num Rows WIth Label: {temp_df.shape[0]}")

        # for i, ind in enumerate(temp_df.index):
        for i, ind in enumerate(df.index):
            if not i < num_examples:
                break
            
            try:
                print(temp_df['play_text'][ind])
            except:
                # print((ind, temp_df['play_text']))
                pass
        
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

def tsne_cluster_2d(df, num_examples):
    print(f"Initial df size: {df.shape}")

    tfidf_vect = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1,2), min_df=1, max_df=0.9,)
    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()
    
    text_df = pd.DataFrame(tfidf, columns=tfidf_vect.get_feature_names_out())

    # temp_df  = pd.concat([df[['Quarter', 'Down', 'time_in_sec', 'ToGo', 'scoring_play']], text_df], axis=1).dropna(subset=['Quarter', 'Down', 'time_in_sec', 'ToGo']).fillna(0)
    temp_df  = pd.concat([df[['boxscore_id', 'scoring_play', "play_text"]], text_df], axis=1).dropna(subset=['play_text']).fillna(0)

    X = temp_df.drop(["boxscore_id", "scoring_play", "play_text"], axis=1)
    y = temp_df['scoring_play']

    for learning_rate in [100]:
        for perp in [50]:
            for early_exaggeration in [50]:
                for metric in ['cityblock']:
                    tsne = TSNE(n_components=2, learning_rate=learning_rate, perplexity=perp, early_exaggeration=early_exaggeration, metric=metric, random_state=42)
                    tsne = tsne.fit_transform(X)
                    
                    tsne_df = pd.DataFrame(tsne, columns=['x', 'y'])
                    tsne_df = pd.concat([temp_df[['boxscore_id', 'play_text', 'scoring_play']], tsne_df], axis=1)

                    fig = px.scatter(
                        tsne_df, 
                        x='x', 
                        y='y',
                        color='scoring_play',
                        title=f"TSNE(learning_rate={learning_rate}, perplexity={perp}, early_exaggeration={early_exaggeration}, metric={metric})",
                        hover_data=['play_text', 'boxscore_id']
                    )
                    fig.show()

def tsne_cluster_3d(df, num_examples):
    print(f"Initial df size: {df.shape}")

    tfidf_vect = TfidfVectorizer()
    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()
    
    text_df = pd.DataFrame(tfidf, columns=tfidf_vect.get_feature_names_out())

    # temp_df  = pd.concat([df[['Quarter', 'Down', 'time_in_sec', 'ToGo', 'scoring_play']], text_df], axis=1).dropna(subset=['Quarter', 'Down', 'time_in_sec', 'ToGo']).fillna(0)
    temp_df  = pd.concat([df[['scoring_play', 'Down']], text_df], axis=1).fillna(0)

    X = temp_df.drop("scoring_play", axis=1)
    y = temp_df['scoring_play']

    for learning_rate in [10000]:
        for perp in [5, 25, 50]:
            tsne = TSNE(n_components=3, learning_rate=learning_rate, perplexity=perp, random_state=42)
            tsne = tsne.fit_transform(X)

            fig = px.scatter_3d(
                tsne, x=0, y=1, z=2,
                color=y
            )
            fig.show()

# def dbscan_2d(df, num_examples):
#     print(f"Initial df size: {df.shape}")

#     positions_df = pd.read_csv("../data/positions.csv")
#     # vocab = [i.lower() for i in positions_df['adj_pos'].unique()] + ['touchdown', 'punt', 'punts', 'penalty', 'field goal', 'kick', 'pass', 'complete', 'incomplete', 'tackle', 'timeout',]
#     vocab = ['punt', 'punts', 'field goal', 'kick', 'kicks off', 'pass', 'timeout'] # BEST
#     # vocab = get_verbs_in_corpus(df)
#     # vocab = [i.lower() for i in positions_df['adj_pos'].unique()] + ['punt', 'punts', 'field goal', 'kick', 'kicks off', 'pass', 'timeout']

#     print(f"vocab: {vocab}")

#     tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1,4), min_df=1, max_df=1.0,)
#     tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()

#     tsne = TSNE(n_components=2, learning_rate=50, perplexity=1, early_exaggeration=100, metric="cityblock", random_state=42) # BEST
#     # tsne = TSNE(n_components=2, learning_rate=100, perplexity=100, early_exaggeration=100, metric="cityblock", random_state=42)
#     tsne = tsne.fit_transform(tfidf)
    
#     highest_silh_score = -1
#     params = {'eps': None, 'min_samples': None}

#     for eps in [1, 5, 10, 25, 50, 75, 100,]:
#         for min_samples in [10, 25, 50, 75, 100]:
#             print(f"eps={eps} | min_samples={min_samples}")

#             # dbscan = DBSCAN(eps=1, min_samples=50) # BEST
#             dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#             dbscan.fit(tsne)
            
#             cluster_labels = dbscan.labels_

#             no_clusters = len(np.unique(cluster_labels))
#             no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)

#             print('Estimated no. of clusters: %d' % no_clusters)
#             print('Estimated no. of noise points: %d' % no_noise)

#             silhouette_score = metrics.silhouette_score(tsne, cluster_labels)
            
#             if silhouette_score > highest_silh_score:
#                 highest_silh_score = silhouette_score
#                 params['eps'] = eps
#                 params['min_samples'] = min_samples

#             print(f"silhouette_score: {silhouette_score}")

#             # show_cluster_examples(df, cluster_labels, no_clusters, num_examples=num_examples, show_top_terms=False, model=None, tfidf_vect=None)

#             # cluster_color_dict = dict(zip(np.unique(cluster_labels), list(colormaps)[:no_clusters]))
#             # colors = list(map(lambda x: x, cluster_labels))
            
#             # # plt.scatter(x=tsne[:, 0], y=tsne[:, 1], c=colors, marker="o", picker=True)
#             # # plt.show()
#             print("=================================================================================")

#     print(f"Highest Silhouette Score: {highest_silh_score}\nBest Parameters: {params}")
    
#     dbscan = DBSCAN(**params)
#     dbscan.fit(tsne)
    
#     cluster_labels = dbscan.labels_

#     no_clusters = len(np.unique(cluster_labels))
#     no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)

#     print('Estimated no. of clusters: %d' % no_clusters)
#     print('Estimated no. of noise points: %d' % no_noise)

#     show_cluster_examples(df, cluster_labels, no_clusters, num_examples=num_examples, show_top_terms=False, model=None, tfidf_vect=None)

#     colors = list(map(lambda x: x, cluster_labels))
#     plt.scatter(x=tsne[:, 0], y=tsne[:, 1], c=colors, marker="o", picker=True)
#     plt.show()

def dbscan_2d(df, num_examples=None, params=None):
    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    vocab = ['punt', 'punts', 'field goal', 'kick', 'kicks off', 'pass', 'timeout'] # BEST

    print(f"vocab: {vocab}")

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1,4), min_df=1, max_df=1.0,)
    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()

    tsne = TSNE(n_components=2, learning_rate=50, perplexity=1, early_exaggeration=100, metric="cityblock", random_state=42) # BEST
    data = tsne.fit_transform(tfidf)
    
    if params is None:
        highest_silh_score = -1
        params = {'eps': None, 'min_samples': None}

        for eps in [1, 5, 10, 25, 50, 75, 100,]:
            for min_samples in [10, 25, 50, 75, 100]:
                print(f"eps={eps} | min_samples={min_samples}")

                # dbscan = DBSCAN(eps=1, min_samples=50) # BEST
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan.fit(data)
                
                cluster_labels = dbscan.labels_

                no_clusters = len(np.unique(cluster_labels))
                no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)

                print('Estimated no. of clusters: %d' % no_clusters)
                print('Estimated no. of noise points: %d' % no_noise)

                silhouette_score = metrics.silhouette_score(data, cluster_labels)
                
                if silhouette_score > highest_silh_score:
                    highest_silh_score = silhouette_score
                    params['eps'] = eps
                    params['min_samples'] = min_samples

                print(f"silhouette_score: {silhouette_score}")

                print("=================================================================================")

        print(f"Highest Silhouette Score: {highest_silh_score}\nBest Parameters: {params}")
    
    dbscan = DBSCAN(**params)
    dbscan.fit(data)
    
    cluster_labels = dbscan.labels_

    no_clusters = len(np.unique(cluster_labels))
    no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)

    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)
    
    if num_examples is not None:
        show_cluster_examples(df, cluster_labels, no_clusters, num_examples=num_examples, show_top_terms=False, model=None, tfidf_vect=None)
    
    # colors = list(map(lambda x: x, cluster_labels))
    # plt.scatter(x=data[:, 0], y=data[:, 1], c=colors, marker="o", picker=True)
    # plt.show()

    df['cluster_label'] = cluster_labels

    df.to_csv("../data/play_by_play_clustered.csv", index=False)


def dbscan_3d(df, num_examples):
    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    # vocab = [i.lower() for i in positions_df['adj_pos'].unique()] + ['touchdown', 'intercepted', 'fumble', 'up', 'fumbles', 'left', 'right', 'deep', 'short', 'sack', 'sacked', 'punt', 'punts', 'penalty', 'field goal', 'extra point', 'kick', 'kicks off', 'return', 'returned', 'defended', 'pass', 'complete', 'incomplete', 'tackle', 'timeout', 'two point']
    # vocab = ['up', 'left', 'right', 'punt', 'punts', 'penalty', 'field goal', 'extra point', 'kick', 'kicks off', 'pass']
    # vocab = ['punt', 'punts', 'penalty', 'field goal', 'extra point', 'kick', 'kicks off', 'pass']
    vocab = ['punt', 'punts', 'field goal', 'kick', 'kicks off', 'pass', 'timeout']

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1,4), min_df=1, max_df=1,)

    print(tfidf_vect.get_feature_names_out())

    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()

    tsne = TSNE(n_components=3, learning_rate=100, perplexity=1, early_exaggeration=100, metric="cityblock", random_state=42)
    tsne = tsne.fit_transform(tfidf)
    
    dbscan = DBSCAN(eps=1, min_samples=100)
    dbscan.fit(tsne)
    
    cluster_labels = dbscan.labels_

    unique_labels = np.unique(cluster_labels)
    no_clusters = len(unique_labels)
    no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)

    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)

    show_cluster_examples(df, cluster_labels, no_clusters, num_examples=num_examples, show_top_terms=False, model=None, tfidf_vect=None)

    cluster_color_dict = dict(zip(unique_labels, list(colormaps)[:no_clusters]))
    colors = list(map(lambda x: x, cluster_labels))
    
    fig = plt.figure(figsize=(10, 8))
    
    ax = fig.add_subplot(projection='3d')

    labels_list = [np.array(cluster_labels==label) for label in unique_labels]

    # ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],c="black",s=150,label="Centers",alpha=1)

    for i, labels in enumerate(labels_list):
        ax.scatter(tsne[labels,0],tsne[labels,1],tsne[labels,2],s=10,label=f"C{i}")

    plt.show()

def dbscan_v2_2d(df, num_examples):
    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    vocab = ['punt', 'punts', 'field goal', 'kick', 'kicks off', 'pass', 'timeout'] # BEST

    print(f"vocab: {vocab}")

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1,4), min_df=1, max_df=1.0,)
    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()

    tsne = TSNE(n_components=2, learning_rate=50, perplexity=1, early_exaggeration=100, metric="cityblock", random_state=42) # BEST
    tsne = tsne.fit_transform(tfidf)
    
    highest_silh_score = -1
    params = {'eps': None, 'min_samples': None}

    for eps in [1, 5, 10, 25, 50, 75, 100,]:
        for min_samples in [10, 25, 50, 75, 100]:
            print(f"eps={eps} | min_samples={min_samples}")

            # dbscan = DBSCAN(eps=1, min_samples=50) # BEST
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(tsne)
            
            cluster_labels = dbscan.labels_

            no_clusters = len(np.unique(cluster_labels))
            no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)

            print('Estimated no. of clusters: %d' % no_clusters)
            print('Estimated no. of noise points: %d' % no_noise)

            silhouette_score = metrics.silhouette_score(tsne, cluster_labels)
            
            if silhouette_score > highest_silh_score:
                highest_silh_score = silhouette_score
                params['eps'] = eps
                params['min_samples'] = min_samples

            print(f"silhouette_score: {silhouette_score}")

            print("=================================================================================")

    print(f"Highest Silhouette Score: {highest_silh_score}\nBest Parameters: {params}")
    
    dbscan = DBSCAN(**params)
    dbscan.fit(tsne)
    
    cluster_labels = dbscan.labels_

    no_clusters = len(np.unique(cluster_labels))
    no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)

    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)

    show_cluster_examples(df, cluster_labels, no_clusters, num_examples=num_examples, show_top_terms=False, model=None, tfidf_vect=None)

    colors = list(map(lambda x: x, cluster_labels))
    plt.scatter(x=tsne[:, 0], y=tsne[:, 1], c=colors, marker="o", picker=True)
    plt.show()
    
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

def get_verbs_in_corpus(df):
    # Load the spaCy English model
    nlp = spacy.load('en_core_web_sm')

    seen_verbs = set()
    seen_pos = set()

    for play in df['play_text']:
        doc = nlp(play)
        
        for token in doc:
            if token.pos_ in ["VERB"]:
                # if not token.lemma_ in seen_verbs:
                #     print(f"Play for verb=[{token.lemma_}]: {play}")
                
                # if token.lemma_ in ['wr', 'abort', 'offset', 'block', 'rough', 'start', 'accept', 'scramble', 'leave', 'return', '']:
                if token.lemma_ in ['wr']:
                    continue

                seen_verbs.add(token.lemma_)

                break
            
        seen_pos |= {token.pos_ for token in doc}

    print(f"seen_verbs: {seen_verbs}")
    print(f"seen_pos: {seen_pos}")

    return seen_verbs


if __name__ == "__main__":
    # from matplotlib import colormaps

    # print(list(colormaps))
    df = get_pbp_dataframe()
    df = preprocess_play_text(df)

    # # kmean_cluster_2d()
    # # kmeans_cluster_3d(df, num_examples=10)
    # # tsne_cluster_3d(df, num_examples=10)
    # # tsne_cluster_2d(df, num_examples=10)
    dbscan_2d(df, num_examples=20, params={'eps': 1, 'min_samples': 10})
    # dbscan_3d(df, num_examples=10)

    # get_verbs_in_corpus(df)