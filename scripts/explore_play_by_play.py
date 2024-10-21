import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import time
import datetime
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from collections import defaultdict
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import umap
import altair as alt
from PIL import Image
import plotly.express as px
from wordcloud import WordCloud

alt.data_transformers.disable_max_rows()
alt.renderers.enable("html")

evaluations = []
evaluations_std = []
def fit_and_evaluate(km, X, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time.time()
        km.fit(X)
        train_times.append(time.time() - t0)
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    train_times = np.asarray(train_times)

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)

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

def remove_player_references(x, pattern, replace_position=True):
    if replace_position:
        # substitute positions with generic tag
        new_text = re.sub(pattern+"|\d+", "position", x)
    else:
        # substitute positions with generic tag
        new_text = re.sub(pattern+"|\d+", "", x)

    # print(f"new_text: {new_text}")

    return new_text

def remove_numbers(x):
    x = re.sub("[\d+]", "", x)
    return x

def remove_unhelpful_words(x, remove_terms):
    for term in remove_terms:
        x = re.sub("\\b"+term+"\\b", "", x)
    return x

def plot_nmf_exploration(df):
    print(f"Initial df size: {df.shape}")
    remove_terms = ['by', 'for', 'yards', 'yard', 'and', 'to', 'gain',]

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])
    
    df['play_text'] = df['play_text'].apply(lambda x: remove_unhelpful_words(x,remove_terms=remove_terms))
    df['new_play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=False))

    # print(df['new_play_text'])

    # vocab = ['punt', 'punts', 'field goal', 'kick', 'kicks off', 'pass', 'timeout'] # BEST
    vocab = None

    print(f"vocab: {vocab}")

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=1.0,)
    tfidf = tfidf_vect.fit_transform(df['new_play_text'].values).toarray()
    tfidf_feature_names = tfidf_vect.get_feature_names_out()

    n_components = 11
    for n_components in [6,7,8,9,10,11]:
        print(f"n_components: {n_components}")

        nmf = NMF(n_components=n_components, init="nndsvd", random_state=42)
        W = nmf.fit_transform(tfidf)
        H = nmf.components_

        top = 5
        topic_index_max = n_components

        all_terms = []
        top_topic_term = dict()
        all_top_terms = []

        for topic_index in range(0, topic_index_max):
            top_indices = np.argsort(H[topic_index, :])[::-1]
            top_terms = []
            for term_index in top_indices[0:top]:
                top_terms.append(tfidf_feature_names[term_index])
                all_terms.append(tfidf_feature_names[term_index])
            
            all_top_terms.append(set(top_terms))
            print("topic ", topic_index, top_terms)

        print(f"W.shape:\n{W.shape}")
        # print(f"W:\n{W}")
        print(f"W:\n{pd.DataFrame(W)}")

    print("==============================================")

def plot_word_clouds(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(8,5), sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]
        weights_normed = [i/sum(weights) for i in weights]
        data = pd.DataFrame(weights_normed, index=top_features)[0].sort_values(ascending=False)
        wc = WordCloud(width=2000, height=2000).generate_from_frequencies(data)
        
        ax = axes[topic_idx]
        ax.imshow(wc)        
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 8})
        ax.axis('off')
        ax.tick_params(axis="both", which="major", labelsize=6)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=12)

    # plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.tight_layout()
    plt.show()

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(6, 4), sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.5)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 8})
        ax.tick_params(axis="both", which="major", labelsize=6)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=12)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

def plot_nmf(df, vocab, text_col, n_components, top_n=5, remove_players_refs=False, remove_terms=None, replace_position=False):
    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])
    
    # df['play_text'] = df['play_text'].apply(lambda x: remove_numbers(x))

    if remove_terms is not None:
        df['play_text'] = df['play_text'].apply(lambda x: remove_unhelpful_words(x,remove_terms=remove_terms))

    if remove_players_refs:
        df['new_play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=replace_position))

    print(f"vocab: {vocab}")

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=0.5,)
    tfidf = tfidf_vect.fit_transform(df[text_col].values).toarray()
    tfidf_feature_names = tfidf_vect.get_feature_names_out()

    print(f"n_components: {n_components}")

    nmf = NMF(n_components=n_components, init="nndsvda", beta_loss="frobenius", random_state=42)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    topic_index_max = n_components

    all_terms = []
    top_topic_term = dict()
    all_top_terms = []

    for topic_index in range(0, topic_index_max):
        top_indices = np.argsort(H[topic_index, :])[::-1]
        top_terms = []
        for term_index in top_indices[0:top_n]:
            top_terms.append(tfidf_feature_names[term_index])
            all_terms.append(tfidf_feature_names[term_index])
        
        all_top_terms.append(set(top_terms))
        print("topic ", topic_index, top_terms)

    tfidf_feature_names = tfidf_feature_names
    plot_top_words(
        nmf, tfidf_feature_names, 3, "Topics In Play-By-Play Data"
    )
    plot_word_clouds(
        nmf, tfidf_feature_names, 5, "Topics Word Cloud"
    )
    # print(f"W.shape:\n{W.shape}")
    # print(f"W:\n{W}")
    # print(f"W:\n{pd.DataFrame(W)}")

    print("==============================================")

def plot_nmf_sens_anal(df, vocab, text_col, n_components, top_n=5, remove_players_refs=False, remove_terms=None, replace_position=False):
    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])
    
    # df['play_text'] = df['play_text'].apply(lambda x: remove_numbers(x))

    if remove_terms is not None:
        df['play_text'] = df['play_text'].apply(lambda x: remove_unhelpful_words(x,remove_terms=remove_terms))

    if remove_players_refs:
        df['new_play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=replace_position))

    print(f"vocab: {vocab}")

    # tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=0.5,)
    tfidf_vect = TfidfVectorizer()
    tfidf = tfidf_vect.fit_transform(df[text_col].values).toarray()
    tfidf_feature_names = tfidf_vect.get_feature_names_out()

    print(f"n_components: {n_components}")

    nmf = NMF(n_components=n_components, init="nndsvda", beta_loss="frobenius", random_state=42)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    topic_index_max = n_components

    all_terms = []
    top_topic_term = dict()
    all_top_terms = []

    for topic_index in range(0, topic_index_max):
        top_indices = np.argsort(H[topic_index, :])[::-1]
        top_terms = []
        for term_index in top_indices[0:top_n]:
            top_terms.append(tfidf_feature_names[term_index])
            all_terms.append(tfidf_feature_names[term_index])
        
        all_top_terms.append(set(top_terms))
        print("topic ", topic_index, top_terms)

    tfidf_feature_names = tfidf_feature_names
    plot_top_words(
        nmf, tfidf_feature_names, 3, "Topics In Play-By-Play Data"
    )
    # print(f"W.shape:\n{W.shape}")
    # print(f"W:\n{W}")
    # print(f"W:\n{pd.DataFrame(W)}")

    print("==============================================")

def lda_tsne_graph_exploration(df):
    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])

    df['new_play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=False))

    vocab1 = None
    vocab2 = [i.lower() for i in positions_df['adj_pos'].unique()]
    vocab3 = ['punt', 'punts', 'field goal', 'kick', 'kicks off', 'pass', 'timeout'] # BEST
    vocab4 = ['complete', 'short', 'pass', 'middle', 'touchdown'] + \
        ['left', 'guard', 'no', 'deep'] + ['middle', 'up', 'no'] + \
            ['incomplete', 'intended', 'pass', 'short', 'defended'] + \
                ['right', 'guard', 'no', 'returned'] + ['end', 'right', 'left', 'touchdown', 'scrambles'] + \
                    ['no', 'good', 'field', 'goal']
    vocab4 = list(set(vocab4))

    for vocab in [vocab4]:
        print(f"Vocab: {vocab}")

        if vocab == vocab2:
            text_col = 'play_text'
        else:
            text_col = 'new_play_text'

        tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=1.0,)
        tfidf = tfidf_vect.fit_transform(df[text_col].values)

        for n_components in [8,9,10]:
            print(f"\tn_components={n_components}")
            lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
            lda.fit(tfidf)

            print()

            arr = pd.DataFrame(lda.transform(tfidf)).fillna(0).values

            # Dominant topic number in each doc
            topic_num = np.argmax(arr, axis=1)

            for dim in [2,3]:
                if dim == 2:
                    # tSNE Dimension Reduction
                    tsne_model = TSNE(n_components=dim, verbose=1, random_state=0, angle=.99, init='pca')
                    tsne_lda = tsne_model.fit_transform(arr)

                    plt.scatter(tsne_lda[:, 0], tsne_lda[:, 1], c=topic_num, cmap='rainbow')
                    plt.show()
                elif dim == 3:
                    # tSNE Dimension Reduction
                    tsne_model = TSNE(n_components=dim, verbose=1, random_state=0, angle=.99, init='pca')
                    tsne_lda = tsne_model.fit_transform(arr)

                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(projection='3d')

                    ax.scatter(tsne_lda[:, 0], tsne_lda[:, 1], tsne_lda[:, 2], c=topic_num, cmap='rainbow')
                    plt.show()

        print("===========================================")

def lda_tsne_graph(df, vocab, text_col, n_components, dim='2d', remove_players_refs=False, replace_position=False):
    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])

    if remove_players_refs:
        df['new_play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=replace_position))

    print(f"Vocab: {vocab}")

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=0.5,)
    tfidf = tfidf_vect.fit_transform(df[text_col].values)

    print(f"\tn_components={n_components}")

    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda.fit(tfidf)

    arr = pd.DataFrame(lda.transform(tfidf)).fillna(0).values

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    if dim == '2d':
        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(arr)

        plt.scatter(tsne_lda[:, 0], tsne_lda[:, 1], c=topic_num, cmap='rainbow')
        plt.show()
    elif dim == '3d':
        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=3, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(arr)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')

        ax.scatter(tsne_lda[:, 0], tsne_lda[:, 1], tsne_lda[:, 2], c=topic_num, cmap='rainbow')
        plt.show()

    print("===========================================") 

def plot_tree_density_est_graph(df):
    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])
    vocab = None

    print(df)

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, ngram_range=(1,1), min_df=1, max_df=1.0,)
    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()
    tfidf_feature_names = tfidf_vect.get_feature_names_out()

    # lda = LatentDirichletAllocation(n_components=4)
    # lda.fit(tfidf)

    # tfidf = pd.DataFrame(lda.transform(tfidf)).fillna(0).values

    for n_components in [10]:
        gm = GaussianMixture(n_components=n_components).fit(tfidf)

        centers = gm.means_

        print(tfidf)
        print(centers)

        plt.figure(figsize=(8,6))
        plt.scatter(tfidf[:,0], tfidf[:,1], label='data')
        plt.scatter(centers[:,0], centers[:,1], c='r', label='centers')
        plt.legend()
        plt.show()

def get_word_freq_chart(df):
    print(f"Initial df size: {df.shape}")
    print(df)

    vectorizer = CountVectorizer(stop_words='english')
    X_counts = vectorizer.fit_transform(df['play_text'].values)

    # Sum up the counts of each word in the vocabulary
    word_counts = X_counts.toarray().sum(axis=0)
    word_freq = [(word, word_counts[idx]) for word, idx in vectorizer.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

    # Plot the top 30 most frequent words
    words = [wf[0] for wf in word_freq[:30]]
    counts = [wf[1] for wf in word_freq[:30]]

    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.xticks(rotation=90)
    plt.title("Top 30 Words Frequency")
    plt.show()

def get_word_freq_chart_processed(df):
    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])
    df['play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=False))
    df['play_text'] = df['play_text'].apply(lambda x: remove_numbers(x))

    print(df)

    vectorizer = CountVectorizer(stop_words='english')
    X_counts = vectorizer.fit_transform(df['play_text'].values)

    # Sum up the counts of each word in the vocabulary
    word_counts = X_counts.toarray().sum(axis=0)
    word_freq = [(word, word_counts[idx]) for word, idx in vectorizer.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

    # Plot the top 30 most frequent words
    words = [wf[0] for wf in word_freq[:30]]
    counts = [wf[1] for wf in word_freq[:30]]
    
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.xticks(rotation=90)
    plt.title("Top 30 Words Frequency")
    plt.show()

def get_dendogram(df):
    df = df[['play_text']]

    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])
    df['play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=False))
    df['play_text'] = df['play_text'].apply(lambda x: remove_numbers(x))
    df = df.reset_index(drop=True)

    vocab = None
    new_vocab = [
        'complete', 'short', 'pass', 'middle', 'left', 
        'deep', 'up', 'incomplete', 'punt', 'punts',
        'intended', 'defended', 'right',
        'end', 'scrambles', 'extra',
        'no', 'good', 'field', 'goal', 'penalty', 'point', 
    ]

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=0.9,)
    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()
    tfidf_feature_names = tfidf_vect.get_feature_names_out()

    tfidf = StandardScaler().fit(tfidf).transform(tfidf)
    
    # Identify non-zero rows
    zero_rows = np.all(tfidf == 0, axis=1)
    df = df[~zero_rows]
    tfidf = tfidf[~zero_rows]

    def plot_clusters(X_pca, labels, title):
        plt.figure(figsize=(10, 5))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=2)
        plt.title(title)
        plt.show()

    ################## kMeans ##################
    # Apply K-Means with a predetermined number of clusters
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf)

    # Get the cluster assignments
    labels_kmeans = kmeans.labels_
    df['cluster_kmeans'] = labels_kmeans


    ################## Agglomerative Hierarchical Clustering ##################
    # Apply Agglomerative Hierarchical Clustering
    agglo = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='average')
    agglo.fit(tfidf)

    # Get the cluster assignments
    labels_agglo = agglo.labels_
    df['cluster_agglo'] = labels_agglo

    # Reduce dimensions to 2 for visualization
    X_pca = PCA(n_components=2).fit_transform(tfidf)

    tfidf_df = pd.DataFrame(tfidf, columns=tfidf_feature_names)
    tfidf_df['play_text'] = df['play_text']
    tfidf_df = tfidf_df.set_index(['play_text'])

    linkage_data = linkage(tfidf_df, method='complete', metric='cosine')
    R = dendrogram(linkage_data, p=4, labels=tfidf_df.index, truncate_mode='level', count_sort='descending', distance_sort='ascending')

    # print([tfidf_feature_names[int(i.strip(")("))] for i in R['ivl']])
    
    plt.title("Hierarchical Clustering Dendogram (truncated)")
    plt.xticks([])
    plt.show()

    # plot_clusters(X_pca, labels_kmeans, 'K-Means Clusters')
    # plot_clusters(X_pca, labels_agglo, 'Agglomerative Clustering Clusters')

def get_elbow_plot(df):
    df = df[['play_text']]

    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])
    df['play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=False))
    df['play_text'] = df['play_text'].apply(lambda x: remove_numbers(x))
    df = df.reset_index(drop=True)

    vocab = None
    new_vocab = [
        'complete', 'short', 'pass', 'middle', 'left', 
        'deep', 'up', 'incomplete', 'punt', 'punts',
        'intended', 'defended', 'right',
        'end', 'scrambles', 'extra',
        'no', 'good', 'field', 'goal', 'penalty', 'point', 
    ]

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=0.5,)
    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()
    tfidf_feature_names = tfidf_vect.get_feature_names_out()

    tfidf = StandardScaler().fit(tfidf).transform(tfidf)
    
    # Identify non-zero rows
    zero_rows = np.all(tfidf == 0, axis=1)
    df = df[~zero_rows]
    tfidf = tfidf[~zero_rows]

    Sum_of_squared_distances = []
    K = range(1,251)
    
    for num_clusters in K :
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(tfidf)
        Sum_of_squared_distances.append(kmeans.inertia_)
    
    plt.plot(K, Sum_of_squared_distances,'bx-')             
    plt.xlabel("Clusters") 
    plt.ylabel('Sum of squared distances/Inertia') 
    plt.title('Elbow Method For Optimal k')
    plt.show()

def get_silhouette_plot(df):
    df = df[['play_text']]

    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])
    df['play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=False))
    df['play_text'] = df['play_text'].apply(lambda x: remove_numbers(x))
    df = df.reset_index(drop=True)

    vocab = None
    new_vocab = [
        'complete', 'short', 'pass', 'middle', 'left', 
        'deep', 'up', 'incomplete', 'punt', 'punts',
        'intended', 'defended', 'right',
        'end', 'scrambles', 'extra',
        'no', 'good', 'field', 'goal', 'penalty', 'point', 
    ]

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=0.5,)
    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()
    tfidf_feature_names = tfidf_vect.get_feature_names_out()
    tfidf = StandardScaler().fit(tfidf).transform(tfidf)

    range_n_clusters = [2, 210]
    # range_n_clusters = list(range(210, 250))

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(tfidf) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(tfidf)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(tfidf, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(tfidf, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(tfidf)

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            tsne_lda[:, 0], tsne_lda[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()

def get_umap_plot(df):
    df = df[['play_text']]

    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])
    df['play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=False))
    df['play_text'] = df['play_text'].apply(lambda x: remove_numbers(x))
    df = df.reset_index(drop=True)

    vocab = None
    new_vocab = [
        'complete', 'short', 'pass', 'middle', 'left', 
        'deep', 'up', 'incomplete', 'punt', 'punts',
        'intended', 'defended', 'right',
        'end', 'scrambles', 'extra',
        'no', 'good', 'field', 'goal', 'penalty', 'point', 
    ]

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=0.5,)
    tfidf = tfidf_vect.fit_transform(df['play_text'].values).toarray()
    tfidf_feature_names = tfidf_vect.get_feature_names_out()
    tfidf = StandardScaler().fit(tfidf).transform(tfidf)
    
    mapper = umap.UMAP(metric='euclidean', random_state=42).fit(tfidf)
    embedding = mapper.transform(tfidf)
    
    print(embedding)

    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.show()

def lda_tsne_graph_plotly(df, vocab, text_col, n_components, dim='2d', remove_players_refs=False, replace_position=False):
    df = df.dropna(subset=["play_text"])

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])

    if remove_players_refs:
        df['play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=replace_position))

    df['play_text'] = df['play_text'].apply(lambda x: remove_numbers(x))

    print(f"Vocab: {vocab}")

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1,1), min_df=1, max_df=0.5,)
    tfidf = tfidf_vect.fit_transform(df[text_col].values)
    tfidf_feature_names = tfidf_vect.get_feature_names_out()

    print(f"\tn_components={n_components}")

    nmf = NMF(n_components=n_components, tol=1e-3, init="nndsvda", beta_loss="frobenius", random_state=42)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_
    
    topic_term_dict = dict()
    n_top_words = 3
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[-n_top_words:][::-1]
        top_features = tfidf_feature_names[top_features_ind]
        weights = topic[top_features_ind]
        
        topic_term_dict[topic_idx] = list(top_features)
    
    print(topic_term_dict)

    topic_num = np.argmax(W, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_nmf = tsne_model.fit_transform(W)

    res_df = pd.DataFrame(tsne_nmf, columns=['x', 'y'])
    res_df['topic'] = topic_num
    res_df['topic_terms'] = res_df['topic'].apply(lambda x: topic_term_dict[x])
    res_df['topic'] = res_df['topic'].astype(str)

    print(f"res_df:\n{res_df}")

    fig = px.scatter(res_df, x='x', y='y', color="topic", hover_data=['topic_terms'])
    fig.show()

    print("===========================================") 

def lda_tsne_graph_plotly_sens_anal(df, vocab, text_col, n_components, dim='2d', remove_players_refs=False, replace_position=False):
    df = df.dropna(subset=["play_text"])

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])

    if remove_players_refs:
        df['play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=replace_position))

    df['play_text'] = df['play_text'].apply(lambda x: remove_numbers(x))

    print(f"Vocab: {vocab}")

    tfidf_vect = TfidfVectorizer()
    tfidf = tfidf_vect.fit_transform(df[text_col].values)
    tfidf_feature_names = tfidf_vect.get_feature_names_out()

    print(f"\tn_components={n_components}")

    nmf = NMF(n_components=n_components, init="nndsvda", beta_loss="frobenius", random_state=42)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_
    
    topic_term_dict = dict()
    n_top_words = 3
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[-n_top_words:][::-1]
        top_features = tfidf_feature_names[top_features_ind]
        weights = topic[top_features_ind]
        
        topic_term_dict[topic_idx] = list(top_features)
    
    print(topic_term_dict)

    topic_num = np.argmax(W, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_nmf = tsne_model.fit_transform(W)

    res_df = pd.DataFrame(tsne_nmf, columns=['x', 'y'])
    res_df['topic'] = topic_num
    res_df['topic_terms'] = res_df['topic'].apply(lambda x: topic_term_dict[x])
    res_df['topic'] = res_df['topic'].astype(str)

    print(f"res_df:\n{res_df}")

    fig = px.scatter(res_df, x='x', y='y', color="topic", hover_data=['topic_terms'])
    fig.show()
    
    print("===========================================") 

if __name__ == "__main__":
    # grab a sample of pbp data
    df = get_pbp_dataframe().head(20000)
    df = preprocess_play_text(df)

    # plot_tree_density_est_graph(df)
    # get_word_freq_chart(df)
    # get_word_freq_chart_processed(df)
    # get_dendogram(df)
    # get_elbow_plot(df)
    # get_silhouette_plot(df)
    # get_umap_plot(df)



    """
    Let's first use LDA and T-SNE to visualize the raw data
    """
    # Use LDA and TSNE to visualize PBP data
    # lda_tsne_graph(
    #     df, 
    #     vocab=None, 
    #     text_col="play_text", 
    #     n_components=12, 
    #     dim='2d', 
    #     remove_players_refs=False,
    #     replace_position=False
    # )

    """
    Now let's use NMF to find the top terms in each topic (type of play)
    """
    plot_nmf(
        df, 
        vocab=None, 
        text_col='new_play_text', 
        n_components=10,
        top_n=10, 
        remove_players_refs=True, 
        remove_terms=['on', 'no', 'to', 'touchdown', 'yard'],
        replace_position=False
    )

    # plot_nmf_sens_anal(
    #     df, 
    #     vocab=None, 
    #     text_col='new_play_text', 
    #     n_components=10,
    #     top_n=10, 
    #     remove_players_refs=True, 
    #     remove_terms=None,
    #     replace_position=False
    # )

    # """
    # Looking at the top terms for each topic, we can start to get a sense
    # of the different types of plays 
    # (e.g. complete/incomplete passes, middle/left/right runs, field goals, punts).

    # Combining these words and using that as the vocabulary when running the T-SNE algorithm may 
    # yield interesting clustering results
    # """
    
    new_vocab = [
        'complete', 'short', 'pass', 'middle', 'left', 
        'deep', 'up', 'incomplete', 'punt', 'punts',
        'intended', 'defended', 'right',
        'end', 'scrambles', 'extra',
        'no', 'good', 'field', 'goal', 'penalty', 'point',
        'fumble', 'fumbles', 'tackle', 'tackled', 
    ]

    # Use LDA and TSNE to visualize PBP data
    # lda_tsne_graph(
    #     df, 
    #     vocab=new_vocab, 
    #     text_col="play_text", 
    #     n_components=12, 
    #     dim='2d', 
    #     remove_players_refs=False,
    #     replace_position=False
    # )

    

    """
    We can see from this graph with constrained vocab that there is more structure 
    and denser clusters. This highlights the structured nature of the play by play data.
    We can exploit this structure to extract additional features from the play by play data.
    For example, we can add a play type feature which has values like pass, run, field goal, etc.
    """
    

    # lda_tsne_graph_plotly(
    #     df, 
    #     vocab=new_vocab, 
    #     # vocab=None, 
    #     text_col="play_text", 
    #     n_components=10, 
    #     dim='2d', 
    #     remove_players_refs=True,
    #     replace_position=False
    # )

    # lda_tsne_graph_plotly_sens_anal(
    #     df, 
    #     # vocab=new_vocab, 
    #     vocab=None, 
    #     text_col="play_text", 
    #     n_components=250, 
    #     dim='2d', 
    #     remove_players_refs=True,
    #     replace_position=False
    # )