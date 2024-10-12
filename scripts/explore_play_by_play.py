import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import datetime
from sklearn.decomposition import LatentDirichletAllocation, NMF


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

def plot_nmf(df, vocab, text_col, n_components, top_n=5, remove_players_refs=False, remove_terms=None, replace_position=False):
    print(f"Initial df size: {df.shape}")

    positions_df = pd.read_csv("../data/positions.csv")
    positions_set = set(positions_df['adj_pos'].unique())
    pattern = "|".join(["\\b"+i+"\\b" for i in positions_set])
    
    # df['play_text'] = df['play_text'].apply(lambda x: remove_numbers(x,remove_terms=remove_terms))

    if remove_terms is not None:
        df['play_text'] = df['play_text'].apply(lambda x: remove_unhelpful_words(x,remove_terms=remove_terms))

    if remove_players_refs:
        df['new_play_text'] = df['play_text'].apply(lambda x: remove_player_references(x, pattern, replace_position=replace_position))

    print(f"vocab: {vocab}")

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=1.0,)
    tfidf = tfidf_vect.fit_transform(df[text_col].values).toarray()
    tfidf_feature_names = tfidf_vect.get_feature_names_out()

    print(f"n_components: {n_components}")

    nmf = NMF(n_components=n_components, init="nndsvd", random_state=42)
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

    tfidf_vect = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), min_df=1, max_df=1.0,)
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

if __name__ == "__main__":
    # grab a sample of pbp data
    df = get_pbp_dataframe().head(10000)
    df = preprocess_play_text(df)

    """
    Let's first use LDA and T-SNE to visualize the raw data
    """
    # Use LDA and TSNE to visualize PBP data
    lda_tsne_graph(
        df, 
        vocab=None, 
        text_col="play_text", 
        n_components=10, 
        dim='2d', 
        remove_players_refs=False,
        replace_position=False
    )

    """
    Now let's use NMF to find the top terms in each topic (type of play)
    """
    plot_nmf(
        df, 
        vocab=None, 
        text_col='new_play_text', 
        n_components=7,
        top_n=7, 
        remove_players_refs=True, 
        remove_terms=None,
        replace_position=False
    )

    """
    Looking at the top terms for each topic, we can start to get a sense
    of the different types of plays 
    (e.g. complete/incomplete passes, middle/left/right runs, field goals, punts).

    Combining these words and using that as the vocabulary when running the T-SNE algorithm may 
    yield interesting clustering results
    """
    
    new_vocab = [
        'complete', 'short', 'pass', 'middle', 'left', 
        'deep', 'up', 'incomplete', 'punt', 'punts',
        'intended', 'defended', 'right',
        'end', 'scrambles', 'extra',
        'no', 'good', 'field', 'goal', 'penalty', 'point', 
    ]

    # Use LDA and TSNE to visualize PBP data
    lda_tsne_graph(
        df, 
        vocab=new_vocab, 
        text_col="play_text", 
        n_components=10, 
        dim='2d', 
        remove_players_refs=False,
        replace_position=False
    )

    """
    We can see from this graph with constrained vocab that there is more structure 
    and denser clusters. This highlights the structured nature of the play by play data.
    We can exploit this structure to extract additional features from the play by play data.
    For example, we can add a play type feature which has values like pass, run, field goal, etc.
    """
    