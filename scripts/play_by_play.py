import pandas as pd
import json
import glob
import warnings
import re
import spacy
import os


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

print(f"script_dir: {script_dir}")

nlp = spacy.load("en_core_web_sm")

warnings.simplefilter(action='ignore', category=FutureWarning)

position_type_dict = {
    'offense': {'QB', 'RB', 'WR', 'TE', 'OL',},
    'defense': {'LB', 'DB', 'DL',},
    'special_teams': {'PR', 'P', 'K',},
}

# inverting aliases dictionary
inv_position_type_dict = dict()

for i in position_type_dict:
    for j in position_type_dict[i]:
        inv_position_type_dict[j] = i

print(f"inv_position_type_dict:\n{inv_position_type_dict}")

def get_play_by_play_df():
    df = pd.read_csv(script_dir+"../data/play_by_play.csv")
    return df

def get_player_pos_dict():
    df = pd.read_csv(script_dir+"../data/players.csv")
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
            pid = player_tag_dict[i]

            if pid in player_pos_dict:
                new_text = new_text.replace(i, player_pos_dict[pid])
            else:
                new_text = new_text.replace(i, f"<PLAYER_NOT_FOUND:{pid}>")

    new_text = re.sub(r'[\|/|)|(|"|#|:]', r'', new_text) # special_char

    return new_text

def preprocess_play_text(df):
    player_pos_dict = get_player_pos_dict()

    df['play_text'] = df['Detail'].apply(prepocess_text, args=(player_pos_dict,),)
    df = df[~((df['play_text'].str.contains("POSITION_NOT_FOUND")) | (df['play_text'].str.contains("PLAYER_NOT_FOUND")))]

    return df

def transform_play_test(x):
    penalty_play = False

    if "penalty" in x.lower():
        penalty_play = True
    
    doc = nlp(x)
    sentences = doc.sents

    sentences = [i.text for i in sentences]
        
    found_pos_by_sent = []
    processed_words = []
    for sent in sentences:
        # remove periods from text
        sent = sent.replace(".", "")
        words = sent.split()

        found = [] # found position

        processed_words.append(words)

        for i in words:
            if i not in inv_position_type_dict:
                continue
            
            found.append(i)

        found_pos_by_sent.append(found)

    return (found_pos_by_sent, penalty_play, len(sentences))

def off_def_players_interactions(x):
    off_player = None
    def_players = []

    for i in x[::-1]:
        if inv_position_type_dict[i] == 'defense':
            def_players.append(i)
        elif inv_position_type_dict[i] == 'offense':
            off_player = i
            break

    return (off_player, def_players)

def get_player_interaction_tuples():
    print("RUNNING get_player_interaction_tuples()")

    df = get_play_by_play_df()


    # dropping rows without Details
    df = df[~df['Detail'].isna()]

    print("Calling preprocess_play_text()")
    df = preprocess_play_text(df)

    print("Calling transform_play_test()")
    df['players_in_text'] = df['play_text'].apply(transform_play_test)
    df[['players_in_text', 'penalty_on_play', 'num_sentences']] = pd.DataFrame(df['players_in_text'].tolist(), index=df.index)

    # sort by num_sentences
    df = df.sort_values(by=['num_sentences'], ascending=False)

    print("Exploding df")
    # explode play_text column to get one row for each sentence in a play's description
    df = df.explode('players_in_text')

    df['num_players'] = df['players_in_text'].str.len()

    print("Calling off_def_players_interactions()")
    df['off_def_interactions'] = df['players_in_text'].apply(off_def_players_interactions)
    df[['offensive_player', 'defensive_players']] = pd.DataFrame(df['off_def_interactions'].tolist(), index=df.index)
    df['num_players'] = df['players'].apply(lambda x: len(re.findall(r"\'.*?\'", x)))

    # drop row with no offensive player
    df = df[~df['offensive_player'].isna()]

    def func(group):
        seen_dict = dict()

        value_list = group['defensive_players'].values.tolist()
        
        for sub_list in value_list:
            for val in sub_list:
                if val not in seen_dict:
                    seen_dict[val] = 1
                else:
                    seen_dict[val] += 1

        return seen_dict
    
    print("Groupby and Apply")

    results = df.groupby(['offensive_player']).apply(func)
    results = results.to_frame(name='data')

    results['data'].apply(lambda x: x)
    results = results.to_dict()['data']

    totals_dict = dict()
    
    # get totals for each 
    for i in results:
        totals_dict[i] = 0
        for j in results[i]:
            totals_dict[i] += results[i][j]
    
    interaction_prob_dict = dict()

    for off_pos in results:
        interaction_prob_dict[off_pos] = dict()

        for def_pos in position_type_dict['defense']:
            if def_pos in results[off_pos]:
                prob = results[off_pos][def_pos] / totals_dict[off_pos]
            else:
                prob = 0

            interaction_prob_dict[off_pos][def_pos] = prob

    prob_df = pd.DataFrame.from_dict(interaction_prob_dict).T

    print(prob_df)

    prob_df.to_csv(script_dir+"../data/interaction_prob.csv", index=True)


if __name__ == "__main__":
    # combine_jsons()
    get_player_interaction_tuples()
