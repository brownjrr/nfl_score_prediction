import pandas as pd
import json
import glob
import warnings
import re
import spacy

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

with open("../data/abbrev_map.json", "r") as f:
    abbrev_map = json.load(f)

print(abbrev_map)

def combine_jsons():
    """
    combining all play by play data (json to pandas df)
    """
    column_len_set = set()
    column_set = set()
    dfs = []
    for df_json_file in [i.replace("\\", "/") for i in glob.glob("C:/Users/Robert Brown/OneDrive/pbp_data_files/dataframes/*.txt")]:
        with open(df_json_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                json_obj = json.loads(line)

                link = json_obj['index']
                data = json_obj['data']

                df = pd.read_json(data)

                for j, col in enumerate(df.columns):
                    if col in abbrev_map:
                        team1_idx = j
                        team1_name = df.columns[j]

                        team2_idx = j+1
                        team2_name = df.columns[j+1]
                        break
                    
                df = df.rename(columns={team1_name: "team_1", team2_name: "team_2"})
                df['team_1_name'] = abbrev_map[team1_name]
                df['team_2_name'] = abbrev_map[team2_name]
                df['link'] = link
                df['boxscore_id'] = df['link'].str.split("/").str[-1].str.replace(".htm", "")
                
                column_len_set.add(len(df.columns))
                column_set |= set(df.columns)

                dfs.append(df)

    df = pd.concat(dfs).drop_duplicates()

    print(df)
    print(f"Num Unique Links: {len(df['link'].unique())}")

    pattern = r'<a href.*?/a>'
    df['players'] = df['Detail'].apply(lambda x: [i for i in re.findall(pattern, x) if "/players/" in i])

    # discard plays where no players were found AND
    # discard plays where only 1 player is part of the interaction
    # (these are often kicking/punting plays)
    # df = df[df['player'].str.len() > 1]
    df['players'] = df['players'].apply(lambda x: [re.findall(r"\".*?\"", i)[0].replace('"', '') for i in x])
    # df = df.explode('player')
    df['players'] = df['players'].apply(lambda x: [i.split("/")[-1].replace(".htm", "") for i in x])

    # creating event_date columns
    df['event_date'] = pd.to_datetime(df['boxscore_id'].str[:-4])

    # converting team names to team abbreviations for consistency with other tables
    # df['team_1_name'] = df['team_1_name'].apply(lambda x: abbrev_map[x])
    # df['team_2_name'] = df['team_2_name'].apply(lambda x: abbrev_map[x])
    
    print(f"Min Date: {df['event_date'].min()} - Max Date: {df['event_date'].max()}")
    print(df)

    df.to_csv("../data/play_by_play.csv", index=False)

    # print(f"column_len_set: {column_len_set}")
    # print(f"column_set: {column_set}")
    # print(f"column_set length: {len(column_set)}")

def get_play_by_play_df():
    df = pd.read_csv("../data/play_by_play.csv")
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

def get_all_interaction_probabilities():
    offensive_positions = pd.Series([i for i in inv_position_type_dict if inv_position_type_dict[i]=="offense"])
    offensive_positions = offensive_positions.to_frame(name="pos_1")

    defensive_positions = pd.Series([i for i in inv_position_type_dict if inv_position_type_dict[i]=="defense"])
    defensive_positions = defensive_positions.to_frame(name="pos_2")

    df = pd.merge(offensive_positions, defensive_positions, how='cross')
    temp_df = df[['pos_2', 'pos_1']].rename(columns={'pos_1': 'pos_2', 'pos_2': 'pos_1'})
    df = pd.concat([df, temp_df])

    pbp_df = get_play_by_play_df()

    def get_interaction_prob(row):
        print(f"Position 1: {row['pos_1']} | Position 2: {row['pos_2']}")
        
        pos1 = row['pos_1']
        pos2 = row['pos_2']

        temp_df = pbp_df.groupby(['Quarter', 'Time', 'Down', 'ToGo', 'Location', 'link_x', "team_1", "team_2"])['position'].apply(list).reset_index()
        pos1_rows = temp_df[[(pos1 in x) for x in temp_df.position]].shape[0]
        num_p1_p2_rows = temp_df[[(pos1 in x) and (pos2 in x) for x in temp_df.position]].shape[0]
        
        return num_p1_p2_rows / pos1_rows

    df['interaction_prob'] = df.apply(lambda row: get_interaction_prob(row), axis=1)
    
    print(df)

    df.to_csv("../data/interaction_prob.csv", index=False)

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
    df = get_play_by_play_df()

    # dropping rows without Details
    df = df[~df['Detail'].isna()]

    df = preprocess_play_text(df)

    df['players_in_text'] = df['play_text'].apply(transform_play_test)
    df[['players_in_text', 'penalty_on_play', 'num_sentences']] = pd.DataFrame(df['players_in_text'].tolist(), index=df.index)

    # sort by num_sentences
    df = df.sort_values(by=['num_sentences'], ascending=False)

    # explode play_text column to get one row for each sentence in a play's description
    df = df.explode('players_in_text')

    df['num_players'] = df['players_in_text'].str.len()

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

    prob_df.to_csv("../data/interaction_prob.csv", index=False)


if __name__ == "__main__":
    # combine_jsons()
    # get_play_by_play_df()
    # get_interaction_prob(get_play_by_play_df(), "RB", "DB") # deprecated
    # get_all_interaction_probabilities() # deprecated
    get_player_interaction_tuples()
