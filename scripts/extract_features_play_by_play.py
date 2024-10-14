import pandas as pd
import numpy as np
import re
import random
import os


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

print(f"script_dir: {script_dir}")

def get_pbp_dataframe():
    df = pd.read_csv(script_dir+"../data/play_by_play.csv").drop_duplicates(subset=['Detail']).dropna(subset=['Detail'])

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
            # print((type(i), i))
            # print(player_tag_dict[i])
            pid = player_tag_dict[i]

            if pid in player_pos_dict:
                new_text = new_text.replace(i, player_pos_dict[pid])
            else:
                new_text = new_text.replace(i, f"<PLAYER_NOT_FOUND:{pid}>")

    new_text = re.sub(r'[\|/|")|(|#]', r'', new_text) # special_char

    return new_text

def get_play_types():
    total = 0
    player_pos_dict = get_player_pos_dict()

    df = get_pbp_dataframe()
    df['play_text'] = df['Detail'].apply(prepocess_text, args=(player_pos_dict,),)

    print(f"Num Original Rows: {df.shape[0]}")
    print(f"Num Unique Games: {len(df['boxscore_id'].unique())}")

    # filtering out passing plays
    pass_terms = ['pass', 'sacked']
    pass_plays_df = df[(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(pass_terms))))]
    pass_plays_df['play_type'] = "passing_play"
    df = df[~(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(pass_terms))))]
    total += pass_plays_df.shape[0]

    print(f"pass_plays_df.shape: {pass_plays_df.shape}")
    
    # filtering out running plays
    run_terms = ['up', 'middle', 'rush', 'rushes', 'up the middle', 'left guard', 'right guard', 'left tackle', 'right tackle', 'left end', 'right end']
    run_plays_df = df[(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(run_terms))))]
    run_plays_df['play_type'] = "running_play"
    df = df[~(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(run_terms))))]
    total += run_plays_df.shape[0]

    print(f"run_plays_df.shape: {run_plays_df.shape}")

    # filtering out field goals
    field_goal_terms = ['field goal', 'extra point']
    field_goal_plays_df = df[(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(field_goal_terms))))]
    field_goal_plays_df['play_type'] = "field_goal"
    df = df[~(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(field_goal_terms))))]
    total += field_goal_plays_df.shape[0]

    print(f"field_goal_plays_df.shape: {field_goal_plays_df.shape}")

    # filtering out punts
    punt_terms = ['punts', 'punt']
    punt_plays_df = df[(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(punt_terms))))]
    punt_plays_df['play_type'] = "punt"
    df = df[~(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(punt_terms))))]
    total += punt_plays_df.shape[0]

    print(f"punt_plays_df.shape: {punt_plays_df.shape}")

    # filtering out kick offs
    kick_off_terms = ['kicks off', 'kicks']
    kickoff_plays_df = df[(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(kick_off_terms))))]
    kickoff_plays_df['play_type'] = "kick_off"
    df = df[~(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(kick_off_terms))))]
    total += kickoff_plays_df.shape[0]

    print(f"kickoff_plays_df.shape: {kickoff_plays_df.shape}")

    # grabbing any remaining running plays
    temp_df = df[df['play_text'].str.contains(" for ")]
    temp_df['play_type'] = "running_play"
    df = df[~df['play_text'].str.contains(" for ")]
    total += temp_df.shape[0]
    run_plays_df = pd.concat([run_plays_df, temp_df])

    print(f"Runs (for).shape: {temp_df.shape}")

    # filtering out timeouts
    timeout_terms = ['timeout']
    timeout_plays_df = df[(df['Detail'].str.lower().str.contains(r'\b(?:{})\b'.format('|'.join(timeout_terms))))]
    timeout_plays_df['play_type'] = "timeout"
    df = df[~(df['Detail'].str.lower().str.contains(r'\b(?:{})\b'.format('|'.join(timeout_terms))))]
    total += timeout_plays_df.shape[0]
    
    print(f"timeout_plays_df.shape: {timeout_plays_df.shape}")

    # filtering out spiked ball plays
    spike_terms = ['spike', 'spiked', 'kneel', 'kneels']
    spiked_plays_df = df[(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(spike_terms))))]
    spiked_plays_df['play_type'] = "spiked_play"
    df = df[~(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(spike_terms))))]
    total += spiked_plays_df.shape[0]
    
    print(f"spiked_plays_df.shape: {spiked_plays_df.shape}")

    # filtering out aborted plays
    abort_terms = ['abort', 'aborted']
    aborted_plays_df = df[(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(abort_terms))))]
    aborted_plays_df['play_type'] = "aborted_play"
    df = df[~(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(abort_terms))))]
    total += aborted_plays_df.shape[0]
    
    print(f"aborted_plays_df.shape: {aborted_plays_df.shape}")

    # filtering out coin toss
    coin_toss_terms = ['coin toss']
    coin_toss_df = df[(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(coin_toss_terms))))]
    coin_toss_df['play_type'] = "coin_toss"
    df = df[~(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(coin_toss_terms))))]
    total += coin_toss_df.shape[0]
    
    print(f"coin_toss_df.shape: {coin_toss_df.shape}")

    # filtering out penalty plays
    penalty_terms = ['penalty']
    penalty_df = df[(df['Detail'].str.lower().str.contains(r'\b(?:{})\b'.format('|'.join(penalty_terms))))]
    penalty_df['play_type'] = "penalty"
    df = df[~(df['Detail'].str.lower().str.contains(r'\b(?:{})\b'.format('|'.join(penalty_terms))))]
    total += penalty_df.shape[0]

    # filtering out challenged plays
    challenge_terms = ['challenge', 'challenged']
    challenge_df = df[(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(challenge_terms))))]
    challenge_df['play_type'] = "challenged_play"
    df = df[~(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(challenge_terms))))]
    total += challenge_df.shape[0]
    
    df['play_type'] = None

    print(f"challenge_df.shape: {challenge_df.shape}")

    print(f"Total Found Rows: {total}")

    print(f"remaining_df.shape: {df.shape}")

    new_df = pd.concat([
        pass_plays_df, run_plays_df, field_goal_plays_df, 
        punt_plays_df, kickoff_plays_df, timeout_plays_df,
        spiked_plays_df, aborted_plays_df, coin_toss_df,
        penalty_df, challenge_df, df
    ])

    print(new_df)

    # subset = random.sample(df["play_text"].tolist(), min([df.shape[0], 1000]))
    # for i in subset:
    #     print(i)
    #     # pass

def assign_play_type_info():
    df = get_pbp_dataframe()

    print(f"Starting Shape: {df.shape}")

    player_pos_dict = get_player_pos_dict()
    df['play_text'] = df['Detail'].apply(prepocess_text, args=(player_pos_dict,),)

    # create passing plays column
    pass_terms = ['pass', 'sacked']
    df['passing_play'] = np.where(df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(pass_terms))), True, False)

    # create running plays column
    run_terms = ['up', 'middle', 'rush', 'rushes', 'up the middle', 'left guard', 'right guard', 'left tackle', 'right tackle', 'left end', 'right end']
    df['running_play'] = np.where(
        (df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(run_terms)))) & (~df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(pass_terms)))), 
        True, False
    )

    # create field goals column
    field_goal_terms = ['field goal', 'extra point']
    df['field_goal'] = np.where(
        df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(field_goal_terms))), 
        True, False
    )

    # create punts column
    punt_terms = ['punts', 'punt']
    df['punt'] = np.where(
        df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(punt_terms))), 
        True, False
    )

    # create kickoffs column
    kick_off_terms = ['kicks off', 'kicks']
    df['kickoff'] = np.where(
        df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(kick_off_terms))),
        True, False
    )

    # create timeouts column
    timeout_terms = ['timeout']
    df['timeout'] = np.where(
        df['Detail'].str.lower().str.contains(r'\b(?:{})\b'.format('|'.join(timeout_terms))),
        True, False
    )

    # create spiked balls column
    spike_terms = ['spike', 'spiked', 'kneel', 'kneels']
    df['spiked_ball'] = np.where(
        df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(spike_terms))),
        True, False
    )

    # create aborted plays column
    abort_terms = ['abort', 'aborted']
    df['aborted_play'] = np.where(
        df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(abort_terms))),
        True, False
    )

    # create coin toss column
    coin_toss_terms = ['coin toss']
    df['coin_toss'] = np.where(
        df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(coin_toss_terms))),
        True, False
    )

    # create penalty plays column
    penalty_terms = ['penalty']
    df['penalty'] = np.where(
        df['Detail'].str.lower().str.contains(r'\b(?:{})\b'.format('|'.join(penalty_terms))),
        True, False
    )

    # create challenged plays column
    challenge_terms = ['challenge', 'challenged']
    df['challenge'] = np.where(
        df['Detail'].str.contains(r'\b(?:{})\b'.format('|'.join(challenge_terms))),
        True, False
    )

    print(df)
    print(f"Ending Shape: {df.shape}")

    df.to_csv(script_dir+"../data/play_by_play_extended.csv", index=False)


if __name__ == "__main__":
    # get_play_types()
    assign_play_type_info()