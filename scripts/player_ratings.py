import pandas as pd
import glob
import re


aliases = {
    'WR': {'WR', 'R', 'FL', 'SE'},
    'LB': {
        'LB', 'MIKE', 'WLB', 'JACK', 'WILL', 'SAM', 'SLB', 
        'OLB', 'ILB', 'LILB', 'RILB', 'RLB', 'LLB', 'MLB', 'LOLB', 'ROLB',
        'RUSH', 'BLB'
    },
    'DB': {'DB', 'LCB', 'RCB', 'CB', 'FS', 'SS', 'S', 'RS', 'NB'},
    'TE': {'TE'},
    'K': {'K', 'PK'},
    'QB': {'QB'},
    'RB': {'RB', 'HB', 'FB', 'B', 'H', 'F'},
    'OL': {'OL', 'OG', 'T', 'RT', 'ROT', 'LT', 'OT', 'G', 'LG', 'RG', 'C', 'LS'},
    'P': {'P'},
    'DL': {'DL', 'NG', 'DE', 'LDE', 'RDE', 'DT', 'NT', 'LDT', 'RDT'},
    'PR': {'PR'},
    'UNKNOWN': {'D', 'L', 'KB', 'LA', 'BT', 'LD', 'TOG', 'GOC'},
}

def check_all_pos_mapped():
    """
    Run a check to make sure all positions can be mapped to the aliases dictionary
    """
    # inverting aliases dictionary
    inv_alias_dict = dict()

    for i in aliases:
        for j in aliases[i]:
            inv_alias_dict[j] = i

    print(f"inv_alias_dict:\n{inv_alias_dict}")

    players_df = pd.read_csv("../data/players.csv")

    print(f"players_df:\n{players_df}")

    stat_files = glob.glob("../data/PlayerStats/*.csv")

    print(f"stat_files: {stat_files}")

    for stat_file in stat_files[:]:
        print(f"stat_file: {stat_file}")

        stat_df = pd.read_csv(stat_file)
        stat_df = stat_df[(~stat_df['pos'].astype(str).str.contains("Missed season")) & (~stat_df['pos'].astype(str).str.contains("Did not play"))]

        print(stat_df.columns)

        positions = stat_df['pos'].unique()
    
        missing_positions = []
        for i in positions:
            if isinstance(i, str):
                for j in re.split(r'[^a-zA-Z]', i):
                    missing_positions.append("" if j in inv_alias_dict else j)

        # missing_positions = ["" if j in inv_alias_dict else j for j in re.split(r'[^a-zA-Z]', i) for i in positions if isinstance(i, str)]

        # print([(i, type(i)) for i in positions])
        # print(positions)
        # print(missing_positions)

        for i in missing_positions:
            assert i == ''

        # final_list = []
        # for i in positions:

        print()

# def check_all_pos_mapped():
#     """
#     Run a check to make sure players with more than one position labeled
#     """

if __name__ == '__main__':
    check_all_pos_mapped()