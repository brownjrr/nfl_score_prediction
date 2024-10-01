import pandas as pd


def create_position_df():
    # getting adjusted position df
    position_dict = {
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

    inv_alias_dict = dict()

    for i in position_dict:
        for j in position_dict[i]:
            inv_alias_dict[j] = i

    adj_pos_df = pd.Series(inv_alias_dict).to_frame(name='adj_pos')
    adj_pos_df.index.name = "position"
    adj_pos_df = adj_pos_df.reset_index()

    print(f"adj_pos_df:\n{adj_pos_df}")

    # getting position type df
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

    pos_type_df = pd.Series(inv_position_type_dict).to_frame(name='position_type')
    pos_type_df.index.name = "adj_pos"
    pos_type_df = pos_type_df.reset_index()

    print(f"pos_type_df:\n{pos_type_df}")

    pos_df = adj_pos_df.merge(pos_type_df, on=['adj_pos'], how='left')

    print(f"pos_df:\n{pos_df.to_string()}")

    pos_df.to_csv("../data/positions.csv", index=False)




if __name__ == "__main__":
    create_position_df()