import os
import glob
import requests
import pandas as pd
from bs4 import BeautifulSoup


def create_player_positions_df(dir_name):
    subfolders= [f.path for f in os.scandir(dir_name) if f.is_dir()]

    print(f"subfolders: {len(subfolders)}")

    position_set = set()
    data = []

    for idx, i in enumerate(subfolders):
        if idx % 100 == 0:
            print(f"{idx}/{len(subfolders)} Folders Seen")

        file = glob.glob(i+"/*.txt")[0]

        pid = i.split("/")[-1]

        # print(pid)

        with open(file, "r", encoding="utf8") as f:
            soup = BeautifulSoup(f.read())
            position_found = False

            try:
                info_div = soup.find(id="info")
                p_tags = info_div.find_all("p")
                
                # checking info div for position
                for i in p_tags:
                    text = i.text.replace("\n", "").replace(" ", "").replace("\t", "").split("Throws")[0]
                    if "Position" in text:
                        position_found = True
                        position = text.lower().replace("position:", "").upper()
                        position_set.add(position)
                        data.append((pid, position))
            except Exception as e:
                print(f"Error: {e} | {file}")
            
            # if position not in info div, look for it in snap counts table
            if not position_found:
                tables = soup.find_all("table")
                table_types = [i.get("id") for i in tables]

                # print(table_types)

                # table was not found. Attempting to fetch page source from link
                if len(table_types) == 0:
                    link = f"https://www.pro-football-reference.com/players/{pid[0].upper()}/{pid}.htm"

                    page = requests.get(link)

                    # define BeautifulSoup object
                    soup = BeautifulSoup(page.content, "html.parser")

                    # get html as str
                    html = str(soup.prettify())

                    tables = soup.find_all("table")
                    table_types = [i.get("id") for i in tables]

                    print(f"redone table_types: {table_types}")                

                for table in tables:
                    temp_df = pd.read_html(table.prettify())[0]

                    if temp_df.columns.nlevels > 1:
                        temp_df.columns = temp_df.columns.droplevel()
                    temp_df.columns = [i.lower() for i in temp_df.columns]

                    if "pos" in temp_df.columns:
                        position_found = True
                        position = "-".join(temp_df['pos'].dropna().unique())
                        position_set.add(position)
                        data.append((pid, position))

            # if not position_found:
            #     print(f"COULD NOT FIND POSITION INFO FOR [{pid}]")

            assert position_found, f"COULD NOT FIND POSITION INFO FOR [{pid}]"

    print(position_set)

    player_position_df = pd.DataFrame(data, columns=['player_id', 'position'])
    null_player_position_df = player_position_df[player_position_df['position'].isna()]

    print(null_player_position_df)
    print(player_position_df)

    player_position_df.to_csv("./data/player_positions.csv", index=False)


if __name__ == "__main__":
    directory = "C:/Users/Robert Brown/OneDrive/player_data_files/"
    create_player_positions_df(directory)