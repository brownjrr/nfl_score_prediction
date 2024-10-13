import requests
import pandas as pd
from bs4 import BeautifulSoup
import glob
import time
import os
import multiprocessing as mp


def get_boxscores():
    base_dir = "C:/Users/Robert Brown/OneDrive/"
    save_folder = f"{base_dir}boxscore_data_files/"
    files = [i.replace("\\", "/") for i in glob.glob(f"{base_dir}pbp_data_files/*.txt")]
    boxscores = [i.split("/")[-1].replace(".txt", "") for i in files]
    urls = [
        f"https://www.pro-football-reference.com/boxscores/{i}.htm"
        for i in boxscores
    ]

    seen_boxscores = {str(f.path).split('/')[-1] for f in os.scandir(save_folder) if f.is_dir()}

    print(f"seen_boxscores: {len(seen_boxscores)}")
    
    step = 0
    for boxscore, i in list(zip(boxscores, urls)):
        if step % 100 == 0:
           print(f"Step {step} of {len(urls)}")

        step += 1

        if boxscore in seen_boxscores: continue

        # print(f"URL: {i}")

        page = requests.get(i)

        # define BeautifulSoup object
        soup = BeautifulSoup(page.content, "lxml")

        # get html as str
        html = str(soup.prettify())

        # print(html)

        boxscore_folder = f"{save_folder}{boxscore}/"
        os.makedirs(boxscore_folder, exist_ok=True)

        with open(f"{boxscore_folder}page_source.txt", "w+", encoding="utf-8") as f:
            f.write(html)

        time.sleep(2)
    


if __name__ == '__main__':
    get_boxscores()
