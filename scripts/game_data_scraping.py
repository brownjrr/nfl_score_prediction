import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser


# setting base url
base_website = "https://www.pro-football-reference.com/boxscores/game-scores.htm"

# get response from websit
page = requests.get(base_website)

# define BeautifulSoup object
soup = BeautifulSoup(page.content, "html.parser")

# get div
results = soup.find(id="all_games")

# creating dataframe from div
df = pd.read_html(results.prettify(), extract_links='body')[0]

# cleaning up the data (visual aid)
df = df.applymap(lambda x: x[0] if x[1] is None else x)

# get link for all games of each score
scores = {}

for ind in df.index:
    tup = df["Unnamed: 7"][ind]
    score = df["Score"][ind]

    scores[score] = tup[1]

# print(scores)

base_url = "https://www.pro-football-reference.com"

for score in scores:
    relative_url = scores[score]

    full_url = base_url + relative_url

    print(full_url)

    page = requests.get(full_url)

    # define BeautifulSoup object
    soup = BeautifulSoup(page.content, "html.parser")

    # get div
    results = soup.find(id="all_games")

    # creating dataframe from div
    df = pd.read_html(results.prettify(), extract_links='body')[0]

    # cleaning up the data (visual aid)
    df = df.applymap(lambda x: x[0] if x[1] is None else x)

    # print(df.columns)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Date'] >= "2013-06-01") & (df['Date'] <= "2024-06-01")]

    # print(df)

    break

# print(results.prettify())

# print(results.prettify())

# print(df)