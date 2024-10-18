import os
import time
import random
import requests
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
import json


def process_nfl_team_data(team_dict, team_colors, logo_dir, cleaned_logo_dir):
    """Download team logos, clean background, and save paths with colors."""
    # Ensure directories exist
    os.makedirs(logo_dir, exist_ok=True)
    os.makedirs(cleaned_logo_dir, exist_ok=True)

    # List to store the results
    data = []

    # Iterate over each unique team identifier
    for team in set(team_dict.values()):
        url = f"https://www.pro-football-reference.com/teams/{team}/"
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            img_tag = soup.find('img', class_='teamlogo')

            # Extract the URL if the img tag is found
            img_url = img_tag['src'] if img_tag else None

            if img_url:
                # Extract image name from the URL
                img_name = os.path.basename(img_url)
                img_path = os.path.join(logo_dir, img_name)

                # Download the image
                img_response = requests.get(img_url)

                # Check if the request was successful and save the image
                if img_response.status_code == 200:
                    with open(img_path, 'wb') as file:
                        file.write(img_response.content)

                    # Append the result as a dictionary
                    data.append({'team_id': team, 'path': f'logos/{img_name}'})
                else:
                    print(f"Failed to download image for {team}. "
                          f"Status code: {img_response.status_code}")
            else:
                print(f"No image found for {team}")

        # Sleep to avoid overwhelming the server with requests
        time.sleep(random.uniform(3.5, 5.5))

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    df['team_color'] = df['team_id'].map(team_colors)

    # Process images to remove white background
    for logo_file in os.listdir(logo_dir):
        img_path = os.path.join(logo_dir, logo_file)
        img = Image.open(img_path).convert("RGBA")

        # Get the image data
        image_data = img.getdata()

        # Define the color to make transparent (white)
        white = (255, 255, 255, 255)

        # Create a new list for modified data
        new_data = [
            (255, 255, 255, 0) if item[:3] == white[:3] else item
            for item in image_data
        ]

        # Update the image data and save the cleaned image
        img.putdata(new_data)
        new_img_path = os.path.join(cleaned_logo_dir, logo_file)
        img.save(new_img_path, "PNG")

    print("White background removed from all logos.")
    return df


# Define team colors
team_colors = {
    'atl': '#A71930', 'buf': '#00338D', 'car': '#0085CA', 'chi': '#C83803',
    'cin': '#FB4F14', 'cle': '#311D00', 'clt': '#002C5F', 'crd': '#97233F',
    'dal': '#003594', 'den': '#FB4F14', 'det': '#0076B6', 'gnb': '#203731',
    'htx': '#03202F', 'jax': '#006778', 'kan': '#E31837', 'mia': '#008E97',
    'min': '#4F2683', 'nor': '#D3BC8D', 'nwe': '#002244', 'nyg': '#0B2265',
    'nyj': '#125740', 'oti': '#4B92DB', 'phi': '#004C54', 'pit': '#FFB612',
    'rai': '#000000', 'ram': '#003594', 'rav': '#241773', 'sdg': '#002A5E',
    'sea': '#002244', 'sfo': '#AA0000', 'tam': '#D50A0A', 'was': '#773141'
}

# Set directory paths
logo_dir = '../data/logos'
new_logo_dir = '../data/cleaned_logos'
filepath = "../data"

# Load team dictionary from JSON file
with open(f"{filepath}/team_dict.json", 'r') as file:
    team_dict = json.load(file)

# Run the function
df = process_nfl_team_data(team_dict, team_colors, logo_dir, new_logo_dir)

# Save the DataFrame to a new CSV file
df.to_csv(f'{filepath}/logos_path.csv', index=False)
