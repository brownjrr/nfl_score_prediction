import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def prepare_vis_data(filepath, folderpath):
    """Prepare home and opponent visualization dataframes with team, model and logos files."""
    # Load data
    games_df = pd.read_csv(f"{filepath}/games_df.csv")
    model_df = pd.read_csv(f"{filepath}/random_forest_model_output.csv")
    logos_path = pd.read_csv(f"{filepath}/logos_path.csv")

    # Load team dictionary
    with open(f"{filepath}/team_dict.json", 'r') as file:
        team_dict = json.load(file)

    # Merge model output with game information
    merged_df = pd.merge(
        model_df,
        games_df[['boxscore_stub', 'home_team_id', 'opp_team_id']],
        left_on='boxscore_stub',
        right_on='boxscore_stub',
        how='left'
    )

    # Calculate mean scores for home and opponent teams
    home_mean_merged_df = (
        merged_df.groupby('home_team_id')[['score_home_test', 'score_home_pred']]
        .mean()
        .reset_index()
        .rename(columns={'home_team_id': 'team_id', 'score_home_test': 'score_test', 'score_home_pred': 'score_pred'})
    )

    opp_mean_merged_df = (
        merged_df.groupby('opp_team_id')[['score_opp_test', 'score_opp_pred']]
        .mean()
        .reset_index()
        .rename(columns={'opp_team_id': 'team_id', 'score_opp_test': 'score_test', 'score_opp_pred': 'score_pred'})
    )

    # Calculate residuals
    home_mean_merged_df['residual'] = home_mean_merged_df['score_test'] - home_mean_merged_df['score_pred']
    opp_mean_merged_df['residual'] = opp_mean_merged_df['score_test'] - opp_mean_merged_df['score_pred']

    # Adjust logo paths for visualization
    logos_path['path'] = logos_path['path'].apply(lambda x: f"{folderpath}{x}")

    # Merge the mean dataframes with logo paths
    home_vis_df = pd.merge(home_mean_merged_df, logos_path, on='team_id', how='left')
    opp_vis_df = pd.merge(opp_mean_merged_df, logos_path, on='team_id', how='left')

    return home_vis_df, opp_vis_df


def getImage(path):
    image = plt.imread(path)
    return OffsetImage(image, zoom=0.15)  # Adjust zoom for appropriate size


def create_model_vis(df, game='Home'):
    """Create a scatter plot visualization for model predictions and residuals."""
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot for team ratings over time with invisible points
    ax.scatter(df['score_pred'], df['residual'], alpha=0)

    # Draw a horizontal line at residual = 0
    ax.axhline(0, color='black', linestyle='-', linewidth=2)

    # Add team logos at respective points and vertical lines from zero
    for _, row in df.iterrows():
        image = getImage(row['path'])
        ab = AnnotationBbox(
            image, (row['score_pred'], row['residual']), frameon=False
        )
        ax.add_artist(ab)

        # Get the team code for the row
        team_code = row['team_id']

        # Draw vertical lines from residual = 0 to the data point with team-specific color
        line_color = row['team_color']
        ax.vlines(
            x=row['score_pred'], ymin=0, ymax=row['residual'],
            color=line_color, linewidth=2
        )

    # Set labels and title
    ax.set_xlabel(f'Predicted {game} Score')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Random Forest Regression 2023 - {game}')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to ensure everything fits well
    plt.tight_layout()

    # Return the plot object
    return plt

# Example usage:
filepath = "../data"
folderpath = "../data/cleaned_" # for new_logos folder or just "../data/" for white background logos
home_vis_df, opp_vis_df = prepare_vis_data(filepath, folderpath)

# For opp teams
create_model_vis(opp_vis_df,game='Opp')
