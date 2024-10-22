run_model: prediction_pipeline/score_prediction.py data/intermediate/games_df.csv data/intermediate/game_rank_matchup.csv
	@echo "Generating model"
	@python prediction_pipeline/score_prediction.py \
	@echo "...model trained"

game_rank_matchup: ./prediction_pipeline/player_ranking_features.py ./data/intermediate/games_df.csv ./data/intermediate/roster_df.csv
	python ./prediction_pipeline/player_ranking_features.py

games_dataframe: ./prediction_pipeline/team_features.py ./data/coach_ratings.csv
	python ./prediction_pipeline/team_features.py

./data/intermediate/roster_df.csv: ./prediction_pipeline/roster_features.py ./data/player_ratings.csv
	python ./prediction_pipeline/roster_features.py

./data/player_ratings.csv: ./scripts/players_rating.py ./data/stat_weights_by_position.csv ./data/stat_weights_overall.csv
	python ./scripts/players_rating.py
	
./data/coach_ratings.csv: ./scripts/coaches_rating.py ./data/game_level_coach_data_extended.csv
	python ./scripts/coaches_rating.py

./data/team_rating.csv: ./scripts/team_rating.py ./data/game_stats_all.csv
	python ./scripts/team_rating.py

./data/game_level_coach_data_extended.csv: ./scripts/coach_stats_augment.py ./data/play_by_play_extended_v2.csv ./data/game_level_coach_data.csv
	python ./scripts/coach_stats_augment.py

./data/play_by_play_extended_v2.csv: ./scripts/play_by_play_augment.py ./data/game_level_coach_data.csv ./data/play_by_play_extended.csv
	python ./scripts/play_by_play_augment.py

./data/play_by_play_extended.csv: ./scripts/extract_features_play_by_play.py
	python ./scripts/extract_features_play_by_play.py

./data/game_level_coach_data.csv: ./scripts/boxscores.py
	python ./scripts/boxscores.py

./data/stat_weights_by_position.csv ./data/stat_weights_overall.csv: ./scripts/statistic_weights.py
	python ./scripts/statistic_weights.py

./data/interaction_prob.csv: ./scripts/play_by_play.py ./data/play_by_play.csv ./data/players.csv
	python ./scripts/play_by_play.py
