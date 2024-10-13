./data/stat_weights_by_position.csv ./data/stat_weights_overall.csv: ./scripts/play_by_play.py
	python ./scripts/statistic_weights.py

./data/interaction_prob.csv: ./scripts/play_by_play.py ./data/play_by_play.csv ./data/players.csv
	python ./scripts/play_by_play.py