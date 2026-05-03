install:
	poetry install

data:
	poetry run python src/data/load_data.py
	poetry run python src/data/preprocess.py

train:
	poetry run python scripts/train_model.py

test:
	poetry run pytest tests/