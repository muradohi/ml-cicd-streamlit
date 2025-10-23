.PHONY: install test train run

install:
	pip install -r requirements.txt

test:
	pytest -q

train:
	python src/train.py

run:
	streamlit run app/app.py
