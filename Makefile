.PHONY: install test train run

install:
	pip install -r requirments.txt

test:
	PYTHONPATH=. pytest -q

train:
	python src/train.py

run:
	streamlit run app/app.py
