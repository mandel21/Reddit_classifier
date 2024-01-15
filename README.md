# Text Processing and FastText Model Training

This repository contains Python code for text processing and FastText model training on the Extrovert-Introvert dataset and trying to judgmental or perceiving Reddit users to introvert or extroverts for the Language ad AI course. The code is organized into three main components:

1. **Data Processing (`data_processor.py`):** This module processes the Extrovert-Introvert dataset, including filtering common IDs with another dataset, writing filtered data to CSV files, and splitting the dataset into training and testing sets.

2. **Text Data Processing (`text_data_processor.py`):** This module handles the lemmatization of text data  of the dataset . The data is split into training and testing sets.

3. **Model Training (`model_trainer.py`):** This module uses FastText to train a text classification model. It takes the  training data and evaluates the model on a separate test dataset.

## Usage

You can find the lines of code executing the task on the bottom of each file

install the required libraries by running `pip install fasttext polars spacy imbalanced-learn`.

## Repository Structure

- `data_processor.py`: Handles data processing and filtering common IDs.
- `text_data_processor.py`: (NOT implemented yet)Manages the lemmatization of text data and oversampling.
- `model_trainer.py`: Takes care of training the FastText model and evaluating its performance.

## Dependencies
- FastText
- Polars
- spaCy
- imbalanced-learn
