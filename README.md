# ND-Data-scientist-P2



## Summary & Description
This Project is part of Data Science Nanodegree Program by Udacity. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

* Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
* Build a machine learning pipeline to train the which can classify text message in various categories
* Run a web app which can show model results in real time

## file structure of the project

- disaster pipeline
  
    |-- Home
        |-- Pipelines Preperations Notebooks
             |-- ETL Pipeline Preparation.ipynb
             |-- ML Pipeline Preparation.ipynb
        |-- workspace
             |-- app
                |-- templates
                    |-- go.html
                    |-- master.html
                |-- run.py
            |-- data
                |-- disaster_categories.csv
                |-- disaster_messages.csv
                |-- DisasterResponse.db
                |-- process_data.py
            |-- models
                |-- classifier.pkl
                |-- train_classifier.py
            |-- README.md



## Getting Started

* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly




## Important files in the repository

* app/templates/*: templates/html files for web app

* data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

* models/train_classifier.py: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

* run.py: This file can be used to launch the Flask web app used to classify disaster messages



## Instructions to run the app


Run the following commands in the project's root directory to set up your database and model.

1- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
2- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
3- Run the following command in the app's directory to run your web app. python run.py

Note: "Pipelines Preperations Notebooks" folder is not necessary for this project to run.




