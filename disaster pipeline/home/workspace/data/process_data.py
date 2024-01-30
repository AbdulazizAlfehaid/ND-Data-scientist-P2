# import necessary libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# load data 
def load_data(messages_filepath, categories_filepath):
    
    
    
    """
    Load Messages Data with Categories Function
    
    Parameters:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Returns:
        df -> Combined data containing messages and categories
    """
  
    # load two csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id', how='inner')


# data pre-processing
def clean_data(df):
    """
    Clean Categories Data Function
    
    Parameters:
        df -> Combined data containing messages and categories
    Returns:
        df -> Combined data containing messages and categories with categories cleaned up
    """

    categories = df["categories"].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories` dataframe
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # change column data type from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join='inner', axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    # drop values equal to 2 in "related" column since all values should be binary
    df = df[df['related'] != 2]
    return df


# save cleaned data  
def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Parameters:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
  
    # save a dataframe to sqlite database
    print('Save {} to {} database: '.format(df, database_filename))
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('ETL_prep_table', engine,if_exists = 'replace', index=False)  


def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """
  
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()