#import necessary libraries
import sys
import nltk
import re
nltk.download(['punkt', 'wordnet'])
import warnings
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier



def load_data(database_filepath):
    """
    Load Data from Database Function
    
    Parameters:
        database_filepath -> Path to SQLite destination database
    Returns:
        X -> a dataframe containing features
        y -> a dataframe containing labels
        category_names -> List of categories names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("ETL_prep_table", engine)
    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names 


def tokenize(text):
    """
    Tokenize the text function
    
    Parameters:
        text -> Text message which needs to be tokenized
    Returns:
        clean_tokens -> List of tokens extracted from the provided text
    """
    # forming a regular expression that detects URLs 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # using url_regex to find all URLs
    detected_urls = re.findall(url_regex, text)
    
    # replace URL with "urlplaceholder"
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens    


def build_model():
    """
    Create a pipeline for model function
    
    Returns:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [8, 15],
        'clf__estimator__min_samples_split': [2],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate the model performance
    
    This function applies a ML pipeline to a test set and prints out the model performance
    
    Parameters:
        model -> A valid scikit ML Pipeline
        X_test -> Test features
        y_test -> Test labels
        category_names -> label names
    """
    y_pred = model.predict(X_test)
    
    # Flatten y_test and y_pred to 1D arrays
    y_test_flat = y_test.values.flatten()
    y_pred_flat = y_pred.flatten()

    print(classification_report(y_test_flat, y_pred_flat, target_names=category_names))
    
    for i in range(y_test.shape[1]):
        print('%25s accuracy : %.2f' % (category_names[i], accuracy_score(y_test.iloc[:, i], y_pred[:, i])))

def save_model(model, model_filepath):
    """
    Save Pipeline model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Parameters:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()