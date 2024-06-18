import pickle
import pandas as pd


# load kmeans model
kmeans = pickle.load(open("pickles/kmeans.pkl", 'rb'))

# Get kmeans label
def kmeans_predict(df):
    feature_list = ['Sex_en', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'has_cabin']
    return kmeans.predict(df[feature_list])

# preprocess
def preprocess(input_dict):
    
    if input_dict['Cabin']:
        has_cabin = 1
    else:
        has_cabin = 0
    if input_dict['Sex'] == 'male':
        Sex_en = 0
    else:
        Sex_en = 1

    if input_dict['Title'] not in ['Mr', 'Miss', 'Mrs', 'Master']:
        title = "Other"
    else:
        title = input_dict['Title']

    df = pd.DataFrame({
        'Pclass': [input_dict['Pclass']],
        'Age': [input_dict['Age']],
        'SibSp':  [input_dict['SibSp']],
        'Parch':  [input_dict['Parch']],
        'Fare':  [input_dict['Fare']],
        'Embarked':  [input_dict['Embarked']],
        'has_cabin':  [has_cabin],
        'Title':  [title],
        'Sex_en':  [Sex_en],
    })

    df['kmeans_label'] = kmeans_predict(df)
    df['Embarked'] = pd.Categorical(df['Embarked'])
    df['Title'] = pd.Categorical(df['Title'])

    return df




