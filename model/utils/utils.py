import pickle
import pandas as pd


# load kmeans model
kmeans = pickle.load(open("pickles/kmeans.pkl", 'rb'))

# Get kmeans label
def kmeans_predict(df):
    feature_list = ['Sex_en', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'has_cabin']
    return kmeans.predict(df[feature_list].values)


def preprocess(input_dict):
    
    # Processing Features
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
        'Cabin': [input_dict['Cabin']],
        'Ticket':[input_dict['Ticket']]
        
    })

    # Adding additional features
    df['family_number'] = df.eval("SibSp + Parch")
    df['is_alone'] = df.eval("family_number < 1")
    df['cabin_type'] = pd.Categorical(df.Cabin.fillna("-").apply(lambda x: x[0]))
    df['ticket_type'] = pd.Categorical(df.Ticket.apply(lambda x: x[:3]))
    df['kmeans_label'] = kmeans_predict(df)
    df['Embarked'] = pd.Categorical(df['Embarked'])
    df['Title'] = pd.Categorical(df['Title'])

    df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

    return df




