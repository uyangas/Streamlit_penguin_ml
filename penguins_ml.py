import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Penguin Classifier: A Machine Learning App")
st.subheader("An app to test the deployment")
st.write("This app uses 6 inputs to predict the species of penguin using \
         a model built on the Palmer Penguins dataset. Use the form below to get started.")

pengiun_file = st.file_uploader('Upload your own penguin data')

if pengiun_file is not None:
    penguin_df = pd.read_csv(pengiun_file)
    penguin_df = penguin_df.dropna()
    output = penguin_df['species']
    features = penguin_df[['island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex']]
    features = pd.get_dummies(features)

    # loading the model
    rf_pickle = open('random_forest_penguin.pickle', 'rb')
    map_pickle = open('output_penguin.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    uniques = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()

    st.write("Using pre-trained model on the submitted data")

else:
    penguin_df = pd.read_csv('penguins.csv')
    penguin_df = penguin_df.dropna()
    output = penguin_df['species']
    features = penguin_df[['island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex']]
    features = pd.get_dummies(features)

    output, uniques = pd.factorize(output)
    x_train, x_test, y_train, y_test = train_test_split(
        features, output, test_size =0.8
    )

    # train and predict the model
    rfc = RandomForestClassifier(random_state=123)
    rfc.fit(x_train.values, y_train)
    y_pred = rfc.predict(x_test.values)
    score = accuracy_score(y_pred, y_test)
    print('Our accuracy score for this model is {}'.format(score))

    # # save the model
    rf_pickle = open('random_forest_penguin.pickle', 'wb')
    pickle.dump(rfc, rf_pickle)
    rf_pickle.close()
    output_pickle = open('output_penguin.pickle', 'wb')
    pickle.dump(uniques, output_pickle)
    output_pickle.close()

    st.write("Trained the model using existing data")


# get data from input/ user
with st.form('user_inputs'):

    island = st.selectbox('Penguin Island', options=['Biscoe','Dream'])
    sex = st.selectbox('Sex', options=['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value = 0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value = 0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value = 0)
    body_mass = st.number_input('Body Mass (g)', min_value = 0)

    st.form_submit_button()

island_biscoe, island_dream, island_torgerson = 0, 0, 0

if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1

sex_female, sex_male = 0, 0

if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1

new_prediction = rfc.predict([[
    bill_length, bill_depth, flipper_length, body_mass, 
    island_biscoe, island_dream, island_torgerson,
    sex_female, sex_male
]])

prediction_species = uniques[new_prediction][0]
st.write(f"We predict your penguin is of the {prediction_species} species")

if st.button('Generate Feature Importance'):

    fig, ax = plt.subplots()
    ax = sns.barplot(x=rfc.feature_importances_, y=features.columns)
    plt.title('Which features are the most important fro species prediction?')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    fig.savefig('feature_importance.png')

    st.write("We used a machine learning (Random Forest) model to predict the species, \
            the features used in this prediction are ranked by relative importance below.")
    st.image('feature_importance.png')
else:
    pass

if st.button('Generate Histogram by Species'):

    fig, ax = plt.subplots()
    ax = sns.displot(x=penguin_df['bill_length_mm'], 
                      hue=penguin_df['species'])
    
    plt.title('Bill Length by Species')
    plt.axvline(bill_length)
    st.pyplot(ax)

    fig, ax = plt.subplots()
    ax = sns.displot(x=penguin_df['bill_depth_mm'], 
                      hue=penguin_df['species'])
    
    plt.title('Bill Depth by Species')
    plt.axvline(bill_depth)
    st.pyplot(ax)

    fig, ax = plt.subplots()
    ax = sns.displot(x=penguin_df['flipper_length_mm'], 
                      hue=penguin_df['species'])
    
    plt.title('Flipper Length by Species')
    plt.axvline(flipper_length)
    st.pyplot(ax)

else:
    pass