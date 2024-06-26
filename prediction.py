import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pickle_in = open('model.pkl', 'rb') 
model = pickle.load(pickle_in)

def preprocess_data(data):
    df = pd.read_csv('data_C.csv')
    X, y = df.drop(columns=['churn']), df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    numeric_col = ['CreditScore', 'Balance', 'EstimatedSalary', 'Age']
    X_train[numeric_col] = scaler.fit(X_train[numeric_col])
    
    data[numeric_col] = scaler.transform(data[numeric_col])
    return data

def main():
    st.title('Predicting Customer Churn')
    
    # Add Input
    gender = st.selectbox(label="What's the customer Gender?" , options=['Male', 'Female'])
    geography = st.selectbox(label="Which country the customer is from?", options=['France', 'Spain', 'Germany'])
    surname_prevalency = st.selectbox(label="How prevalent is the customer surname?", options=['Very Prevalen', 'Prevalent', 'A bit Prevalen', 'Not Prevalen', 'Very Not Prevalen'])
    credit_score = st.number_input(label="What's the customer credit score?", min_value=0, max_value=850)
    age = st.number_input(label="What's the customer age?", min_value=0, max_value=100)
    tenure = st.number_input(label="How long the customer has been with the bank?", min_value=0, max_value=10)
    balance = st.number_input(label="What's the customer balance?", min_value=0, max_value=250000)
    num_of_products = st.slider(label="How many products the customer has?", min_value=1, max_value=4)
    has_credit_card = st.selectbox(label="Does the customer have a credit card?", options=['Yes', 'No'])
    is_active_member = st.selectbox(label="Is the customer an active member?", options=['Yes', 'No'])
    estimated_salary = st.number_input(label="What's the customer estimated salary?", min_value=0, max_value=200000)
    
    # Create a Dictinonary
    data_dict = {
        'CreditScore': 0.0, 'Age': 0.0, 'Tenure': 0, 'Balance': 0.0, 'NumOfProducts': 0, 'HasCrCard': 0, 
        'IsActiveMember': 0, 'EstimatedSalary': 0.0, 'Geography_France': 0.0, 
        'Geography_Germany': 0.0, 'Geography_Spain':0.0, 'Gender_Female': 0.0, 'Gender_Male': 0.0,
        'Surname Prevalency_A bit Prevalen': 0.0, 'Surname Prevalency_Not Prevalen': 0.0,
        'Surname Prevalency_Prevalent': 0.0, 'Surname Prevalency_Very Not Prevalen': 0.0,
        'Surname Prevalency_Very Prevalen': 0.0}
        
    # Set the input values
    data_dict['CreditScore'] = credit_score
    data_dict['Age'] = age
    data_dict['Tenure'] = tenure
    data_dict['Balance'] = balance
    data_dict['NumOfProducts'] = num_of_products
    data_dict['EstimatedSalary'] = estimated_salary
    
    if gender == 'Male':
        data_dict['Gender_Male'] = 1.0
    else:
        data_dict['Gender_Female'] = 1.0
        
    if geography == 'France':
        data_dict['Geography_France'] = 1.0
    elif geography == 'Germany':
        data_dict['Geography_Germany'] = 1.0
    else:
        data_dict['Geography_Spain'] = 1.0
        
    if surname_prevalency == 'Very Prevalen':
        data_dict['Surname Prevalency_Very Prevalen'] = 1.0
    elif surname_prevalency == 'Prevalent':
        data_dict['Surname Prevalency_Prevalent'] = 1.0
    elif surname_prevalency == 'A bit Prevalen':
        data_dict['Surname Prevalency_A bit Prevalen'] = 1.0
    elif surname_prevalency == 'Not Prevalen':
        data_dict['Surname Prevalency_Not Prevalen'] = 1.0
    else:
        data_dict['Surname Prevalency_Very Not Prevalen'] = 1.0
        
    if has_credit_card == 'Yes':
        data_dict['HasCrCard'] = 1.0
    
    if is_active_member == 'Yes':
        data_dict['IsActiveMember'] = 1.0
        
    data = pd.DataFrame(data_dict, index=[0])
        
    data = preprocess_data(data)
    
    st.dataframe(data)
    
    # Make prediction
    if st.button('Predict'):
        pred = model.predict(data)
        if pred == 1:
            st.error('The customer is likely to churn')
        else:
            st.success('The customer is not likely to churn')
            
if __name__ == '__main__':
    main()
    
    
    
    
