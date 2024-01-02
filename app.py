import streamlit as st
import pickle

st.title('NEWS Classifier: Automated News Classification System')

data=st.text_input("Enter The Content")

predict=st.button("Predict")

with open('vector.pkl','rb') as file:
    loaded_vector=pickle.load(file)

with open('model.pkl','rb') as file:
    loaded_model= pickle.load(file)

if predict:
    vector_data=loaded_vector.transform([data])

    preds=loaded_model.predict(vector_data)

    if preds[0]==0:
        result='Business'
    elif preds[0]==1:
        result='Politics'
    elif preds[0]==2:
        result='Sports'
    elif preds[0]==3:
        result='Weather'
    else:
        result='Other News'

    st.text(result)