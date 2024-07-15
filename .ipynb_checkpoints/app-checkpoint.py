# app.py

import pandas as pd
import streamlit as st
from sample import value_calculator, o2d_predictor 

# Load your data
df1 = pd.read_excel('../O2D_delay_data.xlsx', 'logistics')
df2 = pd.read_excel('../O2D_delay_data.xlsx', 'po_level')
df2 = df2[df2['order_status'].str.lower() == 'completed']
df2 = df2[['buyer_name', 'seller_name', 
       'transporter_type', 'order_type','seller_po_number',
       'order_completion_percentage']]

df = df1.merge(df2, left_on='po_number', right_on='seller_po_number', how='left')
df = df.sort_values(by=['po_number', 'vehicle_unloaded_ts'])

# Initialize the value_calculator to get value_dict
calculator = value_calculator(df)
value_dict = calculator.main()

# Streamlit app
st.title('O2D Predictor')

# Sidebar with input fields
category = st.sidebar.selectbox('Category', ['aluminium', 'steel'])
po_qty = st.sidebar.number_input('PO Quantity', value=1)
buyer_name = st.sidebar.selectbox('', value_dict['buyer'].keys())
seller_name = st.sidebar.selectbox('Seller Name', value_dict['seller'].keys())
lane = st.sidebar.selectbox('Lane', value_dict['route'].keys())
# destination = st.sidebar.selectbox('Destination', value_dict['destination'].keys())

# Initialize the predictor class with value_dict
predictor = o2d_predictor(category, po_qty, buyer_name, seller_name, lane, value_dict)

# Button to trigger prediction
if st.sidebar.button('Predict'):
    prediction = predictor.o2d_calculator()
    st.write(f'Predicted O2D Value: {prediction}')

# Optionally, display additional information
st.sidebar.subheader('About')
st.sidebar.text('This app predicts O2D values based on selected inputs.')
