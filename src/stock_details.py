import streamlit as st
from utils import coin_list, historical_data, plot_historical_data
import matplotlib.pyplot as plt 


def stock_details():
    st.title('Crypto Details')
    coin_df = coin_list()
    timerange = 365

    if not coin_df.empty:
        coin_name = st.selectbox('Choose a cryptocurrency', options=coin_df['name'].unique())
        coin_id = coin_df.loc[coin_df['name'] == coin_name, 'id'].iloc[0]

        df = historical_data(coin_id, timerange)
        plot_historical_data(df)

    else:
        st.error("Failed to load cryptocurrency data.")
        
    

