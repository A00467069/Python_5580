import streamlit as st
from utils import coin_list, historical_data, plot_historical_data_comp

def coin_comparison():
    st.title('Image Classifier')
    coin_df = coin_list()
    
    if not coin_df.empty:
        coin_name1 = st.selectbox('Choose the first cryptocurrency', options=coin_df['name'].unique(), key='1')
        coin_name2 = st.selectbox('Choose the second cryptocurrency', options=coin_df['name'].unique(), key='2')
        
        if st.button("Compare"):
            if coin_name1 != coin_name2:
                coin_id1 = coin_df.loc[coin_df['name'] == coin_name1, 'id'].iloc[0]
                coin_id2 = coin_df.loc[coin_df['name'] == coin_name2, 'id'].iloc[0]
                df1 = historical_data(coin_id1)
                df2 = historical_data(coin_id2)
                plot_historical_data_comp(df1, df2, coin_name1, coin_name2)

            else : 
                st.error("Cannot compare same coins")

    else:
        st.error("Failed to load cryptocurrency data.")

if __name__ == "__main__":
    coin_comparison()