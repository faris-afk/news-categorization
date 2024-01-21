import pandas as pd
import streamlit as st
import eda
import prediction

# Set the page title and favicon
st.set_page_config(page_title="News Category Classification", 
                page_icon="📰",
                layout='wide',
                initial_sidebar_state='expanded'
)

# Create a sidebar with a title and a selection box
st.sidebar.title("Choose a page:")
page = st.sidebar.selectbox("", ('Landing Page', 'Data Exploration', 'Data Prediction'))

# Display different content depending on the selected page
if page == 'Data Exploration':
    eda.run()
elif page == 'Data Prediction':
    prediction.run()
else:
    # Add a header and a subheader with some text
    st.title("What kind of news is this?")
    st.subheader("Find out the news category with this space that uses deep learning to do predictions.")

    # Add an image about the case
    st.image("https://thumbs.dreamstime.com/b/online-news-article-tablet-screen-electronic-newspaper-magazine-latest-press-media-mockup-digital-portal-151771038.jpg")
    with st.expander("Backgroud dataset"):
        st.caption("""
                The data set used contains news articles and their respective category.
                """)
    with st.expander("Problem statements"):
        st.caption("The goal is to predict which category belongs to. This project was done in hope to help online news company categorize their news articles.")