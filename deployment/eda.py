import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import wordcloud
from nltk.corpus import stopwords

def run():
    # create title
    st.title('News Category Dataset Exploratory Analysis')

    # add image
    st.image('https://st2.depositphotos.com/3223379/5688/i/450/depositphotos_56880225-Words-News.jpg')

    # add description
    st.write('This page was made to visualize my exploaration on the news categorization Dataset')

    # create markdown line
    st.markdown('---')

    # create dataframe
    df = pd.read_csv('combined_news_data.csv')

    st.header('News Category Distribution')
    counts = df['Category'].value_counts()

    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    ax.set_title('Category Counts')
    ax.set_xlabel('Category')
    ax.set_ylabel('Counts')

    st.pyplot(fig)

    st.write('Upon analyzing the distribution across each category, I concluded that the data is relatively balanced. `Sport` has the highest number of articles, while `Tech` has the fewest, with a difference of at most 100 articles.')

    st.header('Wordcloud')

    # Group the data by the category column
    groups = df.groupby('Category')

    # List of colormaps
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    color = 0

    # set the English stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(['also', 'said', 'would', 'could', 'new', 'one', 'u']) # add some frequent words that won't help the model

    # Create a figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through each category and create a word cloud
    for i, (name, group) in enumerate(groups):
        # Get the text data
        text = ' '.join(group['Text'])
        
        # Create a word cloud object
        wc = wordcloud.WordCloud(width=1200, height=600, background_color='Black', stopwords=stop_words, min_font_size=18, colormap=colormaps[color])
        color += 1
        
        # Generate the word cloud from the text data
        wc.generate(text)
        
        # Plot the word cloud on the subplot
        axes[i].imshow(wc)
        axes[i].set_title(name)
        axes[i].axis('off')

    # Remove the unused subplot (if any)
    if len(groups) < len(axes):
        fig.delaxes(axes[-1])

    st.pyplot(fig)

    st.write('''
            Here are some frequent words in each categories:
            1. `bussiness` : Company, Firm, Market
            2. `entertainment` : Film, Show, New
            3. `politics` : Government, People, Party
            4. `sport` : Game, Win, Player
            5. `tech` : People, Technology, Mobile

            The models might notices these words as a certain pattern. But, at the end we don't really know how the model works.
            ''')

if __name__ == '__main__':
    run()