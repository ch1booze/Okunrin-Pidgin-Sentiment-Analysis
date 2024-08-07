from collections import Counter

import duckdb
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import streamlit as st
from streamlit_option_menu import option_menu

connection = duckdb.connect(database='datasets/pidgin_sentiment/test.duckdb', read_only=True)
connection.execute('SELECT tweet, label FROM data')
dataset = connection.fetchall()
df = pd.DataFrame(dataset, columns=['pdg', 'label'])

def sentiment_analysis_page():
    st.title("Nigerian Pidgin Sentiment Analysis")

    selected_page = option_menu(
        menu_title=None,
        options=["English", "Pidgin", "Dataset", "Models"],
        icons=["chat-square-text", "translate", "database", "diagram-3"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected_page == "English":
        display_english_content()
    elif selected_page == "Pidgin":
        display_pidgin_content()
    elif selected_page == "Dataset":
        display_data_content()
    else:
        display_model_content()

def display_english_content():
    st.markdown("""
    ## What is Sentiment Analysis?

    **Sentiment analysis** is the process of analyzing large volumes of text to determine whether it expresses a:

    - Positive sentiment 😊
    - Negative sentiment 😔
    - Neutral sentiment 😐

    This technique allows us to *automatically understand the emotional tone in text*,
    """)

    st.markdown("---")

    st.header("Why Pidgin Sentiment Analysis is Important 🚀")
    st.markdown("""
    Pidgin is a crucial mode of communication in many parts of West Africa and beyond. Sentiment analysis for Pidgin has numerous applications:

    1. **Social media monitoring**: Understand public opinion expressed in Pidgin online
    2. **Customer feedback analysis**: Gauge customer satisfaction in Pidgin-speaking markets
    3. **Political opinion tracking**: Monitor political sentiments in Pidgin-speaking regions
    4. **Cultural insights**: Gain deeper understanding of Pidgin-speaking communities
    5. **Improve language technology**: Contribute to the development of NLP tools for Pidgin
    """)

    st.markdown("---")

    st.header("How It Works 🧠")
    st.markdown("""
    The process of Pidgin sentiment analysis typically involves these steps:

    1. **Text Input**: The system receives Pidgin text.
    2. **Processing**: The text is analyzed using techniques tailored for Pidgin.
    3. **Classification**: The sentiment is classified as positive, negative, or neutral.
    4. **Output**: The result is presented, often with a confidence score.
    """)

    st.markdown("---")

    st.header("Try It Yourself! 🚀")
    method = st.selectbox(
                "Select method:",
                ["Rule-based 📏", "Pretrained 🤖", "Finetuned 🎯"],
                index=None,
            )
    user_text = st.text_area("", placeholder="Enter some pidgin text...")
    if st.button("Analyze Sentiment"):
        st.info("Analysis result would appear here. Implementation pending for accurate Pidgin text analysis.")

    st.markdown("---")

    st.subheader("Challenges of Pidgin Sentiment Analysis")
    st.markdown("""
    Analyzing sentiment in Pidgin text presents unique challenges:

    - **Language Variety**: Pidgin varies across different regions.
    - **Limited Resources**: There's a scarcity of labeled data and pre-trained models for Pidgin.
    - **Code-Mixing**: Pidgin is often mixed with other languages, complicating analysis.
    - **Evolving Language**: Pidgin evolves rapidly, requiring frequent system updates.

    Despite these challenges, Pidgin sentiment analysis is crucial for understanding the thoughts and feelings of Pidgin speakers.
    """)

def display_pidgin_content():
    st.markdown("""
    ## Wetin be Sentiment Analysis?

    **Sentiment analysis** na di process wey dey analyze plenty text to know if e dey express:

    - Positive sentiment 😊 (Good vibes)
    - Negative sentiment 😔 (Bad vibes)
    - Neutral sentiment 😐 (Normal vibes)

    Dis powerful technique dey allow us to *automatically understand di emotional tone for text*.
    """)

    st.markdown("---")

    st.header("Why Pidgin Sentiment Analysis Dey Important 🚀")
    st.markdown("""
    Pidgin language na important way wey many people for West Africa and oda parts of di world dey communicate. Sentiment analysis for Pidgin get plenty use cases:

    1. **Social media monitoring**: Understand wetin people dey talk for Pidgin online
    2. **Customer feedback analysis**: Know how customers dey feel about your product or service
    3. **Political opinion tracking**: See wetin people dey think about political matters
    4. **Cultural insights**: Get beta understanding of Pidgin-speaking communities
    5. **Improve language technology**: Help make beta tools for Pidgin language processing
    """)

    st.markdown("---")

    st.header("How E Dey Work 🧠")
    st.markdown("""
    Di process of Pidgin sentiment analysis typically involve these steps:

    1. **Text Input**: Di system go collect Pidgin text.
    2. **Processing**: E go break down di text and analyze am using special techniques wey work for Pidgin.
    3. **Classification**: E go decide if di sentiment na positive, negative, or neutral.
    4. **Output**: E go show di result, sometimes with how sure e be about di result.
    """)

    st.markdown("---")

    st.header("Make You Try Am! 🚀")
    method = st.selectbox(
                "Select method:",
                ["Rule-based 📏", "Pretrained 🤖", "Finetuned 🎯"],
                index=None,
            )
    user_text = st.text_area("", placeholder="Put your Pidgin tok for here...")
    if st.button("Analyze Di Sentiment"):
        st.info("Di analysis result go show here. We still dey work to make am function well for true Pidgin text!")

    st.markdown("---")

    st.subheader("Challenges for Pidgin Sentiment Analysis")
    st.markdown("""
    To analyze sentiment for Pidgin text get some special wahala:

    - **Different Pidgin Styles**: Pidgin dey different for different places.
    - **No Plenty Data**: E no get plenty labeled data or pre-trained models like oda languages.
    - **Language Mixing**: People fit mix Pidgin with oda languages, which fit confuse di analysis.
    - **Language Dey Change**: Pidgin dey change quick-quick, so di system need to dey update.

    Even with all these challenges, Pidgin sentiment analysis dey very important to understand wetin Pidgin speakers dey think and feel.
    """)

def display_data_content():
    st.header("📚 Dataset Overview")
    st.markdown("""
    Our sentiment analysis models are trained on a diverse dataset of tweets in Nigerian Pidgin. 
    Here are some key characteristics:
    
    - 🐦 **Source**: Tweets collected from Twitter
    - 🗣️ **Language**: Nigerian Pidgin
    - 📏 **Size**: 10,600 labeled tweets
    - 🏷️ **Labels**: Each tweet is categorized as:
        - 😊 Positive (0)
        - 😐 Neutral (1)
        - 😞 Negative (2)
    - 📌 **Topics**: Wide range including politics, entertainment, sports, and daily life.
    """)
    
    st.markdown("---")
    
    # Exploratory Data Analysis
    st.header("🔍 Exploratory Data Analysis")
    
    # Sample Data
    st.subheader("📝 Sample Tweets")
    sample_size = 5
    sample_df = df.sample(n=sample_size)
    pidgin_texts = sample_df["pdg"].tolist()
    sentiments = ["Positive" if s == 0 else "Neutral" if s == 1 else "Negative" for s in sample_df["label"]]
    
    sample_table = pd.DataFrame({
        'Pidgin Text': pidgin_texts,
        'Sentiment': sentiments,
    })
    
    # Custom CSS for better table styling
    custom_css = """
    <style>
        .dataframe {
            font-size: 12px;
            font-family: Arial, sans-serif;
        }
        .dataframe th {
            background-color: #f0f2f6;
            color: #1e1e1e;
            font-weight: bold;
            padding: 10px;
        }
        .dataframe td {
            padding: 8px;
        }
        .dataframe tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.table(sample_table)
    
    # Sentiment Distribution
    st.subheader("😊😐😞 Sentiment Distribution")
    sentiment_counts = df["label"].map({0: "Positive", 1: "Neutral", 2: "Negative"}).value_counts().sort_index()
    st.bar_chart(sentiment_counts)
    
    # Text Length Distribution
    st.subheader("📏 Tweet Length Distribution")
    text_lengths = [len(text.split()) for text in df["pdg"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(text_lengths, bins=20, edgecolor='black')
    ax.set_title("Distribution of Tweet Lengths", fontsize=16)
    ax.set_xlabel("Number of Words", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    # Word Frequency
    st.subheader("🔤 Most Common Words")
    most_common_words = {
        "dey": 4427, "i": 3583, "na": 2655, "you": 2171, "me": 1877,
        "no": 1876, "for": 1825, "the": 1779, "to": 1652, "go": 1572,
        "like": 1503, "this": 1466, "my": 1449, "e": 1385, "and": 1381,
        "don": 1309, "be": 1165, "wey": 1117, "say": 1042, "una": 852
    }
    
    # Create a horizontal bar chart with custom colors
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(most_common_words)))
    bars = ax.barh(list(most_common_words.keys()), list(most_common_words.values()), color=colors)
    ax.set_title("Top 20 Most Frequent Words", fontsize=16)
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Words", fontsize=12)
    ax.invert_yaxis()  # To display the most frequent word at the top
    
    # Add value labels to the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width}', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

def display_model_content():
    st.header("Sentiment Analysis Models")

    st.subheader("1. Rule-based Method 📏")
    st.markdown("""
    The rule-based method uses a predefined set of rules and a lexicon (dictionary) of words associated with different sentiments.

    - **How it works**: It searches for specific words or phrases in the text and assigns sentiments based on these predefined rules.
    - **Advantages**: Simple to implement and understand; works well for straightforward expressions.
    - **Limitations**: May struggle with context, sarcasm, or complex sentences; requires manual creation and updating of rules.
    """)

    st.subheader("2. Pretrained Method 🤖")
    st.markdown("""
    The pretrained method uses a model that has been trained on a large corpus of text data, which is then applied to our Pidgin sentiment analysis task.

    - **How it works**: It uses transfer learning, applying knowledge from a model trained on a large dataset (often in English) to our specific Pidgin task.
    - **Advantages**: Can capture complex language patterns; doesn't require a large Pidgin-specific dataset.
    - **Limitations**: May not fully capture Pidgin-specific nuances; performance depends on the similarity between the pretraining data and Pidgin.
    """)

    st.subheader("3. Finetuned Method 🎯")
    st.markdown("""
    The finetuned method starts with a pretrained model and further trains it on our Pidgin-specific dataset.

    - **How it works**: It takes a pretrained model and adjusts its parameters using our labeled Pidgin tweets.
    - **Advantages**: Combines the benefits of pretraining with Pidgin-specific learning; often achieves the best performance.
    - **Limitations**: Requires a good amount of labeled Pidgin data; can be computationally intensive.
    """)

def main():
    sentiment_analysis_page()

if __name__ == "__main__":
    main()
