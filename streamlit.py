import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title = "Netflix TV Show", layout = "wide")
st.title("Netflix TV Show")

def load_data():
    """
    load_data : load pickle data into system

    return :
        df : pickle dataframe
        cosine_sim : pickle cosine similarity
    """

    with open("pickle/df.pkl", "rb") as f:
        df = joblib.load(f)
    with open("pickle/tfidf_matrix.pkl", "rb") as f:
        # Jadi karena cosine_sim terlalu berat ketika dump ke pickle jadinya matrixnya yang di dump
        # dan cosine_sim the initialize ketika berada di streamlit
        tfidf_matrix = joblib.load(f)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

df, cosine_sim = load_data() # load data
titles = df['title'].unique().tolist() # gather title
indices = pd.Series(df.index, index=df['title']).drop_duplicates() # title indexing

def recommend(title, top_n=5):
    """
    recommend : give recommendation based on title that user search "Content Based Filtering"

    args : 
        title (str) : the title user want to find.
        top_n (int) : how much recommendation user want (dari soal diminta default 5). 
    
    return :
        list[str] or str if error occur.
    """

    try:
        idx = indices[title]
    except KeyError: # Title Not Found
        return "Title not found in dataset."

    if idx >= cosine_sim.shape[0]: # Idx trying to access bigger than Cosine
        return "No recommendation available for this TV Show."
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = [score for score in sim_scores if score[0] != idx]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    rec_indices = [i[0] for i in sim_scores[:top_n]]

    return df.iloc[rec_indices]['title'].tolist()

# Scene Init
if "selected_title" not in st.session_state:
    st.session_state.selected_title = None

if "view" not in st.session_state:
    st.session_state.view = "search"

# Return Button ( Mengembalikan View pada State Awal )
if st.session_state.view != "search":
    if st.button("Return"):
        st.session_state.view = "search"
        st.session_state.selected_title = None
        st.rerun()

# Search view ( State Awal )
if st.session_state.view == "search":
    user_input = st.selectbox(
        "Enter a TV Show Title",
        options=titles,
        index=None,
        placeholder="Type in Title to Start Searching"
    )

    if st.button("Search Title") and user_input:
        st.session_state.view = "details"
        st.session_state.selected_title = user_input
        st.rerun()
    
    st.subheader("Available TV Shows:")

    # 5 static title | Agar title yang static tidak terganti tiap run 
    start_pos = df.index.get_loc(indices['Signal']) # akan mencari index dari title yang diinginkan
    samp = df.iloc[start_pos : start_pos + 5, :][['title']].copy() # kemudian di copy pada samp
    samp.reset_index(drop=True, inplace=True) # menambahkan link poster pada samp
    samp['poster'] = [
        "assets/signal.jpeg",
        "assets/slasher.jpeg",
        "assets/the_underclass.jpeg",
        "assets/unnatural_selection.jpeg",
        "assets/my_love_six_stories_of_true_love.jpeg"
    ]
    
    cols = st.columns(len(samp))
    for col, row in zip(cols, samp.itertuples()):
        with col:
            st.image(row.poster, use_container_width=True)
            if st.button(row.title, use_container_width=True):
                st.session_state.view = "details"
                st.session_state.selected_title = row.title
                st.rerun()

# State Halaman Detail ( Detail TV Show yang di cari user )
if st.session_state.view == "details" and st.session_state.selected_title:
    title = st.session_state.selected_title
    show_info = df[df['title'] == title].iloc[0]

    st.subheader(f"{title} | {show_info['rating']} | {show_info['duration']}")
    st.markdown(f"Release Year : {show_info['release_year']}")
    st.markdown(f"Genre : {show_info['listed_in']}")
    st.markdown(f"Description : \n {show_info['description']}")
    st.markdown(f"Directors : \n {show_info['director']}")
    st.markdown(f"Cast : \n {show_info['cast']}")

    # Find Recommendation
    st.markdown("### Similar TV Shows : ")
    recs = recommend(title)
    if isinstance(recs, str):
        st.info("There is no similar TV show.")
    else:
        cols = st.columns(len(recs))
        for col, rec_title in zip(cols, recs):
            with col:
                if st.button(rec_title, use_container_width=True):
                    st.session_state.selected_title = rec_title
                    st.rerun()
