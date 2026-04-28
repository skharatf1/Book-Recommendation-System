import pickle
import streamlit as st
import numpy as np

st.header("Books Recommeder System using Machine Learning")

model = pickle.load(open('artifacts/model.pkl', 'rb'))
books_name = pickle.load(open('artifacts/books_name.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

def fecth_poster(suggestion):
    poster_url = []

    for book_id in suggestion:
        book_name = book_pivot.index[book_id]

        idx = np.where(final_rating['title'] == book_name)[0][0]
        url = final_rating.iloc[idx]['image-url'] 

        poster_url.append(url)

    return poster_url

def recommend_books(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]

    distance, suggestion = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1),
        n_neighbors=6
    )

    book_list = []
    
    for i in suggestion[0]:
        book_list.append(book_pivot.index[i])

    poster_url = fecth_poster(suggestion[0])

    return book_list, poster_url

selected_books = st.selectbox(
    "Type or Select a Book",
    books_name
)

if st.button('Show Recommendation'):
    recommendation_books, poster_url = recommend_books(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommendation_books[1])
        st.image(poster_url[1])
    with col2:
        st.text(recommendation_books[2])
        st.image(poster_url[2])
    with col3:
        st.text(recommendation_books[3])
        st.image(poster_url[3])
    with col4:
        st.text(recommendation_books[4])
        st.image(poster_url[4])
    with col5:
        st.text(recommendation_books[5])
        st.image(poster_url[5])