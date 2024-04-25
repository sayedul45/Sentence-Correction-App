import streamlit as st
from gramformer import Gramformer
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Initialize Gramformer model
    set_seed(1212)
    gf = Gramformer(models=1, use_gpu=False)  # Adjust parameters as needed

    # Streamlit UI
    st.title(" Sentence Correction App")

    # Input box for user to enter a sentence
    st.markdown(
        """
        <style>
        .stTextInput>div>div>div>textarea {
            background-color: '#FFF8F8';
            color: black;
            width: 100%;
            height: 200px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    user_input = st.text_area("Enter a sentence:", key='input_area')

    # Button to trigger correction
    if st.button("Correct Sentence"):
        # Perform sentence correction
        corrected_sentences = gf.correct(user_input, max_candidates=1)

        # Display original and corrected sentences
        if corrected_sentences:
            st.write("**Original Sentence:**", user_input)
            corrected_sentences = list(corrected_sentences)
            st.write("**Corrected Sentence:**", corrected_sentences[0])
        else:
            st.write("No correction suggestions found.")

if __name__ == "__main__":
    main()
