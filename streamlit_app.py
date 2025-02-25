import streamlit as st
import test_seq2seq

class app:

    def __init__ (self):
        self.model = test_seq2seq.Seq2SeqModel()

    def run(self):
        st.title("AdrianGPT")
        q = st.text_input("Enter a question")
        if st.button("Generate"):
            res = self.model.generate(q)
            st.write(res)

class main:

    if __name__ == "__main__":
        my_app = app()
        my_app.run()
