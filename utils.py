import streamlit as st

def Title():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "WeLift";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )