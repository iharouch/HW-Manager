import streamlit as st

st.set_page_config(
    page_title="IST 488 HW Manager",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=":material/edit:"
)

HW1 = st.Page("HW/HW1.py", title="HW 1", icon="📝")
HW2 = st.Page("HW/HW2.py", title="HW 2", icon="📝", default=True)

pg = st.navigation([HW1, HW2])
pg.run()
