import streamlit as st

HW1 = st.Page("HW/HW1.py", title = "HW 1", icon="📝")
HW2 = st.Page("HW/HW2.py", title = "HW 2", icon="📝")
HW3 = st.Page("HW/HW3.py", title = "HW 3", icon="📝")
HW4 = st.Page("HW/HW4.py", title = "HW 4", icon="📝")
HW5 = st.Page("HW/HW5.py", title = "HW 5", icon="📝")
HW7 = st.Page("HW/HW7.py", title = "HW 7", icon="📝", default=True)

pg = st.navigation([HW1, HW2, HW3, HW4, HW5, HW7])
st.set_page_config( 
    #Set page title 
    page_title = "IST 488 HW Manager", 
    layout = "wide", 
    initial_sidebar_state = "expanded", 
    page_icon=":material/edit:" 
) 
pg.run()
