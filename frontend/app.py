import streamlit as st
import requests

# Page config
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Hide default Streamlit UI
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    header {visibility: hidden;}

    div[data-testid="stPopover"] button {
        background: transparent !important;
        border: none !important;
        font-size: 28px !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    div[data-testid="stPopover"] button:hover {
        background: transparent !important;
    }

    div[data-baseweb="popover"] svg {
        display: none !important;
    }

    div[data-baseweb="popover"] {
        min-width: 120px !important;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Theme state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Apply theme
if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }

        textarea {
            background-color: #1e1e1e !important;
            color: white !important;
            border: 1px solid #333 !important;
        }

        textarea::placeholder {
            color: #aaa !important;
        }

        div.stButton > button {
            background-color: #1e1e1e !important;
            color: white !important;
            border: 1px solid #333 !important;
        }
        </style>
    """, unsafe_allow_html=True)

elif st.session_state.theme == "light":
    st.markdown("""
        <style>
        .stApp {
            background-color: white;
            color: black;
        }

        textarea {
            background-color: white !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }

        textarea::placeholder {
            color: gray !important;
        }

        div.stButton > button {
            background-color: white !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Header
col1, col2 = st.columns([8, 1])

with col1:
    st.title("🛡 TruthLens")
    st.caption("AI-powered fake news detection & source verification")

with col2:
    with st.popover("⋮"):
        c1, c2 = st.columns(2)

        with c1:
            if st.button("🌙"):
                st.session_state.theme = "dark"
                st.rerun()

        with c2:
            if st.button("☀️"):
                st.session_state.theme = "light"
                st.rerun()

# spacing
st.markdown("<br><br><br>", unsafe_allow_html=True)

# Input section
article_text = st.text_area(
    "",
    height=80,
    placeholder="Message TruthLens..."
)

# Analyze button
analyze_button = st.button("🔍 Analyze")

if analyze_button:
    if article_text.strip() != "":
        with st.spinner("Analyzing the article..."):

            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json={"text": article_text}
            )

            result = response.json()

            st.subheader("🧠 Analysis Result")
            st.write(f"Prediction: {result['prediction']}")
            st.write(f"Confidence: {result['confidence']}")

            try:
                related_response = requests.post(
                    "http://127.0.0.1:5000/related-news",
                    json={"text": article_text}
                )

                related_data = related_response.json()

                st.subheader("📰 Here are some related news articles you may want to check")

                for article in related_data["articles"]:
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#1e1e1e;
                            padding:15px;
                            border-radius:10px;
                            margin-bottom:10px;
                            border:1px solid #333;">
                            <a href="{article['url']}" target="_blank" 
                            style="text-decoration:none; color:#4da6ff; font-size:22px; font-weight:bold;">
                                {article['title']}
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except:
                st.warning("Unable to fetch related news at the moment.")

