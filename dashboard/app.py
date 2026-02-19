import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed/ranked_exoplanets.csv")

st.title("ExoHabitAI Dashboard")

st.subheader("Top Ranked Exoplanets")
st.dataframe(df.head(20))

st.subheader("Habitability Score Distribution")
fig, ax = plt.subplots()
ax.hist(df["habitability_score"], bins=30)
st.pyplot(fig)

st.subheader("Feature Importance (If available)")
