"""Start with apps/streamlit_app/run.sh after loading the private server environment."""

import os

import streamlit as st

from agatha.database.assets import PrivateStorage
from agatha.database.explorer import render_app
from agatha.database.repository import Repository

st.set_page_config(page_title="AGATHA · Scenario Explorer", layout="wide")

if st.get_option("server.address") not in ("127.0.0.1", "localhost", "::1"):
    st.error("Start this private research viewer with apps/streamlit_app/run.sh, which binds to localhost.")
    st.stop()
if not os.environ.get("AGATHA_DATABASE_URL"):
    st.title("AGATHA · Scenario Explorer")
    st.info("Supabase is not configured yet. Follow docs/stage3.md, then restart with the private environment loaded.")
    st.stop()

try:
    storage = PrivateStorage.from_env() if os.environ.get("SUPABASE_SECRET_KEY") else None
    render_app(Repository.from_env(), storage)
except Exception:
    # Never expose driver exceptions: they can contain a password, URL or authorization header.
    st.error("Stage 3 could not read or write its private services. Check the environment and migrations, then retry.")
