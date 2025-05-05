# auth.py
import streamlit as st
import sqlite3
import os

DB_PATH = "db/users.db"

def init_db():
    conn = sqlite3.connect("db/users.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    ''')
    conn.commit()
    conn.close()


def login_user():
    init_db()
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
        result = c.fetchone()
        conn.close()
        if result:
            st.success(f"Logged in as {username}")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = result[0]
            st.rerun()
        else:
            st.error("Invalid credentials")

def signup_user():
    init_db()
    st.subheader("Signup")
    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")
    role = st.selectbox("Choose Role", ["CEO", "CTO", "CMO", "Product", "Growth", "CustomerSupport","Marketing","Engineering","Sales"])
    if st.button("Signup"):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
            conn.commit()
            conn.close()
            st.success("Account created! Now log in.")
        except sqlite3.IntegrityError:
            st.error("Username already exists.")

def get_current_user():
    return st.session_state.get("username", None), st.session_state.get("role", "Viewer")
