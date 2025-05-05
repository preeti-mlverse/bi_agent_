# app/settings.py
import streamlit as st
import sqlite3
import os

DB_PATH = "db/users.db"

def init_user_db():
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT,
            role TEXT
        )
    """)

    # Check if 'display_name' column exists, add it if not
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    if "display_name" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN display_name TEXT")

    conn.commit()
    conn.close()

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username, display_name, email, role FROM users")
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_user(username, field, value):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"UPDATE users SET {field}=? WHERE username=?", (value, username))
    conn.commit()
    conn.close()

def invite_user(username, display_name, email, role):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?)", (username, email, display_name, role))
    conn.commit()
    conn.close()

def settings_ui():
    st.title("⚙️ Settings & User Management")

    init_user_db()

    tab1, tab2 = st.tabs(["🧑 My Profile", "👥 Manage Users"])

    with tab1:
        st.subheader("Update Your Profile")
        current_user = st.session_state.get("current_user", "demo_user")

        display_name = st.text_input("Display Name")
        email = st.text_input("Email")
        role = st.selectbox("Role", ["Admin", "Analyst", "Viewer"])

        if st.button("💾 Save My Profile"):
            update_user(current_user, "display_name", display_name)
            update_user(current_user, "email", email)
            update_user(current_user, "role", role)
            st.success("✅ Profile updated.")

    with tab2:
        st.subheader("Manage & Invite Users")

        with st.form("invite_form"):
            new_username = st.text_input("Username")
            new_display = st.text_input("Display Name")
            new_email = st.text_input("Email")
            new_role = st.selectbox("Role", ["Admin", "Analyst", "Viewer"])
            submitted = st.form_submit_button("➕ Invite User")

            if submitted:
                invite_user(new_username, new_display, new_email, new_role)
                st.success(f"✅ User {new_username} invited.")

        st.divider()
        st.subheader("All Registered Users")

        users = get_all_users()
        for u in users:
            st.markdown(f"- **{u[0]}** ({u[2]}) — *{u[3]}*")