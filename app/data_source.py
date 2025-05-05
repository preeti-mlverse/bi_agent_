# data_source.py
import streamlit as st
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import pymongo
import os
import snowflake.connector

os.makedirs("uploads", exist_ok=True)

def upload_data_ui():
    st.title("🔌 Connect Data Sources")
    if 'connections' not in st.session_state:
        st.session_state.connections = {}

    data_sources = st.multiselect("Select Data Sources to Connect", [
        "SQLite", "Postgres", "MySQL", "Snowflake", "MongoDB", "AWS RDS", "GCP SQL", "Kafka", "SQLWorkbench"
    ])

    for source in data_sources:
        with st.expander(f"🔧 Configure {source}"):
            if source == "SQLite":
                uploaded_file = st.file_uploader("Upload a `.db` file", type=["db"], key=f"file_{source}")
                if uploaded_file:
                    default_name = uploaded_file.name.replace(".db", "")
                    custom_name = st.text_input("🔖 Name your Data Source", value=default_name, key=f"name_{source}")

                    if st.button(f"Connect {source}", key=f"btn_{source}"):
                        try:
                            # Save the uploaded file to local /uploads
                            db_path = os.path.join("uploads", uploaded_file.name)
                            with open(db_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            # Connect and fetch tables
                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                            tables = [row[0] for row in cursor.fetchall()]
                            conn.close()

                            # Save to session state
                            st.session_state.connections[custom_name] = {
                                "type": "sqlite",
                                "database": db_path,
                                "tables": tables,
                                "source": source
                            }

                            st.success(f"✅ `{custom_name}` connected with {len(tables)} tables.")
                            st.markdown("### 📋 Tables in this Database:")
                            for t in tables:
                                st.markdown(f"- `{t}`")

                        except Exception as e:
                            st.error(f"❌ Failed to connect: {str(e)}")


            elif source in ["Postgres", "MySQL", "AWS RDS", "GCP SQL", "SQLWorkbench"]:
                host = st.text_input(f"{source} Host", key=f"{source}_host")
                port = st.text_input(f"{source} Port", key=f"{source}_port")
                user = st.text_input(f"{source} Username", key=f"{source}_user")
                pwd = st.text_input(f"{source} Password", key=f"{source}_pwd", type="password")
                db = st.text_input(f"{source} DB Name", key=f"{source}_db")
                custom_name = st.text_input(f"Name this Connection", key=f"name_{source}")

                if st.button(f"Connect {source}", key=f"btn_{source}"):
                    try:
                        dialect = {
                            "Postgres": "postgresql",
                            "MySQL": "mysql+pymysql",
                            "AWS RDS": "mysql+pymysql",
                            "GCP SQL": "postgresql",
                            "SQLWorkbench": "sqlite"
                        }[source]
                        uri = f"{dialect}://{user}:{pwd}@{host}:{port}/{db}"
                        engine = create_engine(uri)
                        engine.connect()
                        st.session_state.connections[custom_name or source] = engine
                        st.success(f"{source} connected!")
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")

            elif source == "Snowflake":
                acc = st.text_input("Account", key="sf_acc")
                user = st.text_input("User", key="sf_user")
                pwd = st.text_input("Password", type="password", key="sf_pwd")
                db = st.text_input("Database", key="sf_db")
                warehouse = st.text_input("Warehouse", key="sf_wh")
                custom_name = st.text_input("Name this Connection", key="name_sf")

                if st.button("Connect Snowflake"):
                    try:
                        ctx = snowflake.connector.connect(
                            user=user, password=pwd, account=acc, database=db, warehouse=warehouse
                        )
                        st.session_state.connections[custom_name or "Snowflake"] = ctx
                        st.success("Snowflake connected!")
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif source == "MongoDB":
                uri = st.text_input("Mongo URI", key="mongo_uri")
                custom_name = st.text_input("Name this Connection", key="name_mongo")

                if st.button("Connect MongoDB"):
                    try:
                        client = pymongo.MongoClient(uri)
                        st.session_state.connections[custom_name or "MongoDB"] = client
                        st.success("MongoDB connected!")
                    except Exception as e:
                        st.error(f"Mongo error: {str(e)}")

            elif source == "Kafka":
                kafka_host = st.text_input("Kafka Bootstrap Server", key="kafka_boot")
                topic = st.text_input("Kafka Topic", key="kafka_topic")
                custom_name = st.text_input("Name this Connection", key="name_kafka")

                if st.button("Register Kafka"):
                    st.session_state.connections[custom_name or "Kafka"] = {"bootstrap": kafka_host, "topic": topic}
                    st.success("Kafka source added!")

    if st.session_state.connections:
        st.markdown("### ✅ Connected Sources")
        for src, conn in st.session_state.connections.items():
            st.code(f"{src} — Connected")