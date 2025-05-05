# streamlit_app.py
import streamlit as st
import json
import os
import logging
from dotenv import load_dotenv
from app.auth import login_user, signup_user, get_current_user
from app.agent import analyzer_chatbot, get_sqlalchemy_uri, get_db_schema, get_db_relationships
from app.data_source import upload_data_ui
from app.data_health import analyze_data_health
from app.collaboration import collaboration_tab
from app.settings import settings_ui
from app.reports import reports_dashboard
from datetime import datetime
from sqlalchemy import create_engine
from utils.helpers import build_predefined_dashboard, get_comprehensive_db_info


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic BI Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'role' not in st.session_state:
    st.session_state.role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'db_file' not in st.session_state:
    st.session_state.db_file = None
if 'goal' not in st.session_state:
    st.session_state.goal = 'Exploration'
if 'output_modes' not in st.session_state:
    st.session_state.output_modes = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'db_schema_info' not in st.session_state:
    st.session_state.db_schema_info = None
if 'connections' not in st.session_state:
    st.session_state.connections = {}
if 'active_dashboard' not in st.session_state:
    st.session_state.active_dashboard = None


def auto_data_health_check():
    """Automatically perform data health check for all connections"""
    if "connections" not in st.session_state or not st.session_state.connections:
        return

    if "data_health_report" not in st.session_state:
        st.session_state.data_health_report = {}

    for name, conn_info in st.session_state.connections.items():
        db_type = conn_info.get("type")
        if not db_type:
            continue

        try:
            if db_type == "sqlite":
                credentials = {"database": conn_info.get("database")}
            else:
                credentials = {
                    "user": conn_info.get("user"),
                    "password": conn_info.get("password"),
                    "host": conn_info.get("host"),
                    "port": conn_info.get("port"),
                    "database": conn_info.get("database")
                }

            uri = get_sqlalchemy_uri(db_type, credentials)
            report_df = analyze_data_health(uri, db_name=name)

            st.session_state.data_health_report[name] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "report": report_df
            }
            
            # Also analyze and store database schema for context
            engine = create_engine(uri)
            schema_info = get_db_schema(engine)
            relationships = get_db_relationships(schema_info)
            
            # Store schema information for agent context
            st.session_state.db_schema_info = {
                "schema": schema_info,
                "relationships": relationships
            }
            
            logger.info(f"Data health check completed for {name}")
            
        except Exception as e:
            logger.error(f"Error in data health check for {name}: {e}")
            st.session_state.data_health_report[name] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }


# Trigger data health check when connections are first set up
if st.session_state.get("connections") and "data_health_report" not in st.session_state:
    with st.spinner("Analyzing database schema..."):
        auto_data_health_check()


# Main Application
if not st.session_state.logged_in:
    # Login/Signup flow
    st.sidebar.title("Login / Signup")
    option = st.sidebar.radio("Choose", ["Login", "Signup"])
    if option == "Login":
        login_user()
    else:
        signup_user()
else:
    # Main application for logged-in users
    # Horizontal tab navigation
    tabs = ["Home", "Data Source", "Data Health", "Analyzer", "Reports Dashboard", "Collaboration", "Settings", "Logout"]
    selected_tab = st.radio("", tabs, horizontal=True)

    # Welcome message in sidebar
    with st.sidebar:
        st.success(f"Welcome {st.session_state.username} ({st.session_state.role})")
        
        # Show connection info if available
        if st.session_state.get("connections"):
            st.markdown("### Connected Databases")
            for name, info in st.session_state.connections.items():
                db_type = info.get("type", "unknown")
                db_name = os.path.basename(info.get("database", ""))
                st.markdown(f"📊 **{name}** ({db_type}): `{db_name}`")

    # Home tab
    if selected_tab == "Home":
        st.title("🏠 Home Dashboard")
        st.info("Welcome to Agentic BI. Select a tab to proceed.")
        
        # Split into columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 👉 Steps to use Agentic BI:
            1. Go to **Data Source** to upload your SQLite database.
            2. Head to the **Analyzer** tab to define your goal and output types.
            3. Ask questions, generate dashboards or insights.
            4. Save important results to the **Reports Dashboard**.
            5. Collaborate on findings from the **Collaboration** tab.
            """)
            
            if st.session_state.db_file:
                st.success(f"Database connected: {st.session_state.db_file}")
            else:
                st.warning("Upload a SQLite DB file from the Data Source tab to continue.")
        
        with col2:
            # Quick access section
            st.markdown("### Quick Access")
            if st.session_state.get("connections"):
                if st.button("🔍 Start Analyzing Data", use_container_width=True):
                    # Redirect to analyzer tab
                    st.session_state.active_tab = "Analyzer"
                    st.rerun()
                    
                if st.button("📊 View Reports", use_container_width=True):
                    # Redirect to reports tab
                    st.session_state.active_tab = "Reports Dashboard"
                    st.rerun()
            else:
                st.button("📁 Connect Database", disabled=False, use_container_width=True, 
                          help="Go to Data Source tab to connect a database")

    # Data Source tab
    elif selected_tab == "Data Source":
        with st.sidebar:
            st.markdown("### Upload SQLite DB")
            db_file = st.file_uploader("Upload SQLite DB", type=["db"])
            if db_file is not None:
                path = f"temp_uploaded/{db_file.name}"
                os.makedirs("temp_uploaded", exist_ok=True)
                with open(path, "wb") as f:
                    f.write(db_file.read())
                st.session_state.db_file = path
                
                # Add to connections for agent use
                if "connections" not in st.session_state:
                    st.session_state.connections = {}
                
                connection_name = db_file.name.split('.')[0]
                st.session_state.connections[connection_name] = {
                    "type": "sqlite",
                    "database": path
                }
                
                # Trigger schema analysis
                with st.spinner("Analyzing database schema..."):
                    try:
                        uri = get_sqlalchemy_uri("sqlite", {"database": path})
                        engine = create_engine(uri)
                        schema_info = get_db_schema(engine)
                        relationships = get_db_relationships(schema_info)
                        
                        # Store schema information for agent context
                        st.session_state.db_schema_info = {
                            "schema": schema_info,
                            "relationships": relationships
                        }
                        
                        st.success("Database uploaded and analyzed successfully")
                    except Exception as e:
                        logger.error(f"Error analyzing schema: {e}")
                        st.error(f"Error analyzing database schema: {e}")
                
        upload_data_ui()

    # Data Health tab
    elif selected_tab == "Data Health":
        st.title("Data Health Report")

        if "connections" not in st.session_state or not st.session_state.connections:
            st.error("Please upload a DB file first from the Data Source tab.")
            st.stop()

        selected_source = st.selectbox("Choose a connected source", list(st.session_state.connections.keys()))
        conn_info = st.session_state.connections[selected_source]
        db_type = conn_info.get("type")

        if db_type == "sqlite":
            credentials = {"database": conn_info.get("database")}
        else:
            credentials = {
                "user": conn_info.get("user"),
                "password": conn_info.get("password"),
                "host": conn_info.get("host"),
                "port": conn_info.get("port"),
                "database": conn_info.get("database")
            }

        conn_url = get_sqlalchemy_uri(db_type, credentials)

        with st.spinner("Analyzing data health..."):
            df = analyze_data_health(conn_url, db_name=selected_source)
            st.dataframe(df)

            # Add schema information if available
            if st.session_state.db_schema_info:
                st.subheader("Database Schema Information")
                
                # Show tables and their sizes
                schema_data = []
                for table_name, table_info in st.session_state.db_schema_info["schema"].items():
                    row_count = table_info.get("row_count", "Unknown")
                    column_count = len(table_info.get("columns", []))
                    schema_data.append({
                        "Table": table_name,
                        "Rows": row_count,
                        "Columns": column_count,
                        "Primary Keys": ", ".join(table_info.get("primary_keys", [])),
                        "Has Foreign Keys": "Yes" if table_info.get("foreign_keys") else "No"
                    })
                
                schema_df = pd.DataFrame(schema_data)
                st.dataframe(schema_df)

    # Analyzer tab
    elif selected_tab == "Analyzer":
        # Check for database connections
        connections = st.session_state.get("connections", {})
        if not connections:
            st.warning("Upload a database in **Data Source** tab first.")
            st.stop()
        
        # Call the analyzer chatbot with enhanced capabilities
        analyzer_chatbot()

    # Reports Dashboard tab
    elif selected_tab == "Reports Dashboard":
        reports_dashboard()

    # Collaboration tab
    elif selected_tab == "Collaboration":
        collaboration_tab()

    # Settings tab
    elif selected_tab == "Settings":
        settings_ui()

    # Logout tab
    elif selected_tab == "Logout":
        # Clear all session state variables
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.db_file = None
        st.session_state.goal = 'Exploration'
        st.session_state.output_modes = []
        st.session_state.chat_history = []
        st.session_state.db_schema_info = None
        st.session_state.active_dashboard = None
        
        # Rerun the app to show login screen
        st.rerun()

# Add footer with version info
st.sidebar.markdown("---")
st.sidebar.markdown("Agentic BI Platform v1.0.0")
st.sidebar.markdown("© 2025 Agentic BI")