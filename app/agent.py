# app/agent.py

import os
import json
import uuid
import logging
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy.engine import URL
from sqlalchemy import inspect, text
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import (
    plot_chart,
    export_to_excel,
    summarize_with_claude,
    save_to_dashboard,
    infer_chart_type,
    build_basic_dashboard,
    create_simple_dashboard
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

GOAL_PROMPTS = {
    "Exploration": "The user is exploring unknown datasets. Help them find patterns, top items, and anomalies.",
    "KPI Tracking": "The user wants to monitor key metrics like usage, retention, system performance.",
    "Decision Support": "The user needs help making a business decision. Identify tradeoffs, trends, and root causes."
}

def get_sqlalchemy_uri(db_type, credentials):
    if db_type == "sqlite":
        return f"sqlite:///{credentials['database']}"
    elif db_type == "postgres":
        return f"postgresql://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
    elif db_type == "mysql":
        return f"mysql+pymysql://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}"
    elif db_type == "snowflake":
        return URL.create(
            "snowflake",
            username=credentials['user'],
            password=credentials['password'],
            host=credentials['account'],
            database=credentials['database'],
            schema=credentials.get('schema', 'PUBLIC')
        )
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")

def save_to_dashboard_auto(query, goal, df, insight=None, chart_type=None):
    os.makedirs("dashboards", exist_ok=True)
    timestamp = datetime.now().isoformat()
    file_path = f"dashboards/report_{uuid.uuid4().hex}.json"
    data = {
        "query": query,
        "goal": goal,
        "chart_type": chart_type,
        "columns": list(df.columns),
        "preview": df.head(10).to_dict(orient="records"),
        "insight": insight,
        "saved_at": timestamp
    }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def get_db_schema(engine):
    """Extract database schema information to help with context"""
    inspector = inspect(engine)
    schema_info = {}
    
    for table_name in inspector.get_table_names():
        columns = []
        for column in inspector.get_columns(table_name):
            columns.append({
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column.get("nullable", True)
            })
        
        primary_keys = inspector.get_primary_keys(table_name)
        foreign_keys = []
        for fk in inspector.get_foreign_keys(table_name):
            foreign_keys.append({
                "constrained_columns": fk["constrained_columns"],
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"]
            })
        
        indexes = []
        for idx in inspector.get_indexes(table_name):
            indexes.append({
                "name": idx["name"],
                "columns": idx["column_names"],
                "unique": idx["unique"]
            })
        
        schema_info[table_name] = {
            "columns": columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "indexes": indexes
        }
        
        # Get sample data
        try:
            with engine.connect() as connection:
                sample_data = connection.execute(text(f"SELECT * FROM {table_name} LIMIT 5")).fetchall()
                if sample_data:
                    schema_info[table_name]["sample_data"] = [list(row) for row in sample_data]
        except Exception as e:
            logging.error(f"Error getting sample data for {table_name}: {e}")
    
    return schema_info

def get_db_relationships(schema_info):
    """Extract relationships between tables for better context understanding"""
    relationships = []
    
    for table_name, table_info in schema_info.items():
        for fk in table_info.get("foreign_keys", []):
            referred_table = fk.get("referred_table")
            if referred_table:
                relationships.append({
                    "from_table": table_name,
                    "from_columns": fk.get("constrained_columns", []),
                    "to_table": referred_table,
                    "to_columns": fk.get("referred_columns", [])
                })
    
    return relationships

def infer_best_visualization(df, question):
    """Determine the best visualization type based on data and question"""
    question = question.lower()
    
    # Check for distribution-related keywords
    if any(keyword in question for keyword in ["distribution", "histogram", "spread", "frequency"]):
        if len(df.columns) >= 1 and pd.api.types.is_numeric_dtype(df[df.columns[0]]):
            return "histogram"
    
    # Check for time series keywords
    time_keywords = ["time", "trend", "growth", "over time", "daily", "weekly", "monthly", "yearly"]
    if any(keyword in question for keyword in time_keywords):
        # Look for date/time columns
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col]) or 
                    ('date' in col.lower() or 'time' in col.lower() or 'day' in col.lower() or 
                     'month' in col.lower() or 'year' in col.lower())]
        if date_cols and len(df.columns) >= 2:
            return "line"
    
    # Check for comparison keywords
    if any(keyword in question for keyword in ["compare", "comparison", "versus", "vs", "against"]):
        if len(df.columns) >= 2:
            return "bar"
    
    # Check for relationship keywords
    if any(keyword in question for keyword in ["correlation", "relationship", "scatter", "between"]):
        if len(df.columns) >= 2 and all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns[:2]):
            return "scatter"
    
    # Check for proportion keywords
    if any(keyword in question for keyword in ["proportion", "percentage", "ratio", "share", "breakdown"]):
        if len(df.columns) >= 2:
            return "pie"
    
    # Check for geographic keywords
    if any(keyword in question for keyword in ["map", "regional", "geospatial", "country", "state", "city"]):
        # Check for geographic columns
        geo_cols = [col for col in df.columns if any(geo in col.lower() for geo in 
                                                 ["country", "state", "city", "region", "postal", "zip"])]
        if geo_cols:
            return "choropleth"
    
    # Fallback based on data structure
    if len(df.columns) == 2:
        if pd.api.types.is_numeric_dtype(df[df.columns[1]]):
            return "bar"
    elif len(df.columns) > 2:
        if all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns[1:]):
            return "bar"
    
    # Default to table if no better visualization is found
    return "table"

def create_advanced_visualization(df, query, chart_type=None):
    """Create more advanced visualizations based on data and query"""
    if chart_type is None:
        chart_type = infer_best_visualization(df, query)
    
    if chart_type == "table":
        return df
    
    # Ensure datetime columns are properly formatted
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    try:
        if chart_type == "histogram":
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_cols:
                fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
                return fig
        
        elif chart_type == "line":
            # Find potential date columns
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col]) or 
                        ('date' in col.lower() or 'time' in col.lower())]
            
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if date_cols and numeric_cols:
                x_col = date_cols[0]
                
                # Check if there's a categorical column for grouping
                categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and col != x_col]
                
                if categorical_cols and len(df[categorical_cols[0]].unique()) <= 10:
                    # Line chart with color by category
                    fig = px.line(df, x=x_col, y=numeric_cols[0], color=categorical_cols[0], 
                                 title=f"{numeric_cols[0]} over Time by {categorical_cols[0]}", markers=True)
                else:
                    # Simple line chart
                    fig = px.line(df, x=x_col, y=numeric_cols[:3], title=f"Time Series Analysis", markers=True)
                
                return fig
        
        elif chart_type == "bar":
            if len(df.columns) >= 2:
                # Find categorical column for x-axis
                cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
                num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                
                if cat_cols and num_cols:
                    x_col = cat_cols[0]
                    y_col = num_cols[0]
                    
                    # Check if there's another categorical column for grouping
                    if len(cat_cols) > 1:
                        color_col = cat_cols[1]
                        fig = px.bar(df, x=x_col, y=y_col, color=color_col, 
                                    title=f"{y_col} by {x_col}, grouped by {color_col}")
                    else:
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    
                    return fig
        
        elif chart_type == "scatter":
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if len(numeric_cols) >= 2:
                # Find categorical column for coloring
                cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
                
                if cat_cols:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=cat_cols[0],
                                    title=f"Relationship between {numeric_cols[0]} and {numeric_cols[1]} by {cat_cols[0]}")
                else:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                    title=f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}")
                
                # Add trendline
                fig.update_layout(showlegend=True)
                return fig
        
        elif chart_type == "pie":
            if len(df.columns) >= 2:
                # Find categorical and numeric columns
                cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
                num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                
                if cat_cols and num_cols:
                    fig = px.pie(df, names=cat_cols[0], values=num_cols[0], 
                                title=f"Distribution of {num_cols[0]} by {cat_cols[0]}")
                    return fig
        
        elif chart_type == "choropleth":
            # This is a placeholder as choropleth maps require specific geographic data
            # that may not be available in all datasets
            geo_cols = [col for col in df.columns if any(geo in col.lower() for geo in 
                                                    ["country", "state", "city", "region", "postal", "zip"])]
            num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if geo_cols and num_cols:
                st.warning("Geographic visualization requires specific map data that may not be available.")
                # Fallback to bar chart
                fig = px.bar(df, x=geo_cols[0], y=num_cols[0], title=f"{num_cols[0]} by {geo_cols[0]}")
                return fig
        
        # Dashboard type visualizations
        elif chart_type == "dashboard":
            if len(df.columns) >= 2:
                # Create a more complex dashboard with multiple charts
                # 1. Find date, categorical, and numeric columns
                date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col]) or 
                            ('date' in col.lower() or 'time' in col.lower())]
                cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and col not in date_cols]
                num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                
                # Create a subplot figure with multiple visualizations
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "Time Series" if date_cols else "Bar Chart",
                        "Distribution" if num_cols else "Category Breakdown",
                        "Comparison" if cat_cols and num_cols else "Summary",
                        "Details"
                    )
                )
                
                # Add time series if date column exists
                if date_cols and num_cols:
                    x_col = date_cols[0]
                    y_col = num_cols[0]
                    
                    # Group by date and calculate average
                    try:
                        ts_df = df.set_index(x_col)[y_col].resample('D').mean().reset_index()
                        fig.add_trace(
                            go.Scatter(x=ts_df[x_col], y=ts_df[y_col], mode='lines+markers'),
                            row=1, col=1
                        )
                    except:
                        fig.add_trace(
                            go.Scatter(x=df[x_col], y=df[y_col], mode='lines+markers'),
                            row=1, col=1
                        )
                else:
                    # Fallback to bar chart
                    if cat_cols and num_cols:
                        top_cats = df.groupby(cat_cols[0])[num_cols[0]].sum().nlargest(5).index
                        filtered_df = df[df[cat_cols[0]].isin(top_cats)]
                        fig.add_trace(
                            go.Bar(x=filtered_df[cat_cols[0]], y=filtered_df[num_cols[0]]),
                            row=1, col=1
                        )
                
                # Add histogram for numeric column
                if num_cols:
                    fig.add_trace(
                        go.Histogram(x=df[num_cols[0]]),
                        row=1, col=2
                    )
                else:
                    # Fallback to category counts
                    if cat_cols:
                        counts = df[cat_cols[0]].value_counts()
                        fig.add_trace(
                            go.Bar(x=counts.index, y=counts.values),
                            row=1, col=2
                        )
                
                # Add bar chart for category comparison
                if cat_cols and num_cols:
                    top_cats = df.groupby(cat_cols[0])[num_cols[0]].sum().nlargest(5).index
                    filtered_df = df[df[cat_cols[0]].isin(top_cats)]
                    fig.add_trace(
                        go.Bar(x=filtered_df[cat_cols[0]], y=filtered_df[num_cols[0]]),
                        row=2, col=1
                    )
                else:
                    # Fallback to summary statistics
                    if num_cols:
                        stats = df[num_cols[0]].describe()
                        fig.add_trace(
                            go.Bar(x=stats.index, y=stats.values),
                            row=2, col=1
                        )
                
                # Add table trace for details
                headers = df.columns.tolist()
                cells = [df[col].tolist()[:5] for col in headers]  # First 5 rows
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=headers),
                        cells=dict(values=cells)
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, title_text="Multi-chart Dashboard")
                return fig
        
        # Fallback to standard plot_chart function
        return plot_chart(df, query=query, chart_type=chart_type)
    
    except Exception as e:
        logging.error(f"Error creating visualization: {e}")
        return plot_chart(df, query=query, chart_type=chart_type)

def analyzer_chatbot() -> None:
    st.title("🧠 Data Analyzer")

    # Handle connection to data source
    conns = st.session_state.get("connections", {})
    if not conns:
        st.warning("No datasource connected. Please go to the Data Source tab.")
        return

    db_key = st.selectbox("📂 Data Source", list(conns.keys()))
    info = conns[db_key]
    st.success(f"Using **{db_key}** → `{os.path.basename(info['database'])}`")

    # Goal and output selection
    col1, col2 = st.columns(2)
    goals = col1.multiselect("🎯 Goals", ["Exploration", "KPI Tracking", "Decision Support"], 
                            default=st.session_state.get("goal", ["Exploration"]))
    outputs = col2.multiselect("📤 Outputs", 
                              ["data", "visualization", "insight", "dashboard"], 
                              default=st.session_state.get("output_modes", ["visualization"]))

    st.session_state.goal = goals
    st.session_state.output_modes = outputs

    # Initialize conversation memory if not present
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="human_input",
            output_key="ai_output"
        )

    # Display chat history
    hist = st.session_state.setdefault("chat_history", [])
    for msg in hist:
        st.chat_message(msg["role"]).markdown(msg["text"])

    # Setup the LLM
    llm = ChatAnthropic(
        model_name="claude-3-haiku-20240307",
        temperature=0.3,
        anthropic_api_key=ANTHROPIC_KEY
    )

    # Connect to the database
    uri = get_sqlalchemy_uri(info["type"], info)
    db = SQLDatabase.from_uri(uri)
    
    # Get user input
    user_q = st.chat_input("Ask your question about the data...")
    if not user_q:
        return

    st.chat_message("user").markdown(user_q)
    hist.append({"role": "user", "text": user_q})

    # Process the query with detailed goal context
    goal_context = f"Analysis goals: {', '.join(goals)}\n\n"
    prompt = goal_context + user_q

    # Use SQLDatabaseChain for query execution
    sql_chain = SQLDatabaseChain.from_llm(
        llm,
        db,
        verbose=False,
        return_intermediate_steps=True,
        top_k=100
    )

    with st.spinner("Analyzing data..."):
        try:
            # Execute query and get results
            out = sql_chain(prompt)
            sql_txt = None
            
            # Extract SQL from intermediate steps - but don't show to user
            for step in out.get("intermediate_steps", []):
                if isinstance(step, tuple):
                    cand_sql, _ = step
                elif isinstance(step, dict):
                    cand_sql = step.get("query") or step.get("sql") or step.get("statement")
                else:
                    continue
                
                if cand_sql and cand_sql.strip().lower().startswith(("select", "with")):
                    sql_txt = cand_sql
                    break

            # If no SQL query was found, provide a helpful response
            if not sql_txt:
                answer = out.get("result", "I'm not able to analyze that specific question with the available data. Could you try rephrasing or ask something else about the data?")
                st.chat_message("assistant").markdown(answer)
                hist.append({"role": "assistant", "text": answer})
                return

            # Execute the SQL query and get results as DataFrame
            df = pd.read_sql(sql_txt, db._engine)
            
            # Generate and display the response
            answer = out["result"]
            st.chat_message("assistant").markdown(answer)
            hist.append({"role": "assistant", "text": answer})

            # Output detection
            want_data = "data" in outputs
            want_vis = "visualization" in outputs
            want_ins = "insight" in outputs
            want_dash = "dashboard" in outputs

            # Display data table if requested
            if want_data:
                st.subheader("📄 Data Results")
                st.dataframe(df)

            # Create and display visualization if requested
            if want_vis and len(df.columns) >= 2:
                st.subheader("📊 Visualization")
                chart_type = infer_chart_type(user_q)
                fig = plot_chart(df, user_q, chart_type)
                
                if hasattr(fig, 'update_layout'):
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # If fig is a DataFrame, display it
                    st.dataframe(fig)

            # Generate and display insights if requested
            if want_ins:
                st.subheader("🧠 Insights")
                insight_txt = summarize_with_claude(df, user_q, ", ".join(goals))
                st.write(insight_txt)

            # Create dashboard if requested
            if want_dash:
                st.subheader("📋 Dashboard")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Key Metrics", "Data Details", "Analysis"])
                
                with tab1:
                    # Identify column types
                    categorical_cols = []
                    numerical_cols = []
                    datetime_cols = []
                    
                    for col in df.columns:
                        try:
                            if pd.api.types.is_datetime64_dtype(df[col]) or (
                                isinstance(df[col].iloc[0], str) and 
                                any(date_term in col.lower() for date_term in ['date', 'time', 'day'])
                            ):
                                datetime_cols.append(col)
                                # Try to convert to datetime if it's not already
                                if not pd.api.types.is_datetime64_dtype(df[col]):
                                    df[col] = pd.to_datetime(df[col], errors='ignore')
                            elif pd.api.types.is_numeric_dtype(df[col]):
                                numerical_cols.append(col)
                            else:
                                categorical_cols.append(col)
                        except:
                            categorical_cols.append(col)
                    
                    # Display key metrics in columns if numerical data available
                    if numerical_cols:
                        metric_cols = st.columns(min(3, len(numerical_cols)))
                        for i, col in enumerate(numerical_cols[:3]):
                            with metric_cols[i % 3]:
                                avg_val = df[col].mean()
                                max_val = df[col].max()
                                st.metric(
                                    label=col.replace('_', ' ').title(),
                                    value=f"{avg_val:.2f}",
                                    delta=f"{max_val - avg_val:.2f} (max)"
                                )
                    
                    # Create primary visualization based on data types
                    if categorical_cols and numerical_cols:
                        # Bar chart
                        fig1 = px.bar(
                            df, 
                            x=categorical_cols[0], 
                            y=numerical_cols[0],
                            color=categorical_cols[1] if len(categorical_cols) > 1 else None,
                            title=f"{numerical_cols[0]} by {categorical_cols[0]}"
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    elif datetime_cols and numerical_cols:
                        # Line chart
                        fig1 = px.line(
                            df, 
                            x=datetime_cols[0], 
                            y=numerical_cols[0],
                            markers=True,
                            title=f"{numerical_cols[0]} Over Time"
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    elif len(numerical_cols) >= 2:
                        # Scatter plot
                        fig1 = px.scatter(
                            df, 
                            x=numerical_cols[0], 
                            y=numerical_cols[1],
                            title=f"Relationship between {numerical_cols[0]} and {numerical_cols[1]}"
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                
                with tab2:
                    # Show detailed data table
                    st.dataframe(df)
                    
                    # Show summary statistics if numerical columns exist
                    if numerical_cols:
                        st.subheader("Summary Statistics")
                        st.dataframe(df[numerical_cols].describe())
                
                with tab3:
                    # Additional analysis based on data
                    if numerical_cols:
                        # Distribution of a numerical column
                        selected_col = numerical_cols[0]
                        fig3 = px.histogram(
                            df, 
                            x=selected_col, 
                            title=f"Distribution of {selected_col}"
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    if categorical_cols:
                        # Distribution of a categorical column
                        selected_cat = categorical_cols[0]
                        cat_counts = df[selected_cat].value_counts().reset_index()
                        cat_counts.columns = [selected_cat, 'count']
                        fig4 = px.pie(
                            cat_counts, 
                            values='count', 
                            names=selected_cat, 
                            title=f"Breakdown by {selected_cat}"
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                
                # Add option to save dashboard
                if st.button("💾 Save Dashboard"):
                    dashboard_name = f"Dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    path = save_to_dashboard(dashboard_name, user_q, df)
                    st.success(f"Dashboard saved as {dashboard_name}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            hist.append({"role": "assistant", "text": f"I encountered an error while analyzing your request. Could you please try a simpler question or provide more specific details?"})