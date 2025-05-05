# utils/helpers.py
import os
import uuid
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import streamlit as st
import io
import re
from sqlalchemy import inspect, text

load_dotenv()
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

# Chart type mapping dictionary
CHART_TYPE_MAP = {
    "retention": "line", "signup": "line", "conversion": "funnel", 
    "comparison": "bar", "breakdown": "pie", "revenue": "line", 
    "distribution": "histogram", "correlation": "scatter", 
    "trend": "line", "time series": "line", 
    "proportion": "pie", "percentage": "pie",
    "ranking": "bar", "top": "bar"
}

def infer_chart_type(query):
    """Determine the best chart type based on the query"""
    if not query:
        return "bar"  # Default
        
    query = query.lower()
    
    # Check for specific chart types in query
    for keyword, chart_type in CHART_TYPE_MAP.items():
        if keyword in query:
            return chart_type
    
    # Pattern-based inference
    if re.search(r"(over|across|during).*(time|period|year|month|week|day)", query):
        return "line"
    elif re.search(r"(compare|comparison|versus|vs)", query):
        return "bar"
    elif re.search(r"(distribution|histogram|frequency)", query):
        return "histogram"
    elif re.search(r"(correlation|relationship|between)", query):
        return "scatter"
    elif re.search(r"(breakdown|percentage|ratio|share)", query):
        return "pie"
    
    # Default to bar chart
    return "bar"

def plot_chart(df, query=None, chart_type=None):
    """Create appropriate visualization based on the dataframe and chart type"""
    # Use inferred chart type if none provided
    chart_type = chart_type or infer_chart_type(query)
    
    # Pre-process the dataframe
    # Try to convert date columns to datetime
    for col in df.columns:
        if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']):
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    # Identify column types
    categorical_cols = []
    numerical_cols = []
    datetime_cols = []
    
    for col in df.columns:
        if pd.api.types.is_datetime64_dtype(df[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    try:
        # PIE CHART
        if chart_type in ["pie", "donut"]:
            if len(df.columns) >= 2:
                name_col = categorical_cols[0] if categorical_cols else df.columns[0]
                value_col = numerical_cols[0] if numerical_cols else df.columns[1]
                
                # Limit to top categories for readability
                top_df = df.groupby(name_col)[value_col].sum().nlargest(8).reset_index()
                
                fig = px.pie(
                    top_df, 
                    values=value_col, 
                    names=name_col, 
                    hole=0.4 if chart_type == "donut" else 0,
                    title=f"Distribution of {value_col} by {name_col}"
                )
            else:
                # If only one column, use value counts
                counts = df[df.columns[0]].value_counts().reset_index()
                counts.columns = ["category", "count"]
                fig = px.pie(
                    counts, 
                    values="count", 
                    names="category", 
                    hole=0.4 if chart_type == "donut" else 0,
                    title=f"Distribution of {df.columns[0]}"
                )
        
        # BAR CHART
        elif chart_type == "bar":
            if categorical_cols and numerical_cols:
                # Simple bar chart
                x_col = categorical_cols[0]
                y_col = numerical_cols[0]
                
                # Limit to top categories for readability
                if len(df[x_col].unique()) > 10:
                    top_cats = df.groupby(x_col)[y_col].sum().nlargest(10).index
                    filtered_df = df[df[x_col].isin(top_cats)]
                else:
                    filtered_df = df
                
                fig = px.bar(
                    filtered_df, 
                    x=x_col, 
                    y=y_col,
                    title=f"{y_col} by {x_col}",
                    color=categorical_cols[1] if len(categorical_cols) > 1 else None
                )
            else:
                # Basic bar chart
                fig = px.bar(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                    title="Bar Chart"
                )
        
        # LINE CHART
        elif chart_type == "line":
            if datetime_cols and numerical_cols:
                # Time series chart
                x_col = datetime_cols[0]
                y_cols = numerical_cols[:3]  # Limit to 3 lines for readability
                
                # Check if we have a categorical column for grouping
                if categorical_cols and len(df[categorical_cols[0]].unique()) <= 5:
                    color_col = categorical_cols[0]
                    fig = px.line(
                        df, 
                        x=x_col, 
                        y=y_cols[0], 
                        color=color_col,
                        markers=True,
                        title=f"{y_cols[0]} Over Time by {color_col}"
                    )
                else:
                    fig = px.line(
                        df, 
                        x=x_col, 
                        y=y_cols if len(y_cols) <= 3 else y_cols[:3], 
                        markers=True,
                        title="Time Series Analysis"
                    )
            elif numerical_cols and len(numerical_cols) >= 2:
                # Line chart with numeric x-axis
                x_col = numerical_cols[0]
                y_col = numerical_cols[1]
                
                fig = px.line(
                    df, 
                    x=x_col, 
                    y=y_col,
                    markers=True,
                    title=f"{y_col} vs {x_col}"
                )
            else:
                # Basic line chart
                fig = px.line(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1:min(4, len(df.columns))],
                    markers=True,
                    title="Line Chart"
                )
        
        # SCATTER CHART
        elif chart_type == "scatter":
            if len(numerical_cols) >= 2:
                x_col = numerical_cols[0]
                y_col = numerical_cols[1]
                
                # Add color dimension if available
                if categorical_cols:
                    color_col = categorical_cols[0]
                    fig = px.scatter(
                        df, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=f"Relationship between {x_col} and {y_col} by {color_col}"
                    )
                else:
                    fig = px.scatter(
                        df, 
                        x=x_col, 
                        y=y_col,
                        title=f"Relationship between {x_col} and {y_col}"
                    )
                
                # Add trendline
                fig.update_layout(showlegend=True)
            else:
                # Fallback to basic scatter
                fig = px.scatter(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                    title="Scatter Plot"
                )
        
        # HISTOGRAM
        elif chart_type == "histogram":
            if numerical_cols:
                x_col = numerical_cols[0]
                
                if categorical_cols:
                    # Histogram with color grouping
                    color_col = categorical_cols[0]
                    fig = px.histogram(
                        df, 
                        x=x_col, 
                        color=color_col,
                        barmode="overlay",
                        opacity=0.7,
                        title=f"Distribution of {x_col} by {color_col}"
                    )
                else:
                    # Simple histogram
                    fig = px.histogram(
                        df, 
                        x=x_col,
                        title=f"Distribution of {x_col}"
                    )
            else:
                # Fallback
                fig = px.histogram(
                    df, 
                    x=df.columns[0],
                    title=f"Histogram of {df.columns[0]}"
                )
        
        # FUNNEL CHART
        elif chart_type == "funnel":
            if len(df.columns) >= 2:
                stage_col = categorical_cols[0] if categorical_cols else df.columns[0]
                value_col = numerical_cols[0] if numerical_cols else df.columns[1]
                
                # Sort by value to create proper funnel
                df = df.sort_values(value_col, ascending=False)
                fig = px.funnel(df, x=value_col, y=stage_col)
            else:
                # If only one column, use it as stages and count occurrences
                counts = df[df.columns[0]].value_counts().reset_index()
                counts.columns = ["stage", "count"]
                fig = px.funnel(counts, x="count", y="stage")
                
        # HEATMAP
        elif chart_type == "heatmap":
            if len(numerical_cols) >= 2:
                # Create correlation heatmap for numerical columns
                corr_df = df[numerical_cols].corr()
                fig = px.imshow(
                    corr_df, 
                    text_auto=True, 
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap"
                )
            else:
                # Fallback to correlation of all columns
                fig = px.imshow(
                    df.corr(), 
                    text_auto=True,
                    title="Data Correlation"
                )
                
        # TABLE
        elif chart_type == "table":
            # Return the dataframe itself for table display
            return df.head(15)
            
        # Default to bar chart if no other match
        else:
            if categorical_cols and numerical_cols:
                x_col = categorical_cols[0]
                y_col = numerical_cols[0]
                
                # Limit to top categories for readability
                if len(df[x_col].unique()) > 10:
                    top_cats = df.groupby(x_col)[y_col].sum().nlargest(10).index
                    filtered_df = df[df[x_col].isin(top_cats)]
                else:
                    filtered_df = df
                
                fig = px.bar(
                    filtered_df, 
                    x=x_col, 
                    y=y_col,
                    title=f"{y_col} by {x_col}"
                )
            else:
                # Basic bar chart
                fig = px.bar(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                    title="Bar Chart"
                )
                
        # Apply common layout improvements
        if hasattr(fig, 'update_layout'):
            fig.update_layout(
                margin=dict(l=40, r=40, t=60, b=40),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )
            
        return fig
    except Exception as e:
        st.error(f"❌ Failed to plot chart: {str(e)}")
        # Return basic table as fallback
        return df.head(15)

def create_simple_dashboard(df, title="Interactive Dashboard"):
    """Create a simple dashboard with multiple visualizations"""
    try:
        # Identify column types
        categorical_cols = []
        numerical_cols = []
        datetime_cols = []
        
        for col in df.columns:
            if pd.api.types.is_datetime64_dtype(df[col]):
                datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Create a multi-chart dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Overview", "Distribution", "Time Trend", "Details")
        )
        
        # Overview chart
        if categorical_cols and numerical_cols:
            x_col = categorical_cols[0]
            y_col = numerical_cols[0]
            
            # Limit to top 5 categories for readability
            if len(df[x_col].unique()) > 5:
                top_cats = df.groupby(x_col)[y_col].sum().nlargest(5).index
                filtered_df = df[df[x_col].isin(top_cats)]
            else:
                filtered_df = df
                
            fig.add_trace(
                go.Bar(
                    x=filtered_df[x_col],
                    y=filtered_df[y_col],
                    name="Overview"
                ),
                row=1, col=1
            )
        
        # Distribution chart
        if numerical_cols:
            fig.add_trace(
                go.Histogram(
                    x=df[numerical_cols[0]],
                    name="Distribution"
                ),
                row=1, col=2
            )
        
        # Time trend chart
        if datetime_cols and numerical_cols:
            fig.add_trace(
                go.Scatter(
                    x=df[datetime_cols[0]],
                    y=df[numerical_cols[0]],
                    mode="lines+markers",
                    name="Time Trend"
                ),
                row=2, col=1
            )
        
        # Details table
        headers = df.columns.tolist()
        cells = [df[col].head(5).tolist() for col in headers]
        
        fig.add_trace(
            go.Table(
                header=dict(values=headers),
                cells=dict(values=cells)
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text=title)
        return fig
    
    except Exception as e:
        st.error(f"❌ Error creating dashboard: {str(e)}")
        return None

def summarize_with_claude(df, query, goal):
    """Generate insights based on data and query using Claude"""
    try:
        # Prepare data sample and statistics
        sample = df.head(5).to_string()
        stats = df.describe().to_string() if df.select_dtypes(include=['number']).shape[1] > 0 else ""
        
        # Create prompt for better insights
        prompt = f"""
        A user asked this query: '{query}' with analysis goal: {goal}.
        
        Here's a sample of the data (first 5 rows):
        {sample}
        
        Basic statistics of the numerical columns:
        {stats}
        
        Based on this data, provide 3-4 clear business insights that would be valuable for the user's query and goal.
        Focus on patterns, anomalies, trends, and actionable insights.
        Format as bullet points with each insight being 1-2 sentences.
        """
        
        # Use Claude to generate insights
        llm = ChatAnthropic(model="claude-3-haiku-20240307", anthropic_api_key=anthropic_key)
        insights = llm.invoke(prompt)
        
        return insights
    except Exception as e:
        return f"❌ Could not generate insights: {str(e)}"

def export_to_excel(df: pd.DataFrame) -> io.BytesIO:
    """Return an in-memory Excel file for Streamlit download"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        
        # Auto-adjust column widths
        worksheet = writer.sheets["Results"]
        for i, col in enumerate(df.columns):
            max_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + i)].width = min(max_length, 30)  # Limit width to 30
    
    buffer.seek(0)
    return buffer

def save_to_dashboard(name, query, df):
    """Save dashboard configuration to file for future reference"""
    os.makedirs("dashboards", exist_ok=True)
    timestamp = datetime.now().isoformat()
    path = f"dashboards/{name.replace(' ', '_').lower()}.json"
    
    # Detect column types for better visualization
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_dtype(df[col]):
            column_types[col] = "datetime"
        else:
            column_types[col] = "categorical"
    
    # Create dashboard entry with metadata
    entry = {
        "name": name,
        "query": query,
        "columns": list(df.columns),
        "column_types": column_types,
        "preview": df.head(10).to_dict(orient="records"),
        "timestamp": timestamp,
        "stats": df.describe().to_dict() if df.select_dtypes(include=['number']).shape[1] > 0 else {}
    }
    
    with open(path, "w") as f:
        json.dump(entry, f, indent=2)
    
    return path

def get_table_info(engine):
    """Get basic information about database tables"""
    inspector = inspect(engine)
    tables_info = {}
    
    for table_name in inspector.get_table_names():
        columns = []
        for column in inspector.get_columns(table_name):
            columns.append({
                "name": column["name"],
                "type": str(column["type"])
            })
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.scalar()
        except:
            row_count = "Unknown"
        
        tables_info[table_name] = {
            "columns": columns,
            "row_count": row_count
        }
    
    return tables_info

def build_basic_dashboard(dashboard_type, engine):
    """Build a basic dashboard based on database tables"""
    st.markdown(f"## 📊 {dashboard_type}")
    
    try:
        # Get table names
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            
        if not tables:
            st.error("No tables found in the database")
            return
        
        # Event table analysis
        event_table = next((t for t in tables if "event" in t.lower()), None)
        if event_table and dashboard_type in ["Event Analysis", "User Activity"]:
            st.subheader(f"📊 {event_table} Analysis")
            
            # Event type distribution
            try:
                df = pd.read_sql(f"""
                    SELECT event_type, COUNT(*) AS count 
                    FROM {event_table} GROUP BY event_type ORDER BY count DESC
                """, engine)
                
                fig = px.pie(df, values='count', names='event_type', title='Event Type Distribution')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating event distribution chart: {e}")
            
            # Event trends over time (if time column exists)
            try:
                time_col = next((c[0] for c in engine.execute(f"PRAGMA table_info({event_table})").fetchall() 
                               if 'time' in c[1].lower() or 'date' in c[1].lower()), None)
                
                if time_col:
                    trend_df = pd.read_sql(f"""
                        SELECT strftime('%Y-%m-%d', {time_col}) AS day, COUNT(*) AS events
                        FROM {event_table} GROUP BY day ORDER BY day
                    """, engine)
                    
                    fig = px.line(trend_df, x='day', y='events', title='Daily Event Trend', markers=True)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating time trend chart: {e}")
        
        # User table analysis
        user_table = next((t for t in tables if "user" in t.lower()), None)
        if user_table and dashboard_type in ["User Analysis", "User Metrics"]:
            st.subheader(f"👥 {user_table} Analysis")
            
            # Show user counts
            try:
                count = pd.read_sql(f"SELECT COUNT(*) AS total_users FROM {user_table}", engine).iloc[0, 0]
                st.metric("Total Users", count)
                
                # Try to find user attributes for grouping
                columns = [row[1] for row in engine.execute(f"PRAGMA table_info({user_table})").fetchall()]
                
                for col in ['role', 'type', 'status', 'country', 'plan']:
                    if any(col in column.lower() for column in columns):
                        group_col = next(column for column in columns if col in column.lower())
                        
                        group_df = pd.read_sql(f"""
                            SELECT {group_col}, COUNT(*) AS count
                            FROM {user_table} GROUP BY {group_col} ORDER BY count DESC
                        """, engine)
                        
                        fig = px.bar(group_df, x=group_col, y='count', title=f'Users by {group_col}')
                        st.plotly_chart(fig, use_container_width=True)
                        break
            except Exception as e:
                st.error(f"Error analyzing user data: {e}")
        
        # Show table information if no specific dashboard type matched
        if not (event_table or user_table) or dashboard_type in ["Table Info", "Database Overview"]:
            st.subheader("📋 Database Tables")
            
            table_info = []
            for table in tables:
                try:
                    count = pd.read_sql(f"SELECT COUNT(*) AS count FROM {table}", engine).iloc[0, 0]
                    columns = [row[1] for row in engine.execute(f"PRAGMA table_info({table})").fetchall()]
                    
                    table_info.append({
                        "Table": table,
                        "Rows": count,
                        "Columns": len(columns),
                        "Column Names": ", ".join(columns[:5]) + ("..." if len(columns) > 5 else "")
                    })
                except Exception:
                    continue
            
            st.table(pd.DataFrame(table_info))
    
    except Exception as e:
        st.error(f"❌ Error building dashboard: {str(e)}")

def export_chart_image(fig, file_format="png"):
    """Export chart as an image file"""
    os.makedirs("static/images", exist_ok=True)
    file_path = f"static/images/chart_{uuid.uuid4()}.{file_format}"
    
    try:
        if hasattr(fig, 'write_image'):
            fig.write_image(file_path, format=file_format)
            return file_path
        else:
            return f"❌ Cannot export this visualization type"
    except Exception as e:
        return f"❌ Export image error: {str(e)}"

def export_chart_pdf(fig):
    """Export chart as a PDF document"""
    os.makedirs("static/pdfs", exist_ok=True)
    image_path = export_chart_image(fig, file_format="png")
    
    if image_path.startswith("❌"):
        return image_path

    pdf_path = f"static/pdfs/chart_{uuid.uuid4()}.pdf"
    
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Data Visualization", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.image(image_path, x=10, y=30, w=190)
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        return f"❌ PDF export error: {str(e)}"
def infer_chart_type(query):
    """Determine the best chart type based on the query"""
    query = query.lower()
    
    # Search for keywords that match chart types
    for keyword, chart_type in CHART_TYPE_MAP.items():
        if keyword in query:
            return chart_type
    
    # More advanced inference based on query patterns
    if re.search(r"(over|across|during|throughout).*(time|period|year|month|week|day)", query):
        return "line"
    elif re.search(r"(compare|comparison|versus|vs|against)", query):
        return "bar"
    elif re.search(r"(distribution|histogram|frequency|spread)", query):
        return "histogram"
    elif re.search(r"(correlation|relationship|between|scatter)", query):
        return "scatter"
    elif re.search(r"(breakdown|percentage|ratio|composition|share)", query):
        return "pie"
    elif re.search(r"(funnel|conversion|journey|steps)", query):
        return "funnel"
    elif re.search(r"(dashboard|comprehensive|complete|full|report)", query):
        return "dashboard"
    
    # Default to bar chart for most queries
    return "bar"

def plot_chart(df, query=None, chart_type=None):
    """Create a visualization based on the dataframe and chart type"""
    chart_type = chart_type or infer_chart_type(query)
    
    # Pre-process the dataframe
    # Try to convert date columns to datetime
    for col in df.columns:
        if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']):
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    # Try to identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []
    datetime_cols = []
    
    for col in df.columns:
        if pd.api.types.is_datetime64_dtype(df[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    try:
        if chart_type == "funnel":
            # For funnel charts, we need a stage and value columns
            if len(df.columns) >= 2:
                stage_col = categorical_cols[0] if categorical_cols else df.columns[0]
                value_col = numerical_cols[0] if numerical_cols else df.columns[1]
                
                # Sort by value to create proper funnel
                df = df.sort_values(value_col, ascending=False)
                fig = px.funnel(df, x=value_col, y=stage_col)
            else:
                # If only one column, use it as stages and count occurrences
                counts = df[df.columns[0]].value_counts().reset_index()
                counts.columns = ["stage", "count"]
                fig = px.funnel(counts, x="count", y="stage")
        
        elif chart_type == "donut" or chart_type == "pie":
            if len(df.columns) >= 2:
                name_col = categorical_cols[0] if categorical_cols else df.columns[0]
                value_col = numerical_cols[0] if numerical_cols else df.columns[1]
                
                # Limit to top 10 categories for readability
                top_df = df.groupby(name_col)[value_col].sum().nlargest(10).reset_index()
                
                fig = px.pie(
                    top_df, 
                    values=value_col, 
                    names=name_col, 
                    hole=0.4 if chart_type == "donut" else 0,
                    title=f"Distribution of {value_col} by {name_col}"
                )
            else:
                # If only one column, use value counts
                counts = df[df.columns[0]].value_counts().reset_index()
                counts.columns = ["category", "count"]
                fig = px.pie(
                    counts, 
                    values="count", 
                    names="category", 
                    hole=0.4 if chart_type == "donut" else 0,
                    title=f"Distribution of {df.columns[0]}"
                )
        
        elif chart_type == "heatmap":
            if len(numerical_cols) >= 2:
                # Create correlation heatmap for numerical columns
                corr_df = df[numerical_cols].corr()
                fig = px.imshow(
                    corr_df, 
                    text_auto=True, 
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap"
                )
            elif len(categorical_cols) >= 2 and len(numerical_cols) >= 1:
                # Create a crosstab heatmap
                pivot = pd.crosstab(
                    df[categorical_cols[0]], 
                    df[categorical_cols[1]],
                    values=df[numerical_cols[0]] if numerical_cols else None,
                    aggfunc='mean' if numerical_cols else 'count'
                )
                fig = px.imshow(
                    pivot,
                    text_auto=True,
                    color_continuous_scale="Viridis",
                    title=f"{categorical_cols[0]} vs {categorical_cols[1]}"
                )
            else:
                # Fallback to correlation of all columns
                fig = px.imshow(
                    df.corr(), 
                    text_auto=True,
                    title="Data Correlation"
                )
        
        elif chart_type == "calendar_heatmap":
            return plot_calendar_heatmap(df, datetime_cols[0] if datetime_cols else df.columns[0], numerical_cols[0] if numerical_cols else df.columns[1])
        
        elif chart_type == "sankey":
            return plot_sankey(df)
        
        elif chart_type == "line":
            if datetime_cols and numerical_cols:
                # Time series chart
                x_col = datetime_cols[0]
                y_cols = numerical_cols[:3]  # Limit to 3 lines for readability
                
                # Check if we have a categorical column for grouping
                if categorical_cols:
                    color_col = categorical_cols[0]
                    # Limit to top 5 categories
                    top_cats = df[color_col].value_counts().nlargest(5).index
                    filtered_df = df[df[color_col].isin(top_cats)]
                    
                    fig = px.line(
                        filtered_df, 
                        x=x_col, 
                        y=y_cols[0], 
                        color=color_col,
                        markers=True,
                        title=f"{y_cols[0]} Over Time by {color_col}"
                    )
                else:
                    fig = px.line(
                        df, 
                        x=x_col, 
                        y=y_cols, 
                        markers=True,
                        title="Time Series Analysis"
                    )
            elif numerical_cols and len(numerical_cols) >= 2:
                # Line chart with numeric x-axis
                x_col = numerical_cols[0]
                y_col = numerical_cols[1]
                
                fig = px.line(
                    df, 
                    x=x_col, 
                    y=y_col,
                    markers=True,
                    title=f"{y_col} vs {x_col}"
                )
            else:
                # Basic line chart
                fig = px.line(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1:min(4, len(df.columns))],
                    markers=True,
                    title="Line Chart"
                )
        
        elif chart_type == "scatter":
            if len(numerical_cols) >= 2:
                x_col = numerical_cols[0]
                y_col = numerical_cols[1]
                
                # Add color dimension if available
                if categorical_cols:
                    color_col = categorical_cols[0]
                    fig = px.scatter(
                        df, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=f"Relationship between {x_col} and {y_col} by {color_col}"
                    )
                else:
                    fig = px.scatter(
                        df, 
                        x=x_col, 
                        y=y_col,
                        title=f"Relationship between {x_col} and {y_col}"
                    )
                
                # Add trendline
                fig.update_layout(showlegend=True)
            else:
                # Fallback to basic scatter
                fig = px.scatter(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                    title="Scatter Plot"
                )
        
        elif chart_type == "box":
            if numerical_cols:
                y_col = numerical_cols[0]
                
                if categorical_cols:
                    # Box plot grouped by category
                    x_col = categorical_cols[0]
                    fig = px.box(
                        df, 
                        x=x_col, 
                        y=y_col,
                        title=f"Distribution of {y_col} by {x_col}"
                    )
                else:
                    # Simple box plot
                    fig = px.box(
                        df, 
                        y=y_col,
                        title=f"Distribution of {y_col}"
                    )
            else:
                # Fallback
                fig = px.box(
                    df, 
                    x=df.columns[0] if len(df.columns) > 1 else None, 
                    y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                    title="Box Plot"
                )
        
        elif chart_type == "histogram":
            if numerical_cols:
                x_col = numerical_cols[0]
                
                if categorical_cols:
                    # Histogram with color grouping
                    color_col = categorical_cols[0]
                    fig = px.histogram(
                        df, 
                        x=x_col, 
                        color=color_col,
                        barmode="overlay",
                        opacity=0.7,
                        title=f"Distribution of {x_col} by {color_col}"
                    )
                else:
                    # Simple histogram
                    fig = px.histogram(
                        df, 
                        x=x_col,
                        title=f"Distribution of {x_col}"
                    )
            else:
                # Fallback
                fig = px.histogram(
                    df, 
                    x=df.columns[0],
                    title=f"Histogram of {df.columns[0]}"
                )
        
        elif chart_type == "choropleth":
            # This is a placeholder as choropleth maps require specific geographic data
            st.warning("Geographic visualization requires specific map data that may not be available.")
            # Fallback to bar chart
            if categorical_cols and numerical_cols:
                fig = px.bar(
                    df, 
                    x=categorical_cols[0], 
                    y=numerical_cols[0],
                    title=f"{numerical_cols[0]} by {categorical_cols[0]}"
                )
            else:
                fig = px.bar(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                    title="Bar Chart (Fallback from Choropleth)"
                )
        
        elif chart_type == "table":
            # Return the dataframe itself for table display
            return df.head(15)
        
        elif chart_type == "dashboard":
            # Create a multi-chart dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Overview", "Distribution", "Time Trend", "Details")
            )
            
            # Add charts based on available data
            if categorical_cols and numerical_cols:
                # Bar chart for overview
                top_cats = df.groupby(categorical_cols[0])[numerical_cols[0]].sum().nlargest(5).index
                filtered_df = df[df[categorical_cols[0]].isin(top_cats)]
                
                fig.add_trace(
                    go.Bar(
                        x=filtered_df[categorical_cols[0]],
                        y=filtered_df[numerical_cols[0]],
                        name="Overview"
                    ),
                    row=1, col=1
                )
            
            if numerical_cols:
                # Histogram for distribution
                fig.add_trace(
                    go.Histogram(
                        x=df[numerical_cols[0]],
                        name="Distribution"
                    ),
                    row=1, col=2
                )
            
            if datetime_cols and numerical_cols:
                # Time series for trend
                fig.add_trace(
                    go.Scatter(
                        x=df[datetime_cols[0]],
                        y=df[numerical_cols[0]],
                        mode="lines+markers",
                        name="Time Trend"
                    ),
                    row=2, col=1
                )
            
            # Table for details
            headers = df.columns.tolist()
            cells = [df[col].head(5).tolist() for col in headers]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=headers),
                    cells=dict(values=cells)
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="Interactive Dashboard")
        
        else:  # Default to bar chart
            if categorical_cols and numerical_cols:
                # Simple bar chart
                x_col = categorical_cols[0]
                y_col = numerical_cols[0]
                
                # Limit to top 10 categories for readability
                top_cats = df.groupby(x_col)[y_col].sum().nlargest(10).index
                filtered_df = df[df[x_col].isin(top_cats)]
                
                fig = px.bar(
                    filtered_df, 
                    x=x_col, 
                    y=y_col,
                    title=f"{y_col} by {x_col}"
                )
            else:
                # Basic bar chart
                fig = px.bar(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                    title="Bar Chart"
                )
                
        # Apply common layout improvements
        if hasattr(fig, 'update_layout'):
            fig.update_layout(
                margin=dict(l=40, r=40, t=60, b=40),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )
            
        return fig
    except Exception as e:
        st.error(f"❌ Failed to plot chart: {str(e)}")
        # Return basic table as fallback
        return df.head(15)

def plot_sankey(df):
    """Create a Sankey diagram visualization for flow data"""
    try:
        # Ensure we have at least three columns for source, target, and value
        if len(df.columns) < 3:
            # Try to get value counts for two columns if only two are provided
            if len(df.columns) == 2:
                # Create a count column
                value_df = df.copy()
                value_df['count'] = 1
                grouped = value_df.groupby([df.columns[0], df.columns[1]]).count().reset_index()
                labels = list(pd.concat([grouped[df.columns[0]], grouped[df.columns[1]]]).unique())
                label_idx = {l: i for i, l in enumerate(labels)}
                sources = grouped[df.columns[0]].map(label_idx)
                targets = grouped[df.columns[1]].map(label_idx)
                values = grouped['count']
            else:
                st.error("Need at least two columns for a Sankey diagram")
                return None
        else:
            # Use the provided three columns
            labels = list(pd.concat([df.iloc[:, 0], df.iloc[:, 1]]).unique())
            label_idx = {l: i for i, l in enumerate(labels)}
            sources = df.iloc[:, 0].map(label_idx)
            targets = df.iloc[:, 1].map(label_idx)
            values = df.iloc[:, 2]

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        
        fig.update_layout(
            title_text="Flow Diagram",
            font_size=12,
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Sankey plot error: {str(e)}")
        return None

def build_predefined_dashboard(dashboard_type: str, engine):
    """Build a predefined dashboard based on dashboard type"""
    st.markdown(f"## 📊 {dashboard_type}")
    
    # Get all tables from the database
    tables = []
    try:
        # Get table names
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            
        if not tables:
            st.error("No tables found in the database")
            return
            
        # Find key tables based on naming conventions
        event_table = next((t for t in tables if "event" in t.lower()), None)
        user_table = next((t for t in tables if "user" in t.lower()), None)
        content_table = next((t for t in tables if "content" in t.lower()), None)
        account_table = next((t for t in tables if "account" in t.lower()), None)
        search_table = next((t for t in tables if "search" in t.lower()), None)
        company_table = next((t for t in tables if "company" in t.lower()), None)
        
        # User Activity Dashboard
        if dashboard_type == "User Activity Dashboard":
            if user_table and event_table:
                st.subheader("👥 User Activity Overview")
                
                # Active users chart
                try:
                    user_activity = pd.read_sql(f"""
                        SELECT u.role, COUNT(DISTINCT e.user_id) as active_users
                        FROM {user_table} u
                        LEFT JOIN {event_table} e ON u.user_id = e.user_id
                        GROUP BY u.role
                    """, engine)
                    
                    fig = px.bar(
                        user_activity,
                        x='role',
                        y='active_users',
                        title='Active Users by Role',
                        color='active_users'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating user activity chart: {e}")
                
                # Event type distribution
                if event_table:
                    try:
                        event_dist = pd.read_sql(f"""
                            SELECT event_type, COUNT(*) as count
                            FROM {event_table}
                            GROUP BY event_type
                            ORDER BY count DESC
                        """, engine)
                        
                        fig = px.pie(
                            event_dist,
                            values='count',
                            names='event_type',
                            title='Event Type Distribution',
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating event distribution chart: {e}")
            else:
                st.info("This dashboard requires user and event tables")
        
        # Content Engagement Dashboard
        elif dashboard_type == "Content Engagement Dashboard":
            if content_table and event_table:
                st.subheader("📚 Content Engagement")
                
                # Top content by engagement
                try:
                    content_engagement = pd.read_sql(f"""
                        SELECT c.title, COUNT(e.event_id) as engagement
                        FROM {content_table} c
                        LEFT JOIN {event_table} e ON c.content_id = e.content_id
                        GROUP BY c.title
                        ORDER BY engagement DESC
                        LIMIT 10
                    """, engine)
                    
                    fig = px.bar(
                        content_engagement,
                        y='title',
                        x='engagement',
                        orientation='h',
                        title='Top 10 Content by Engagement',
                        color='engagement'
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating content engagement chart: {e}")
                
                # Content by source if available
                try:
                    if 'source' in pd.read_sql(f"PRAGMA table_info({content_table})", engine)['name'].values:
                        content_sources = pd.read_sql(f"""
                            SELECT source, COUNT(*) as count
                            FROM {content_table}
                            GROUP BY source
                            ORDER BY count DESC
                        """, engine)
                        
                        fig = px.pie(
                            content_sources,
                            values='count',
                            names='source',
                            title='Content by Source'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating content sources chart: {e}")
            else:
                st.info("This dashboard requires content and event tables")
        
        # Search Analytics Dashboard
        elif dashboard_type == "Search Analytics Dashboard":
            if search_table:
                st.subheader("🔍 Search Analytics")
                
                # Top search queries
                try:
                    top_searches = pd.read_sql(f"""
                        SELECT query, COUNT(*) as count
                        FROM {search_table}
                        GROUP BY query
                        ORDER BY count DESC
                        LIMIT 10
                    """, engine)
                    
                    fig = px.bar(
                        top_searches,
                        y='query',
                        x='count',
                        orientation='h',
                        title='Top 10 Search Queries',
                        color='count'
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating top searches chart: {e}")
                
                # Search trends over time
                if 'timestamp' in pd.read_sql(f"PRAGMA table_info({search_table})", engine)['name'].values:
                    try:
                        search_trends = pd.read_sql(f"""
                            SELECT strftime('%Y-%m-%d', timestamp) as date, COUNT(*) as searches
                            FROM {search_table}
                            GROUP BY date
                            ORDER BY date
                        """, engine)
                        
                        fig = px.line(
                            search_trends,
                            x='date',
                            y='searches',
                            title='Daily Search Volume',
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating search trends chart: {e}")
            else:
                st.info("This dashboard requires a search table")
        
        # Account Overview Dashboard
        elif dashboard_type == "Account Overview Dashboard":
            if account_table and user_table:
                st.subheader("👤 Account Overview")
                
                # Subscription plan distribution
                if 'subscription_plan' in pd.read_sql(f"PRAGMA table_info({account_table})", engine)['name'].values:
                    try:
                        plan_dist = pd.read_sql(f"""
                            SELECT subscription_plan, COUNT(*) as count
                            FROM {account_table}
                            GROUP BY subscription_plan
                        """, engine)
                        
                        fig = px.pie(
                            plan_dist,
                            values='count',
                            names='subscription_plan',
                            title='Subscription Plan Distribution',
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating subscription plan chart: {e}")
                
                # Users per account
                try:
                    users_per_account = pd.read_sql(f"""
                        SELECT a.account_id, COUNT(u.user_id) as users
                        FROM {account_table} a
                        LEFT JOIN {user_table} u ON a.account_id = u.account_id
                        GROUP BY a.account_id
                        ORDER BY users DESC
                    """, engine)
                    
                    fig = px.bar(
                        users_per_account,
                        x='account_id',
                        y='users',
                        title='Users per Account',
                        color='users'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating users per account chart: {e}")
            else:
                st.info("This dashboard requires account and user tables")
                
        # Default dashboard if type not recognized
        else:
            st.subheader("📊 Database Overview")
            
            # Display table information
            table_info = []
            for table in tables:
                try:
                    row_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", engine).iloc[0, 0]
                    columns = pd.read_sql(f"PRAGMA table_info({table})", engine)['name'].tolist()
                    
                    table_info.append({
                        "Table": table,
                        "Rows": row_count,
                        "Columns": len(columns),
                        "Column Names": ", ".join(columns[:5]) + ("..." if len(columns) > 5 else "")
                    })
                except Exception:
                    continue
            
            st.dataframe(pd.DataFrame(table_info))
            
            # Get table relationships if possible
            try:
                st.subheader("Table Relationships")
                relations = []
                
                # Check for common foreign key patterns
                for table in tables:
                    columns = pd.read_sql(f"PRAGMA table_info({table})", engine)['name'].tolist()
                    for col in columns:
                        if col.endswith('_id') and col != f"{table.rstrip('s')}_id":
                            related_table = col.replace('_id', '')
                            if related_table + 's' in tables or related_table in tables:
                                relations.append({
                                    "From Table": table,
                                    "To Table": related_table + ('s' if related_table + 's' in tables else ''),
                                    "Foreign Key": col
                                })
                
                if relations:
                    st.dataframe(pd.DataFrame(relations))
                else:
                    st.info("No clear table relationships detected")
            except Exception as e:
                st.error(f"Error detecting relationships: {e}")

    except Exception as e:
        st.error(f"❌ Error building dashboard: {e}")
        st.warning("There was a problem generating the requested dashboard")

def get_comprehensive_db_info(engine):
    """Get comprehensive database information for agent context"""
    info = {}
    
    try:
        # Get schema information
        schema_info = {}
        inspector = inspect(engine)
        
        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column.get("nullable", True)
                })
            
            schema_info[table_name] = {
                "columns": columns,
                "row_count": get_table_row_count(engine, table_name)
            }
        
        info["schema"] = schema_info
        
        # Get table relationships
        relationships = []
        for table_name, table_info in schema_info.items():
            for column in table_info["columns"]:
                col_name = column["name"]
                # Check for common foreign key patterns
                if col_name.endswith('_id') and col_name != f"{table_name.rstrip('s')}_id":
                    related_table = col_name.replace('_id', '')
                    if related_table + 's' in schema_info or related_table in schema_info:
                        relationships.append({
                            "from_table": table_name,
                            "from_column": col_name,
                            "to_table": related_table + ('s' if related_table + 's' in schema_info else ''),
                            "to_column": "id" if col_name != "id" else related_table + "_id"
                        })
        
        info["relationships"] = relationships
        
        # Get sample data for each table
        sample_data = {}
        for table_name in schema_info.keys():
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 3"))
                    sample_data[table_name] = [dict(row) for row in result]
            except:
                sample_data[table_name] = []
        
        info["sample_data"] = sample_data
        
        return info
    except Exception as e:
        return {"error": str(e)}