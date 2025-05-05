# app/dashboard_builder.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import datetime, timedelta
import uuid
import os
import json
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

class DashboardBuilder:
    """Advanced dashboard builder for Agentic BI Platform"""
    
    def __init__(self, engine, query=None, goal=None):
        """Initialize with database engine and optional query/goal"""
        self.engine = engine
        self.query = query
        self.goal = goal
        self.llm = ChatAnthropic(
            model_name="claude-3-haiku-20240307",
            temperature=0.3,
            anthropic_api_key=ANTHROPIC_KEY
        )
        
    def analyze_query_intent(self, query=None):
        """Extract intent from user query to determine dashboard type"""
        if query is None:
            query = self.query
            
        # Check for explicit dashboard requests
        dashboard_types = {
            r'\b(trend|time series|over time|historical)\b': 'trend',
            r'\b(compare|comparison|versus|vs)\b': 'comparison',
            r'\b(distribution|spread|histogram)\b': 'distribution',
            r'\b(relationship|correlation|scatter)\b': 'relationship',
            r'\b(breakdown|composition|proportion)\b': 'breakdown',
            r'\b(geo|map|regional|location)\b': 'geospatial',
            r'\b(kpi|metric|performance indicator)\b': 'kpi',
            r'\b(funnel|conversion|pipeline)\b': 'funnel',
            r'\b(cohort|retention)\b': 'cohort',
            r'\babnormal|anomaly|outlier\b': 'anomaly',
            r'\b(forecast|predict|projection)\b': 'forecast'
        }
        
        intent = 'exploratory'  # Default intent
        for pattern, intent_type in dashboard_types.items():
            if re.search(pattern, query, re.IGNORECASE):
                intent = intent_type
                break
                
        # Use LLM for more nuanced intent detection if needed
        if intent == 'exploratory' and len(query.split()) > 5:
            prompt = f"""
            Based on this user query: "{query}"
            What type of dashboard would be most helpful?
            Choose one: trend, comparison, distribution, relationship, breakdown, geospatial, kpi, funnel, cohort, anomaly, forecast, exploratory
            Just respond with a single word.
            """
            try:
                intent = self.llm.invoke(prompt).strip().lower()
                # Fallback if LLM returns something unexpected
                if intent not in dashboard_types.values() and intent != 'exploratory':
                    intent = 'exploratory'
            except Exception:
                # If LLM fails, stick with exploratory
                pass
                
        return intent
        
    def detect_time_dimension(self, df):
        """Find the most likely time dimension in dataframe"""
        date_patterns = [
            r'date', r'time', r'day', r'month', r'year', r'week', r'quarter',
            r'created', r'updated', r'timestamp', r'datetime'
        ]
        
        # Check column names
        for col in df.columns:
            for pattern in date_patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    # Verify it's a date type or can be converted
                    try:
                        # Try to convert a sample to datetime
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            return col
                        
                        # Try first non-null value
                        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                        if sample and isinstance(sample, str):
                            pd.to_datetime(sample)
                            return col
                    except:
                        continue
        
        return None
        
    def detect_numeric_dimensions(self, df, exclude_cols=None):
        """Find numeric columns suitable for measures"""
        if exclude_cols is None:
            exclude_cols = []
            
        numeric_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                # Skip likely ID columns
                if col.lower().endswith('id') or col.lower().startswith('id'):
                    if df[col].nunique() > 0.5 * len(df):  # High cardinality
                        continue
                numeric_cols.append(col)
        
        return numeric_cols
        
    def detect_categorical_dimensions(self, df, exclude_cols=None):
        """Find categorical columns suitable for dimensions"""
        if exclude_cols is None:
            exclude_cols = []
            
        categorical_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
                
            # Include explicit categorical columns
            if pd.api.types.is_categorical_dtype(df[col]):
                categorical_cols.append(col)
                continue
                
            # Include string columns with reasonable cardinality
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                if df[col].nunique() <= max(20, 0.2 * len(df)):  # Not too many unique values
                    categorical_cols.append(col)
                continue
                
            # Include low-cardinality numeric columns that are likely categorical
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 20:  # Small number of unique values
                    categorical_cols.append(col)
                    
        return categorical_cols
        
    def generate_dashboard_title(self, df, query):
        """Generate a descriptive dashboard title"""
        prompt = f"""
        Create a concise, professional dashboard title (5-7 words) based on this query: "{query}"
        Data columns include: {', '.join(df.columns.tolist())}
        Just provide the title with no explanations or quotation marks.
        """
        try:
            title = self.llm.invoke(prompt).strip()
            # Remove extra quotation marks if present
            title = title.replace('"', '').replace("'", '')
            return title
        except Exception:
            # Fallback title
            return f"Dashboard: {query[:30]}..." if len(query) > 30 else f"Dashboard: {query}"
            
    def generate_insight_text(self, df, query, chart_type=None):
        """Generate insight text for the dashboard"""
        # Data description
        rows = len(df)
        cols = len(df.columns)
        col_summary = ", ".join(df.columns.tolist()[:5])
        if len(df.columns) > 5:
            col_summary += f" and {len(df.columns) - 5} more"
            
        data_preview = df.head(3).to_string()
        
        prompt = f"""
        Provide 3 short, data-driven insights based on this query: "{query}"
        
        Data summary:
        - {rows} rows, {cols} columns
        - Columns: {col_summary}
        - Sample data:
        {data_preview}
        
        Focus on business implications, trends, and actionable takeaways.
        Format as 3 numbered bullet points.
        Keep total response under 120 words.
        """
        
        try:
            insights = self.llm.invoke(prompt).strip()
            return insights
        except Exception:
            # Fallback insight
            return "Data analysis complete. Explore the visualizations for insights."
            
    def build_trend_dashboard(self, df):
        """Build a dashboard focusing on trends over time"""
        time_col = self.detect_time_dimension(df)
        if time_col is None:
            st.error("No time dimension detected for trend analysis")
            return
            
        # Convert time column to datetime if needed
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            st.error(f"Failed to convert {time_col} to datetime: {e}")
            return
            
        # Get numeric measures
        measures = self.detect_numeric_dimensions(df, exclude_cols=[time_col])
        if not measures:
            st.error("No numeric measures found for trend analysis")
            return
            
        # Generate title
        title = self.generate_dashboard_title(df, self.query)
        st.title(f"📈 {title}")
        
        # Create main trend chart with up to 3 measures
        display_measures = measures[:3]  # Limit to avoid cluttered chart
        
        # Time aggregation based on date range
        date_range = (df[time_col].max() - df[time_col].min()).days
        if date_range > 365*2:  # More than 2 years
            freq = 'YS'  # Yearly
            period_label = 'Year'
        elif date_range > 90:  # More than 3 months
            freq = 'MS'  # Monthly
            period_label = 'Month'
        elif date_range > 21:  # More than 3 weeks
            freq = 'W'  # Weekly
            period_label = 'Week'
        else:
            freq = 'D'  # Daily
            period_label = 'Day'
            
        # Create aggregated dataframe
        trend_df = df.set_index(time_col).resample(freq).agg({m: 'sum' for m in display_measures})
        trend_df.reset_index(inplace=True)
        
        # Main trend chart
        fig = px.line(
            trend_df, 
            x=time_col, 
            y=display_measures,
            title=f"Trends Over Time ({period_label})",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Column charts for metric totals
        col1, col2 = st.columns(2)
        
        # Recent period vs previous period comparison
        if freq == 'YS':
            latest_period = df[time_col].dt.year.max()
            prev_period = latest_period - 1
            latest_label = str(latest_period)
            prev_label = str(prev_period)
            filter_latest = df[time_col].dt.year == latest_period
            filter_prev = df[time_col].dt.year == prev_period
        elif freq == 'MS':
            latest_period = df[time_col].dt.to_period('M').max()
            prev_period = latest_period - 1
            latest_label = latest_period.strftime('%b %Y')
            prev_label = prev_period.strftime('%b %Y')
            filter_latest = df[time_col].dt.to_period('M') == latest_period
            filter_prev = df[time_col].dt.to_period('M') == prev_period
        elif freq == 'W':
            latest_period = df[time_col].dt.isocalendar().week.max()
            prev_period = latest_period - 1
            latest_label = f"Week {latest_period}"
            prev_label = f"Week {prev_period}"
            filter_latest = df[time_col].dt.isocalendar().week == latest_period
            filter_prev = df[time_col].dt.isocalendar().week == prev_period
        else:  # Daily
            latest_period = df[time_col].max()
            prev_period = latest_period - timedelta(days=1)
            latest_label = latest_period.strftime('%b %d')
            prev_label = prev_period.strftime('%b %d')
            filter_latest = df[time_col] == latest_period
            filter_prev = df[time_col] == prev_period
            
        with col1:
            st.subheader("Period Comparison")
            comparison_data = []
            
            for measure in display_measures:
                latest_val = df[filter_latest][measure].sum()
                prev_val = df[filter_prev][measure].sum() if not df[filter_prev].empty else 0
                
                comparison_data.append({
                    'Metric': measure,
                    latest_label: latest_val,
                    prev_label: prev_val,
                    'Change': (latest_val / prev_val - 1) * 100 if prev_val else 0
                })
                
            comp_df = pd.DataFrame(comparison_data)
            fig = px.bar(
                comp_df,
                x='Metric',
                y=[latest_label, prev_label],
                barmode='group',
                title="Current vs Previous Period"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Add categorical breakdown if available
        categorical_cols = self.detect_categorical_dimensions(df, exclude_cols=[time_col])
        
        with col2:
            if categorical_cols:
                selected_dim = categorical_cols[0]  # Use first categorical dimension
                st.subheader(f"Breakdown by {selected_dim}")
                
                # Get top categories by measure
                primary_measure = display_measures[0]
                top_cats = df.groupby(selected_dim)[primary_measure].sum().nlargest(5).index.tolist()
                breakdown_df = df[df[selected_dim].isin(top_cats)].groupby([time_col.name, selected_dim])[primary_measure].sum().reset_index()
                
                fig = px.line(
                    breakdown_df, 
                    x=time_col,
                    y=primary_measure,
                    color=selected_dim,
                    title=f"Top 5 {selected_dim} Performance"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Show YoY or MoM growth rates if no categorical dimensions
                st.subheader("Growth Rate")
                
                growth_df = trend_df.copy()
                for measure in display_measures:
                    growth_df[f'{measure}_growth'] = growth_df[measure].pct_change() * 100
                    
                growth_cols = [f'{m}_growth' for m in display_measures]
                fig = px.bar(
                    growth_df.iloc[1:],  # Skip first row (no growth rate)
                    x=time_col,
                    y=growth_cols,
                    title="Period-over-Period Growth Rate (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        # Bottom section - insights and detailed table
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Key Insights")
            insights = self.generate_insight_text(df, self.query, chart_type="trend")
            st.markdown(insights)
            
        with col2:
            st.subheader("Detailed Data")
            st.dataframe(df.head(10), use_container_width=True)
            
    def build_comparison_dashboard(self, df):
        """Build a dashboard focused on comparing categories"""
        # Detect categorical dimensions
        cat_dims = self.detect_categorical_dimensions(df)
        if not cat_dims:
            st.error("No categorical dimensions found for comparison")
            return
            
        # Get numeric measures
        measures = self.detect_numeric_dimensions(df, exclude_cols=cat_dims)
        if not measures:
            st.error("No numeric measures found for comparison")
            return
            
        # Generate title
        title = self.generate_dashboard_title(df, self.query)
        st.title(f"🔄 {title}")
        
        # Select primary dimension and measures
        primary_dim = cat_dims[0]
        primary_measure = measures[0]
        secondary_measures = measures[1:3] if len(measures) > 1 else []
        
        # Time dimension for trends
        time_col = self.detect_time_dimension(df)
        
        # Main comparison chart
        fig = px.bar(
            df,
            x=primary_dim,
            y=primary_measure,
            title=f"{primary_measure} by {primary_dim}",
            color=cat_dims[1] if len(cat_dims) > 1 else None,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Secondary charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performers
            st.subheader(f"Top {primary_dim} by {primary_measure}")
            top_df = df.groupby(primary_dim)[primary_measure].sum().reset_index().nlargest(5, primary_measure)
            fig = px.bar(
                top_df,
                x=primary_dim,
                y=primary_measure,
                title=f"Top 5 {primary_dim}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Multi-measure comparison
            if secondary_measures:
                st.subheader("Multi-Metric Comparison")
                
                # Normalize for comparison
                comp_df = df.groupby(primary_dim)[measures].sum().reset_index()
                for measure in measures:
                    max_val = comp_df[measure].max()
                    if max_val > 0:
                        comp_df[f"{measure}_norm"] = comp_df[measure] / max_val
                        
                norm_measures = [f"{m}_norm" for m in measures]
                
                fig = px.bar(
                    comp_df,
                    x=primary_dim,
                    y=norm_measures,
                    title="Normalized Comparison",
                    labels={m: m.replace('_norm', '') for m in norm_measures}
                )
                st.plotly_chart(fig, use_container_width=True)
            elif time_col:
                # Trend over time if no secondary measures
                st.subheader("Trend Analysis")
                top_cats = top_df[primary_dim].tolist()
                trend_df = df[df[primary_dim].isin(top_cats)].copy()
                
                if not pd.api.types.is_datetime64_any_dtype(trend_df[time_col]):
                    try:
                        trend_df[time_col] = pd.to_datetime(trend_df[time_col])
                    except:
                        pass
                        
                fig = px.line(
                    trend_df,
                    x=time_col,
                    y=primary_measure,
                    color=primary_dim,
                    title=f"{primary_measure} Trend by Top {primary_dim}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Pie chart alternative
                st.subheader("Proportion Analysis")
                fig = px.pie(
                    df,
                    values=primary_measure,
                    names=primary_dim,
                    title=f"{primary_measure} Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        # Bottom section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Key Insights")
            insights = self.generate_insight_text(df, self.query, chart_type="comparison")
            st.markdown(insights)
            
        with col2:
            st.subheader("Detailed Data")
            st.dataframe(df.head(10), use_container_width=True)
            
    def build_distribution_dashboard(self, df):
        """Build a dashboard focused on distributions and statistics"""
        # Get numeric measures
        measures = self.detect_numeric_dimensions(df)
        if not measures:
            st.error("No numeric measures found for distribution analysis")
            return
            
        # Generate title
        title = self.generate_dashboard_title(df, self.query)
        st.title(f"📊 {title}")
        
        # Select primary measure
        primary_measure = measures[0]
        
        # Main histogram
        fig = px.histogram(
            df,
            x=primary_measure,
            title=f"Distribution of {primary_measure}",
            height=400
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Secondary charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Box plot
            st.subheader("Box Plot")
            fig = px.box(
                df,
                y=primary_measure,
                title=f"{primary_measure} Box Plot"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Statistics table
            st.subheader("Statistics")
            stats = df[primary_measure].describe().reset_index()
            stats.columns = ['Statistic', 'Value']
            
            # Add more statistics
            additional_stats = pd.DataFrame({
                'Statistic': ['Median', 'Mode', 'Range', 'IQR', 'Skewness', 'Kurtosis'],
                'Value': [
                    df[primary_measure].median(),
                    df[primary_measure].mode()[0] if not df[primary_measure].mode().empty else None,
                    df[primary_measure].max() - df[primary_measure].min(),
                    df[primary_measure].quantile(0.75) - df[primary_measure].quantile(0.25),
                    df[primary_measure].skew(),
                    df[primary_measure].kurtosis()
                ]
            })
            
            stats = pd.concat([stats, additional_stats])
            st.dataframe(stats, use_container_width=True)
            
        with col3:
            # Distribution across categories
            cat_dims = self.detect_categorical_dimensions(df)
            if cat_dims:
                st.subheader(f"By {cat_dims[0]}")
                fig = px.violin(
                    df,
                    x=cat_dims[0],
                    y=primary_measure,
                    box=True,
                    title=f"Distribution by {cat_dims[0]}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Cumulative distribution
                st.subheader("Cumulative Distribution")
                fig = px.ecdf(
                    df,
                    x=primary_measure,
                    title="Cumulative Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        # Second row charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Outlier analysis
            st.subheader("Outlier Analysis")
            
            # Calculate IQR bounds
            q1 = df[primary_measure].quantile(0.25)
            q3 = df[primary_measure].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[primary_measure] < lower_bound) | (df[primary_measure] > upper_bound)]
            normal = df[(df[primary_measure] >= lower_bound) & (df[primary_measure] <= upper_bound)]
            
            fig = px.scatter(
                df,
                x=df.index,
                y=primary_measure,
                color_discrete_sequence=['blue'],
                title="Values and Outliers"
            )
            
            fig.add_trace(go.Scatter(
                x=outliers.index,
                y=outliers[primary_measure],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Outliers'
            ))
            
            fig.add_shape(
                type='line',
                x0=0, x1=len(df),
                y0=upper_bound, y1=upper_bound,
                line=dict(color='red', dash='dash')
            )
            
            fig.add_shape(
                type='line',
                x0=0, x1=len(df),
                y0=lower_bound, y1=lower_bound,
                line=dict(color='red', dash='dash')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Compare measures if multiple available
            if len(measures) > 1:
                st.subheader("Compare Distributions")
                dist_df = pd.DataFrame()
                
                for measure in measures[:3]:  # Limit to 3 measures
                    # Normalize for comparison
                    measure_values = df[measure].dropna()
                    if len(measure_values) > 0:
                        min_val = measure_values.min()
                        max_val = measure_values.max()
                        
                        if max_val > min_val:
                            normalized = (measure_values - min_val) / (max_val - min_val)
                            dist_df[measure] = normalized
                
                if not dist_df.empty:
                    fig = go.Figure()
                    for measure in dist_df.columns:
                        fig.add_trace(go.Violin(
                            y=dist_df[measure],
                            name=measure,
                            box_visible=True,
                            meanline_visible=True
                        ))
                    
                    fig.update_layout(title="Normalized Distributions")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Time-based distribution if time column available
                time_col = self.detect_time_dimension(df)
                if time_col:
                    st.subheader("Time-based Analysis")
                    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                        try:
                            df[time_col] = pd.to_datetime(df[time_col])
                        except:
                            pass
                            
                    # Extract time components
                    try:
                        time_df = df.copy()
                        time_df['hour'] = time_df[time_col].dt.hour
                        time_df['day'] = time_df[time_col].dt.day_name()
                        time_df['month'] = time_df[time_col].dt.month_name()
                        
                        # Create heatmap of measure by day and hour
                        heatmap_df = time_df.groupby(['day', 'hour'])[primary_measure].mean().reset_index()
                        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        
                        # Filter to days present in the data
                        days_present = [day for day in days_order if day in heatmap_df['day'].unique()]
                        
                        fig = px.density_heatmap(
                            heatmap_df,
                            x='hour',
                            y='day',
                            z=primary_measure,
                            category_orders={'day': days_present},
                            title=f"{primary_measure} by Day and Hour"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        # Fallback to simple time series
                        fig = px.line(
                            df,
                            x=time_col,
                            y=primary_measure,
                            title=f"{primary_measure} Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # Additional histogram with density curve
                    st.subheader("Density Plot")
                    fig = ff.create_distplot(
                        [df[primary_measure].dropna()],
                        [primary_measure],
                        show_hist=True,
                        show_rug=False
                    )
                    fig.update_layout(title=f"{primary_measure} Density")
                    st.plotly_chart(fig, use_container_width=True)
                    
        # Bottom section
        st.subheader("Key Insights")
        insights = self.generate_insight_text(df, self.query, chart_type="distribution")
        st.markdown(insights)
        
    def build_relationship_dashboard(self, df):
        """Build a dashboard focused on relationships between variables"""
        # Get numeric measures
        measures = self.detect_numeric_dimensions(df)
        if len(measures) < 2:
            st.error("Need at least 2 numeric measures for relationship analysis")
            return
        
        # Get categorical dimensions
        cat_dims = self.detect_categorical_dimensions(df)
        
        # Generate title
        title = self.generate_dashboard_title(df, self.query)
        st.title(f"🔗 {title}")
        
        # Main scatter plot
        x_var = measures[0]
        y_var = measures[1]
        color_var = cat_dims[0] if cat_dims else None
        fig = px.scatter(
            df, 
            x=x_var, 
            y=y_var, 
            color=color_var,
            trendline="ols" if color_var is None else None,
            title=f"Relationship between {x_var} and {y_var}",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        correlation = df[[x_var, y_var]].corr().iloc[0, 1]
        
        # Secondary charts
        col1, col2 = st.columns(2)
        with col1:
            # Correlation matrix for multiple measures
            if len(measures) > 2:
                st.subheader("Correlation Matrix")
                corr_matrix = df[measures].corr()
                fig = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    color_continuous_scale="RdBu_r", 
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display correlation value with color coding
            st.subheader("Correlation Analysis")
            corr_color = "green" if abs(correlation) > 0.7 else "orange" if abs(correlation) > 0.4 else "red"
            st.markdown(f"<h3 style='color:{corr_color}'>Correlation: {correlation:.3f}</h3>", unsafe_allow_html=True)
            
            # Add correlation strength interpretation
            if abs(correlation) > 0.7:
                st.write("Strong correlation detected")
            elif abs(correlation) > 0.4:
                st.write("Moderate correlation detected")
            else:
                st.write("Weak or no correlation detected")
        
        # Add distribution plots
        st.subheader("Distribution Analysis")
        dist_cols = st.columns(2)
        
        with dist_cols[0]:
            st.write(f"Distribution of {x_var}")
            fig = px.histogram(df, x=x_var, title=f"Distribution of {x_var}")
            st.plotly_chart(fig, use_container_width=True)
        
        with dist_cols[1]:
            st.write(f"Distribution of {y_var}")
            fig = px.histogram(df, y=y_var, title=f"Distribution of {y_var}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Add variable selection options
        st.subheader("Explore Different Relationships")
        col1, col2 = st.columns(2)
        with col1:
            new_x = st.selectbox("Select X variable", options=measures, index=measures.index(x_var))
        with col2:
            new_y = st.selectbox("Select Y variable", options=measures, index=measures.index(y_var))
        
        # Update chart if selections changed
        if new_x != x_var or new_y != y_var:
            new_correlation = df[[new_x, new_y]].corr().iloc[0, 1]
            fig = px.scatter(
                df, 
                x=new_x, 
                y=new_y, 
                color=color_var,
                trendline="ols" if color_var is None else None,
                title=f"Relationship between {new_x} and {new_y}",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Correlation: {new_correlation:.3f}")