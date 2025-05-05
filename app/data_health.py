# data_health.py
import pandas as pd
import sqlalchemy
from datetime import datetime

def analyze_data_health(conn_url, db_name="(unknown)"):
    engine = sqlalchemy.create_engine(conn_url)
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    report = []

    for table_name in meta.tables:
        table = meta.tables[table_name]
        conn = engine.connect()

        # Read table as DataFrame
        df = pd.read_sql_table(table_name, con=conn)
        conn.close()

        row_count = len(df)
        col_info = []
        last_updated = None

        for col in df.columns:
            nulls = df[col].isnull().sum()
            dtype = str(df[col].dtype)
            col_info.append(f"{col} ({dtype}) — {nulls} nulls")

        # Guess update time
        if "updated_at" in df.columns:
            last_updated = pd.to_datetime(df["updated_at"]).max()
        elif "timestamp" in df.columns:
            last_updated = pd.to_datetime(df["timestamp"]).max()
        elif "created_at" in df.columns:
            last_updated = pd.to_datetime(df["created_at"]).max()

        report.append({
            "Database": db_name,
            "Table": table_name,
            "Columns": ", ".join(col_info),
            "Rows": row_count,
            "Last Updated": last_updated if last_updated else "Unknown"
        })

    return pd.DataFrame(report)
