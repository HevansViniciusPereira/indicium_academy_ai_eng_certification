import os, json
import pandas as pd
from langchain_core.tools import tool

# Loading CSV
DF_PATH = "../data/df.csv"
df = pd.read_csv(DF_PATH)

# Defining tools

@tool
def tool_schema() -> str:
    """Returns column names and data types as JSON."""
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return json.dumps(schema)

@tool
def tool_nulls() -> str:
    """Returns columns with the number of missing values as JSON (only columns with missing values)."""
    nulls = df.isna().sum()
    result = {col: int(n) for col, n in nulls.items() if n > 0}
    return json.dumps(result)

@tool
def tool_describe(input_str: str) -> str:
    """
    Returns describe() method.
    Optional: input_str can contain a comma-separated list of columns.
    """
    cols = None
    if input_str and input_str.strip():
        cols = [c.strip() for c in input_str.split(",") if c.strip() in df.columns]
    stats = df[cols].describe() if cols else df.describe()
    # describe() has a MultiIndex, let's adapt it for the LLM
    return stats.to_csv(index=True)







# add to main.py
tools = [tool_schema, tool_nulls, tool_describe]