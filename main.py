import os
from langgraph.graph import START, END, StateGraph
from src.agent import ReportState
from src.agent import (
    node_recent_csv_url,
    node_getting_dates,
    node_download_csv,
    node_generate_graphics,
    node_search_news,
    node_metrics,
    node_create_content,
    node_analyze_graphics,
    node_analyze_metrics,
    node_generate_report
)
from dotenv import load_dotenv

load_dotenv()

# Loading variables from .env file
server_api_key = os.getenv("SERPER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Selecting relevant columns
selected_columns = [
    "NU_NOTIFIC", # número da notificação
    "DT_NOTIFIC", # data da notificação
    "UTI", # internado em UTI
    "EVOLUCAO", # evolução do caso
    "VACINA" # recebeu vacina contra gripe
    ]

# Instantiating the ReportState class
initial_state = ReportState(
    query="Notícias sobre Síndrome Respiratória Aguda Grave no Brasil.",
    root_url = "https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024/resource/20c49de3-ddc3-4b76-a942-1518eaae9c91",
    local_path = "data/",
    selected_columns = selected_columns
)


# Creating graph
builder = StateGraph(ReportState)
builder.add_node("recent_csv_url", node_recent_csv_url)
builder.add_node("getting_dates", node_getting_dates)
builder.add_node("download_csv", node_download_csv)
builder.add_node("generate_graphics", node_generate_graphics)
builder.add_node("search_news", node_search_news)
builder.add_node("metrics", node_metrics)
builder.add_node("create_content", node_create_content)
builder.add_node("analyze_graphics", node_analyze_graphics)
builder.add_node("analyze_metrics", node_analyze_metrics)
builder.add_node("generate_report", node_generate_report)

builder.add_edge(START, "recent_csv_url")
builder.add_edge("recent_csv_url", "getting_dates")
builder.add_edge("getting_dates", "download_csv")
builder.add_edge("download_csv", "generate_graphics")
builder.add_edge("generate_graphics", "search_news")
builder.add_edge("search_news", "metrics")
builder.add_edge("metrics", "create_content")
builder.add_edge("create_content", "analyze_graphics")
builder.add_edge("analyze_graphics", "analyze_metrics")
builder.add_edge("analyze_metrics", "generate_report")
builder.add_edge("generate_report", END)
graph = builder.compile()
graph.invoke(initial_state)
