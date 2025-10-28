from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from tools import (
    get_csv_file_details,
    getting_dates,
    download_file_if_missing,
    generate_case_time_series_charts,
    search_online_news,
    calculate_epidemiology_rates,
    generate_pdf_report
)


class ReportState(TypedDict):
    """descricao"""

    root_url: str
    download_url: str
    start_date: str
    end_date: str
    local_path: str
    selected_columns: list

def node_recent_csv_url(state: ReportState) -> ReportState:
    state["download_url"] = get_csv_file_details(state["root_url"])
    return state

def node_getting_dates(state: ReportState) -> ReportState:
    start_date, end_date = getting_dates(state["download_url"])
    state["start_date"] = start_date
    state["end_date"] = end_date
    return state

def node_download_csv(state: ReportState) -> ReportState:
    download_url = state["download_url"]
    local_path = state["local_path"]
    download_file_if_missing(download_url, local_path)
    return state

def node_generate_graphics(state: ReportState) -> ReportState:
    local_path = state["local_path"]
    date_col = 'DT_NOTIFIC'
    generate_case_time_series_charts(local_path, date_col)
    return state

def node_search_news(state: ReportState) -> ReportState:
    start_date = state["start_date"]
    end_date = state["end_date"]

    search_online_news(
        query = "Notícias sobre Síndrome Respiratória Aguda Grave no Brasil",
        num_results = 5,
        start_date = start_date,
        end_date = end_date
    )

    return state

def node_metrics(state: ReportState) -> ReportState:
    local_path = state["local_path"]
    selected_columns = state["selected_columns"]
    end_date = state["end_date"]

    calculate_epidemiology_rates(
        local_path,
        selected_columns,
        end_date
    )

    return state


def node_generate_report(state: ReportState) -> ReportState:
    start_date = state["start_date"]
    end_date = state["end_date"]

    generate_pdf_report(
        start_date=start_date,
        end_date=end_date
        )

    return state

# Creating graph
builder = StateGraph(ReportState)
builder.add_node("recent_csv_url", node_recent_csv_url)
builder.add_node("getting_dates", node_getting_dates)
builder.add_node("download_csv", node_download_csv)
builder.add_node("generate_graphics", node_generate_graphics)
builder.add_node("search_news", node_search_news)
builder.add_node("metrics", node_metrics)
builder.add_node("generate_report", node_generate_report)

builder.add_edge(START, "recent_csv_url")
builder.add_edge("recent_csv_url", "getting_dates")
builder.add_edge("getting_dates", "download_csv")
builder.add_edge("download_csv", "generate_graphics")
builder.add_edge("generate_graphics", "search_news")
builder.add_edge("search_news", "metrics")
builder.add_edge("metrics", "generate_report")
builder.add_edge("generate_report", END)
graph = builder.compile()

#import matplotlib.pyplot as plt
#plt.figure(figsize=(12, 6))
#print(graph.get_graph().draw_mermaid_png())
#plt.savefig("output/graphics/graph.png")
#plt.close()


selected_columns = [
    "NU_NOTIFIC", # número da notificação
    "DT_NOTIFIC", # data da notificação
    "UTI", # internado em UTI
    "DT_ENTUTI", # data da internação na UTI
    "DT_SAIDUTI", # data da saída da UTI
    "CLASSI_FIN", # classificação final do caso
    "EVOLUCAO", # evolução do caso
    "DT_EVOLUCA", # data da alta ou óbito
    "VACINA" # recebeu vacina contra gripe
    ]


initial_state = ReportState(
    root_url = "https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024/resource/20c49de3-ddc3-4b76-a942-1518eaae9c91",
    local_path = "data/",
    selected_columns = selected_columns
)
graph.invoke(initial_state)