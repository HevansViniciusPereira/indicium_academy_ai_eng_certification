from typing import TypedDict
import json
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
import os
from langchain_huggingface import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
#from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()


server_api_key = os.getenv("SERPER_API_KEY")
#huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
google_api_key = os.getenv("GOOGLE_API_KEY")


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

class Report(TypedDict):
    content: str
    graphic_12months_desc: str
    graphic_30days_desc: str


class ReportState(TypedDict):
    """descricao"""

    query: str
    root_url: str
    download_url: str
    start_date: str
    end_date: str
    local_path: str
    selected_columns: list
    news: str
    content: str

initial_state = ReportState(
    query="Notícias sobre Síndrome Respiratória Aguda Grave no Brasil.",
    root_url = "https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024/resource/20c49de3-ddc3-4b76-a942-1518eaae9c91",
    local_path = "data/",
    selected_columns = selected_columns
)

###########################################################

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
    query = state["query"]

    news = search_online_news(
        query = query,
        num_results = 5,
        start_date = start_date,
        end_date = end_date
    )

    state["news"] = news

    return state

def node_metrics(state: ReportState) -> ReportState:
    local_path = state["local_path"]
    selected_columns = state["selected_columns"]
    end_date = state["end_date"]

    print("Calculating metrics.")

    calculate_epidemiology_rates(
        local_path,
        selected_columns,
        end_date
    )

    return state

def node_create_content(state: ReportState) -> ReportState:

    news = state["news"]
    query = state["query"]
    end_date = state["end_date"]

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        prompt = f"""
            You are an expert geopolitical and epidemiological analyst. Your task is to synthesize news content.
            The primary subject is: '{query}'.
            The following is a JSON array of the most recent news articles found:
            --- NEWS DATA START ---
            {news}
            --- NEWS DATA END ---

            Generate a detailed report content in Markdown format, following these instructions:
            1. Start with a main heading '## Key Findings and Context for "{query}"'.
            2. Write a one-paragraph executive summary (under 5 sentences).
            3. Create a section '### Recent Developments' with a bulleted list of 3-5 key facts/events, citing the source title for each.
            4. Conclude with a section '### Outlook' containing a one-paragraph synthesis of the overall trend or risk.
        """

        result = llm.invoke(prompt)

    except Exception as e:
        print(f"ERROR: Failed to initialize LLM. {e}")
        return None

    state["content"] = result.content

    return state

def node_generate_report(state: ReportState) -> ReportState:
    start_date = state["start_date"]
    end_date = state["end_date"]
    #content = state["content"]

    print("Generating report.")

    generate_pdf_report(
        start_date=start_date,
        end_date=end_date,
        content="textooooooooooooooooooooooooooooooooooooooooooooooooooo"
        )
    
    print("Final")

    return state




# Creating graph
builder = StateGraph(ReportState)
builder.add_node("recent_csv_url", node_recent_csv_url)
builder.add_node("getting_dates", node_getting_dates)
builder.add_node("download_csv", node_download_csv)
builder.add_node("generate_graphics", node_generate_graphics)
builder.add_node("search_news", node_search_news)
builder.add_node("metrics", node_metrics)
builder.add_node("create_content", node_create_content)
builder.add_node("generate_report", node_generate_report)

builder.add_edge(START, "recent_csv_url")
builder.add_edge("recent_csv_url", "getting_dates")
builder.add_edge("getting_dates", "download_csv")
builder.add_edge("download_csv", "generate_graphics")
builder.add_edge("generate_graphics", "search_news")
builder.add_edge("search_news", "metrics")

#builder.add_edge("metrics", "create_content")
#builder.add_edge("create_content", "generate_report")

builder.add_edge("metrics", "generate_report")

builder.add_edge("generate_report", END)
graph = builder.compile()
graph.invoke(initial_state)


#import matplotlib.pyplot as plt
#plt.figure(figsize=(12, 6))
#print(graph.get_graph().draw_mermaid_png())
#plt.savefig("output/graphics/graph.png")
#plt.close()