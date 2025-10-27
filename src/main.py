import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
#from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
#from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import (
    get_csv_file_details,
    getting_dates,
    download_file_if_missing,
    analyze_csv,
    generate_case_time_series_charts,
    search_online_news,
    calculate_epidemiology_rates,
    generate_pdf_report
)

# Configure LLM
# Using a lower temperature for more stable tool usage
os.getenv("HAGGINGFACE_API_KEY")

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

system_prompt = (
    "You are a data-focused assistant."
    "If a question requires information from the CSV, first use an appropriate tool."
    "Use only one tool call per step if possible."
    "Answer concisely and in a structured way."
    "If no tool fits, briefly explain why.\n\n"
    "Available tools:\n{tools}\n"
    "Use only these tools: {tool_names}."
)


messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_query),
]

tools = [
    get_csv_file_details,
    getting_dates,
    download_file_if_missing,
    analyze_csv,
    generate_case_time_series_charts,
    search_online_news,
    calculate_epidemiology_rates,
    generate_pdf_report
]

# _tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools)
# _tool_names = ", ".join(t.name for t in tools)
# prompt = prompt.partial(tools=_tool_desc, tool_names=_tool_names)

# Create and run tool-calling agent
chat_model = ChatHuggingFace(llm=llm)

if __name__ == "__main__":
    user_query = "Qual é o link onde podemos encontrar os dados?"
    result = chat_model.invoke(messages)
    print("\nAGENT ANSWER")
    print(result["output"])

def ask_agent(query: str) -> str:
    return chat_model.invoke({"input": query})["output"]



###########################################################
# # Open DataSUS base URL
# url = "https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024/resource/20c49de3-ddc3-4b76-a942-1518eaae9c91"

# # URL of data in CSV
# url_default = get_csv_file_details(url)
# ###########################################################

# ###########################################################
# focus_column = [
#     "NU_NOTIFIC", # número da notificação
#     "DT_NOTIFIC", # data da notificação
#     "UTI", # internado em UTI
#     "DT_ENTUTI", # data da internação na UTI
#     "DT_SAIDUTI", # data da saída da UTI
#     "CLASSI_FIN", # classificação final do caso
#     "EVOLUCAO", # evolução do caso
#     "DT_EVOLUCA", # data da alta ou óbito
#     "VACINA" # recebeu vacina contra gripe
#     ]

# #analyze_csv(folder_path="data/", focus_column=focus_column)

# ###########################################################

# News Search Tool
#import os
#from typing import Dict
# from langchain_community.utilities import GoogleSerperAPIWrapper 

# #####################################################################
# from dotenv import load_dotenv

# load_dotenv()

# server_api_key = os.getenv("SERPER_API_KEY")

# # Initialize the Serper API wrapper as the search tool
# serper_search = GoogleSerperAPIWrapper(type="news")

# start_date, end_date = getting_dates(url_default)


# # search_online_news(
# #     query = "Síndrome Respiratória Aguda Grave",
# #     num_results = 5,
# #     start_date = start_date,
# #     end_date = end_date)
# #####################################################################

