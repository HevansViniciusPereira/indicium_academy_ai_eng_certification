from typing import TypedDict, Dict
from src.tools import (
    get_csv_file_details,
    getting_dates,
    download_file_if_missing,
    generate_case_time_series_charts,
    search_online_news,
    calculate_epidemiology_rates,
    create_content,
    analyze_graphic,
    analyze_metrics,
    generate_pdf_report
)

##########################################################
##################### Creating Class #####################
##########################################################

class ReportState(TypedDict):
    """
    A type definition class that serves as the central data container 
    (or state object) for the entire data processing and report generation pipeline.
    """
    
    root_url: str # The initial URL used to locate the desired data source.
    download_url: str # The direct link to download the dataset (e.g., the CSV file).
    start_date: str # The start date of the reporting period, often extracted from the download URL.
    end_date: str # The end date of the reporting period, used for filtering and context.
    local_path: str # The local file path where the downloaded dataset is saved.
    query: str # The primary subject or search term for news retrieval and report focus.
    selected_columns: list # A list of column names essential for metric calculation.
    metrics: Dict # A dictionary containing the calculated epidemiological rates and quantitative indicators.
    news: str # The raw or processed content from online news searches.
    summary: str # The synthesized executive summary text for the final report.
    recent_developments: str # The textual synthesis of recent events (geopolitical/epidemiological) derived from the news.
    perspectives: str # The textual synthesis providing an outlook or future risk assessment.
    desc_metrics: str # A human-readable description/analysis of the calculated metrics.
    desc_12_months: str # A textual description of the case trends observed over the last 12 months.
    desc_30_days: str # A textual description of the case trends observed over the last 30 days.

##########################################################
##################### Creating Nodes #####################
##########################################################

def node_recent_csv_url(state: ReportState) -> ReportState:
    """
    Retrieves the download URL for the most recent CSV data file 
    from a specified root URL and updates the ReportState.

    Args:
        state (ReportState): The state dictionary containing the
                             base URL under the 'root_url' key.

    Returns:
        state (ReportState): The updated state dictionary, now including the 
                     CSV download URL under the 'download_url' key.
    """
    state["download_url"] = get_csv_file_details(state["root_url"])
    return state

def node_getting_dates(state: ReportState) -> ReportState:
    """
    Extracts the start and end dates relevant to the data source 
    using the previously acquired download URL, and updates the ReportState.

    Args:
        state (ReportState): The state dictionary containing the 
                             CSV download URL under the 'download_url' key.

    Returns:
        state (ReportState): The updated state dictionary, now including the 
                     'start_date' and 'end_date' keys with their respective date strings.
    """
    start_date, end_date = getting_dates(state["download_url"])
    state["start_date"] = start_date
    state["end_date"] = end_date
    return state

def node_download_csv(state: ReportState) -> ReportState:
    """
    This function checks if the required CSV file is already present 
    at the designated 'local_path'. If the file is missing, it initiates 
    the download from the 'download_url'.

    Args:
        state (ReportState): The state dictionary containing the source 
                             URL ('download_url') and the intended 
                             local save path ('local_path').

    Returns:
        state (ReportState): The state dictionary, unchanged if the download 
                     is successful or skipped, but confirming the file 
                     is ready for the next processing step.
    """
    download_url = state["download_url"]
    local_path = state["local_path"]
    download_file_if_missing(download_url, local_path)
    return state

def node_generate_graphics(state: ReportState) -> ReportState:
    """
    Generates time series charts and graphics based on the downloaded dataset.

    Args:
        state (ReportState): The state dictionary containing the local file 
                             path of the CSV data under the 'local_path' key.

    Returns:
        state (ReportState): The state dictionary, returned unchanged, as the 
                     function's main purpose is the side effect of saving 
                     files (graphics) to disk.
    """
    local_path = state["local_path"]
    date_col = 'DT_NOTIFIC'
    
    generate_case_time_series_charts(local_path=local_path,
        output_dir="output/graphics",
        date_col=date_col)
    return state

def node_search_news(state: ReportState) -> ReportState:
    """
    Searches for and retrieves recent news articles relevant to the report's 
    query and time frame.

    Args:
        state (ReportState): The state dictionary containing the search 
                             parameters: 'query', 'start_date', and 'end_date'.

    Returns:
        state (ReportState): The updated state dictionary, now including the retrieved 
                     news data (e.g., a list of articles/summaries) under 
                     the 'news' key.
    """
    start_date = state["start_date"]
    end_date = state["end_date"]
    query = state["query"]
    news_output_dir = "output/news"

    news = search_online_news(
        query = query,
        num_results = 5,
        start_date = start_date,
        end_date = end_date,
        news_output_dir=news_output_dir
    )

    state["news"] = news

    return state

def node_metrics(state: ReportState) -> ReportState:
    """
    Calculates key epidemiological metrics and rates from the processed dataset.

    Args:
        state (ReportState): The state dictionary containing the necessary inputs:
                             'local_path' (path to the dataset), 
                             'selected_columns' (list of columns required for calculation), 
                             and 'end_date' (reference date for cumulative calculations).

    Returns:
        state (ReportState): The updated state dictionary, now including the calculated 
                     metrics (e.g., a dictionary or object) under the 'metrics' key.
    """
    local_path = state["local_path"]
    selected_columns = state["selected_columns"]
    end_date = state["end_date"]

    metrics = calculate_epidemiology_rates(
        local_path,
        selected_columns,
        end_date
    )

    state["metrics"] = metrics

    return state

def node_analyze_graphics(state: ReportState) -> ReportState:
    """
    Analyzes generated graphics to derive descriptive textual summaries of trends.

    Args:
        state (ReportState): The state dictionary, although the function 
                             primarily uses hardcoded paths to the generated graphics.

    Returns:
        state (ReportState): The updated state dictionary, now including the textual 
                     descriptions of the analyzed trends under the 
                     'desc_12_months' and 'desc_30_days' keys.
    """
    
    desc_12_months, desc_30_days = analyze_graphic(
        graphic_12_months_path="../output/graphics/cases_last_12_months.png",
        graphic_30_days_path="../output/graphics/cases_last_30_days.png"
    )

    state["desc_12_months"] = desc_12_months
    state["desc_30_days"] = desc_30_days
    
    return state

def node_analyze_metrics(state: ReportState) -> ReportState:
    """
    Analyzes generated metrics to derive descriptive textual summaries of trends.

    Args:
        state (ReportState): The state dictionary, although the function 
                             primarily uses hardcoded paths to the generated metrics.

    Returns:
        state (ReportState): The updated state dictionary, now including the textual 
                     descriptions of the analyzed metrics.
    """
    metrics = state["metrics"]
    desc_metrics = analyze_metrics(metrics)
    state["desc_metrics"] = desc_metrics

    return state

def node_create_content(state: ReportState) -> ReportState:
    """
    Generates the core textual content for the final report based on news articles 
    and the main query, and parses this content into distinct sections.

    Args:
        state (ReportState): The state dictionary containing the raw news data 
                             under the 'news' key and the primary 'query'.

    Returns:
        state (ReportState): The updated state dictionary, now including the parsed 
                     textual content under the 'summary', 'recent_developments', 
                     and 'perspectives' keys.
    """
    news = state["news"]
    query = state["query"]

    contents = create_content(news, query)

    # Assumes 'create_content' returns a list/tuple where elements 1, 2, and 3 
    # are strings that start with specific section headings.
    summary = contents[1].split("Resumo Executivo")[1]
    recent_developments = contents[2].split("Desenvolvimentos Recentes")[1]
    perspectives = contents[3].split("Perspectivas")[1]
    
    state["summary"] = summary
    state["recent_developments"] = recent_developments
    state["perspectives"] = perspectives

    return state

def node_generate_report(state: ReportState) -> ReportState:
    """
    Generates the final comprehensive PDF report using all synthesized and 
    analyzed components stored in the ReportState.

    Args:
        state (ReportState): The state dictionary containing all necessary 
                             data and narrative elements generated by previous nodes.

    Returns:
        state (ReportState): The state dictionary, returned unchanged, as its sole 
                     purpose is the side effect of saving the final PDF report 
                     to the file system.
    """
    start_date = state["start_date"]
    end_date = state["end_date"]
    summary = state["summary"]
    #recent_developments = state["recent_developments"]
    perspectives = state["perspectives"]
    desc_12_months = state["desc_12_months"]
    desc_30_days = state["desc_30_days"]
    desc_metrics = state["desc_metrics"]

    print("Gerando relatório.")

    generate_pdf_report(
        start_date=start_date,
        end_date=end_date,
        summary=summary,
        #recent_developments=recent_developments,
        perspectives=perspectives,
        desc_12_months=desc_12_months,
        desc_30_days=desc_30_days,
        desc_metrics=desc_metrics
        )
    
    print("Relatório gerado.")

    return state
