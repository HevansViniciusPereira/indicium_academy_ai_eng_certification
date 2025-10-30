# indicium_academy_ai_eng_certification
This repository contains the code related to the Indicium Academy Certification to the AI Engineer.

# 🦠 Epidemiological Report Generator Agent

## 🌟 Overview

This project implements a sophisticated **Generative AI Agentic Workflow** designed to automate the end-to-end process of generating comprehensive epidemiological situation reports. It seamlessly integrates data sourcing, statistical analysis, graphic generation, and natural language generation (NLG) from large language models (LLMs).

The entire process is orchestrated as a directed acyclic graph (DAG) or a state machine (likely built with **LangGraph** or similar framework), where a central **ReportState** is passed between sequential and parallel processing nodes.

## 💡 Agentic Architecture and Data Flow

The core of this system is the **ReportState** and a series of independent, chained **nodes** (functions) that act as specialized agents or tools. Each node performs a specific task, updates the central state, and hands the control to the next node in the pipeline.



---

## 🛠️ Components

### 1. The Central State: `ReportState`

The `ReportState` is the single source of truth, managing all inputs, intermediate results, and final report content.

| Key | Type | Description |
| :--- | :--- | :--- |
| `root_url` | `str` | Initial URL to locate the data source. |
| `download_url` | `str` | Direct link to the raw data file (e.g., CSV). |
| `start_date` | `str` | Start date for data filtering and reporting period. |
| `end_date` | `str` | End date for data filtering and reporting period. |
| `local_path` | `str` | Local path where the downloaded dataset is saved. |
| `query` | `str` | Primary subject/search term for news retrieval (e.g., "Dengue Fever"). |
| `selected_columns` | `list` | Columns necessary for epidemiological rate calculation. |
| `metrics` | `Dict` | Quantitative indicators and calculated rates. |
| `news` | `str` | Raw/processed content from online news searches. |
| `summary` | `str` | AI-generated executive summary. |
| `recent_developments` | `str` | AI-generated synthesis of recent geopolitical/epidemiological events. |
| `perspectives` | `str` | AI-generated outlook or future risk assessment. |
| `desc_metrics` | `str` | Human-readable analysis of the calculated metrics (AI-generated). |
| `desc_12_months` | `str` | Textual description of case trends over the last 12 months (AI-generated from graphics). |
| `desc_30_days` | `str` | Textual description of case trends over the last 30 days (AI-generated from graphics). |

### 2. The Agentic Nodes

The nodes are the functional steps in the pipeline. They utilize specialized **Tools** (like `get_csv_file_details`, `calculate_epidemiology_rates`, `analyze_graphic`, `create_content`) to perform their tasks.

| Node | Function | Tools Used | Description |
| :--- | :--- | :--- | :--- |
| **`node_recent_csv_url`** | Data Sourcing | `get_csv_file_details` | Finds the most current CSV download link from a root page. |
| **`node_getting_dates`** | Pre-processing | `getting_dates` | Extracts the report's date range from the download URL (metadata). |
| **`node_download_csv`** | Data Ingestion | `download_file_if_missing` | Downloads the raw data file locally for processing. |
| **`node_metrics`** | Data Analysis | `calculate_epidemiology_rates` | Calculates key rates (e.g., incidence, mortality) from the dataset. |
| **`node_generate_graphics`** | Visualization | `generate_case_time_series_charts` | Creates and saves time-series plots (e.g., last 12 months, last 30 days). |
| **`node_search_news`** | Context Retrieval | `search_online_news` | Retrieves relevant and recent news articles based on the report's **query** and date range. |
| **`node_analyze_metrics`** | NLG Agent | `analyze_metrics` | Uses an LLM to interpret and generate a descriptive analysis of the calculated numerical metrics. |
| **`node_analyze_graphics`** | Vision Agent | `analyze_graphic` | Uses a Multimodal LLM to analyze the generated time-series charts (PNG/JPEG) and produce textual summaries of the trends. |
| **`node_create_content`** | Synthesis Agent | `create_content` | Uses an LLM to synthesize the **news** articles into structured report sections: **Summary**, **Recent Developments**, and **Perspectives**. |
| **`node_generate_report`** | Output Generation | `generate_pdf_report` | Compiles all generated content (metrics analysis, graphic analysis, news synthesis) into the final PDF document. |

---

## 🚀 Getting Started

### Prerequisites

* Python 3.12+
* LangChain/LangGraph (or similar orchestration framework)
* Access to an LLM/Multimodal model (e.g., Gemini, GPT-4V) for the analysis and content generation steps.
* Required data science libraries (e.g., pandas, matplotlib/seaborn) for data processing and graphics.

### Installation

```bash
git clone [https://github.com/your-repo/epidemiological-report-agent.git](https://github.com/your-repo/epidemiological-report-agent.git)
cd epidemiological-report-agent
pip install -r requirements.txt