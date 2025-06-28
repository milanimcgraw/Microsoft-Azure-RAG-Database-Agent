# Build a Database Agent with Azure OpenAI

## ⚙️ Project Overview

In this repository, we combine **[Microsoft Azure OpenAI's](https://learn.microsoft.com/en-us/azure/ai-services/openai/)** capabilities with **LangChain** to build an intelligent AI database agent that interacts with tabular data and SQL databases using natural language, simplifying the process for querying and extracting insights. 

- Implement **Retrieval Augmented Generation (RAG)** to query tabular data using natural language
- Translate user prompts into SQL queries using LangChain agents
- Use OpenAI’s **function calling** and **code interpreter** tools
- Work with CSV data or connect to SQL databases
- Build modular, reproducible agent workflows 

## ⚙️ Azure OpenAI
Microsoft's [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview) provides secure, enterprise-grade access to [OpenAI](https://openai.com/)'s most advanced models—such as [GPT-4](https://platform.openai.com/docs/models/gpt-4), [GPT-4o](https://openai.com/index/gpt-4o), [GPT-3.5-Turbo](https://platform.openai.com/docs/models/gpt-3-5), and embedding series—through the Azure cloud platform. Developers can use REST APIs or SDKs (Python, C#, JavaScript, Java, Go) to integrate generative capabilities like text generation, summarization, translation, image understanding, and natural language to code conversion into their applications.

#### Key Features: 

- **Model Deployment**: Create and deploy your own instances of OpenAI models in Azure with private endpoints and regional availability.
- **Function Calling & Assistants API**: Enable tool-based reasoning by registering Python functions or working with the code interpreter to analyze files and execute logic.
- **Enterprise Security**: Built-in support for [Microsoft Entra ID](https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id), virtual networks, and content filtering aligned with [Microsoft’s Responsible AI principles](https://www.microsoft.com/en-us/ai/principles-and-approach).
- **Prompt Engineering Flexibility**: Use prompt-based interaction with fine-tuning support and access to multi-modal models like [GPT-4 Turbo](https://platform.openai.com/docs/models/gpt-4-turbo) with Vision.

With [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview), developers get the power of [OpenAI](https://openai.com/) models backed by Microsoft’s cloud reliability and compliance standards.


## 🦜️🔗LangChain 
[LangChain](https://www.langchain.com/) is a flexible framework for building applications powered by large language models (LLMs). It offers a unified interface for working with chat models, vector stores, embedding models, and more—making it easy to compose intelligent, tool-augmented LLM agents.

[LangChain](https://www.langchain.com/) simplifies the full lifecycle of LLM-powered systems:

- **Development**: Build agents and chains with reusable components. Use `langchain`, `langchain-openai`, or `langchain-community` to integrate with hundreds of APIs and services.

- **Orchestration**: Combine tools, prompts, models, and memory into production-ready pipelines using [LangGraph](https://www.langchain.com/langgraph) for persistence, streaming, and agent control.

- **Evaluation**: Trace and monitor your LLM pipelines with [LangSmith](https://smith.langchain.com/), enabling continuous optimization and quality control.

LangChain’s modular design makes it ideal for constructing database agents, retrieval-augmented generation (RAG) systems, or any tool-using assistant that needs to reason, retrieve, and respond.

> See below for a detailed breakdown of project details and workflow.
---
## ⚙️ Getting Started
**To run the database agent workflows in this repository:**
### 1. Clone the Repository
```bash
git clone https://github.com/milanimcgraw/Microsoft-Azure-RAG-Database-Agent.git
cd Microsoft-Azure-RAG-Database-Agent
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
**or**
```bash
pip install pyodbc==5.1.0 tabulate==0.9.0 openai langchain==0.1.6 langchain-community==0.0.20 langchain-core==0.1.23 langchain-experimental==0.0.49 langchain-openai==0.0.5 pandas==2.2.2 jupyter notebook numpy sqlalchemy ipython matplotlib ipywidgets
```
### 3. Set Up API Keys (via .env or Terminal)
> This project supports both OpenAI and Azure OpenAI. You can choose either or both depending on the notebook you're running.

**🗂️ If using an `.env` file:**
```python
OPENAI_API_KEY=your_openai_key_here

AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_API_VERSION=2024-04-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4-1106
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```
**Then load them in your notebook using**
```python
from src.utils import get_openai_api_key, get_azure_openai_key

openai_key = get_openai_api_key()
azure_key = get_azure_openai_key()
```
**⚡ Export from Terminal (for quick setup)**

To connect your application to Azure OpenAI, you'll need to set the following environment variables:

```bash
# OpenAI Key
export OPENAI_API_KEY="your_openai_key_here"

# Azure OpenAI Keys
export AZURE_OPENAI_KEY="your_azure_openai_key"
export AZURE_OPENAI_API_VERSION="2024-04-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-4-1106"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
```
**Use in Python code:**
```python
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    openai_api_version="2024-04-01-preview",
    azure_deployment="gpt-4-1106",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
```
>⚠️ ⚠️ Be sure to replace all placeholder values with your real credentials. Never commit .env files or API keys to source control.

### 5. Launch Jupyter Notebook
Start the notebook environment to explore the agent workflows:
```bash
jupyter notebook
```
Then run a notebook! 

## 🛠️ Technical Stack
- **Knowledge Base**: `all-states-history.csv`,`test.db`
- **LLM Providers**: OpenAI (GPT-4o), Azure OpenAI (GPT-4, GPT-4-1106)
- **RAG Framework**: LangChain with Langchain-OpenAI integration
- **Agent Orchestration**: LangChain Agents, Assistants API (Azure)
- **Query Execution**: Function Calling, Code Interpreter 
- **Interface**: Jupyter Notebook
- **Package Management**: `requirements.txt`, pip

## ⚙️ Dependencies
Dependencies are listed in `requirements.txt`. Key packages:
- **openai**: Access OpenAI’s GPT models including GPT-4o
- **langchain**: Core framework for RAG, agents, and chains
- **langchain-core**: Base abstractions for chat models and chains
- **langchain-community**: Community-maintained integrations
- **langchain-experimental**: Experimental LangChain modules and workflows
- **langchain-openai**: Integration with OpenAI and Azure OpenAI APIs
- **pyodbc**: Connect to SQL databases via ODBC
- **sqlalchemy**: ORM and database abstraction
- **tabulate**: Format query results into tables
- **pandas**: Load and manipulate CSV/tabular data
- **numpy**: Numeric and array processing
- **jupyter** / **notebook**: Interactive development environment
- **ipywidgets**: Add interactivity to notebooks
- **matplotlib**: Basic plotting for results visualization


## ⚙️ Notebooks
| Filename | Description |
|----------|-------------|
| `setup_azure_openai_api.ipynb` | Connect to Azure OpenAI and run a simple prompt using LangChain |
| `langchain_csv_dataframe_agent.ipynb` | Use a Pandas agent to query a CSV via natural language |
| `csv_to_sql_agent_pipeline.ipynb` | Convert CSV data into SQL and use a SQL agent to extract insights |
| `azure-openai-function_calling_with_sql_tools.ipynb` | Define custom functions and call them using Azure OpenAI |
| `data-querying-azure-openai-assistants.ipynb` | Use the Assistants API with `code_interpreter` to answer query dataset|


## ⚙️ License
This project is released under MIT license. 

> ## 📌 Credits
> 📦  This project builds on concepts and starter code introduced in the [Building Your Own Database Agent](https://learn.deeplearning.ai/courses/building-your-own-database-agent) Adrian Gonzalez Sanchez (Data & AI Specialist at Microsoft), offered through [DeepLearning.AI](https://www.deeplearning.ai/short-courses/). While the original instructional materials provided foundational examples, this implementation has been customized and extended.
