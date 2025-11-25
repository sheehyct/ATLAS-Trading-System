# Build a Working AI Finance Agent with Python | by Random Access | Coding Nexus | Sep, 2025 | Medium

Member-only story

# Build a Working AI Finance Agent with Python

## Deploying LLM for Financial Analysis

[

![Random Access](https://miro.medium.com/v2/resize:fill:48:48/1*IYmdy1jxDa25_hddwCXYVQ.png)





](/@raccess21?source=post_page---byline--d531b34b4b95---------------------------------------)

[Random Access](/@raccess21?source=post_page---byline--d531b34b4b95---------------------------------------)

Follow

5 min read

·

Sep 23, 2025

15

1

Listen

Share

More

![](https://miro.medium.com/v2/resize:fit:1050/0*g3O6uJjTT8aoIXGe)
Photo by Alex Knight on Unsplash

We already know that AI can read and understand financial statements. What about analyzing stock price data?

You may have heard about AI agents and agentic workflows. Agents are essentially a series of steps orchestrated to achieve a specific goal. These steps are connected through specialized prompts that enable the AI to reason about the next action.

You can also read this article for free [here](/@raccess21/d531b34b4b95?sk=83a8d7eb6bc999873726177843e04c86).

Agents represent the future of AI, offering powerful ways to accelerate your data processing workflows. Comment if you want to learn more about **agents** in a future article.

In this article, you’ll get Python code to build an AI agent that generates and executes its own Python code to analyze stock price data.

Let’s dive in!

## Why Python and LLMs for Finance?

Python is leading the way in integration of large language models (LLMs) into market analysis workflows.

Traditionally, analysts used to spend countless hours sifting through market data and run analytics. We’ve already discussed in multiple articles on how to leverage python libraries for high output data manipulation. Now, AI integration can ingest, process, and analyze data autonomously.

When combined with LLMs, Python unlocks new levels of power and scalability in financial analysis. This is where LlamaIndex comes into play.

Many companies and research organizations rely on LlamaIndex to streamline data workflows with LLMs.

LlamaIndex is an open-source framework for building, managing, and querying applications powered by LLMs with external data. It provides a flexible ecosystem for integrating LLMs, handling data indexing, retrieval, and query processing. Using LlamaIndex you can focus on insights rather than infrastructure.

Let’s see it in action.

## Prerequisites

Before running the code, ensure you have the following:

-   An Anthropic API key (sign up at [anthropic.com](http://anthropic.com/) if needed). Store it in a `.env` file as `ANTHROPIC_API_KEY=your_key_here`.
-   Python 3.8+ installed.
-   Install the required packages via pip:

pip install llama-index llama-index\-llms-anthropic python-dotenv yfinance

Note: The agent will generate code that fetches stock data, likely using the `yfinance` library (which we'll assume is installed in your environment). If the generated code uses other libraries, you may need to install them separately.

## Imports and Setup

Start by importing the necessary libraries and loading your environment variables.

from llama\_index.llms.anthropic import Anthropic  
from llama\_index.core import Settings  
from llama\_index.tools.code\_interpreter.base import CodeInterpreterToolSpec  
from llama\_index.core.agent import FunctionCallingAgent  
from dotenv import load\_dotenv  
  
load\_dotenv()

Here, we load the modules and your API key from the `.env` file. Next, initialize the code interpreter tools:

code\_spec = CodeInterpreterToolSpec()  
tools = code\_spec.to\_tool\_list()

This creates a specification for a code interpreter tool, which allows the agent to execute Python code dynamically. We convert it into a list of tools for the agent.

## Configure the LLM with Anthropic’s Claude 3.5 Sonnet

Next, set up the tokenizer and language model.

tokenizer = Anthropic().tokenizer  
Settings.tokenizer = tokenizer  
llm\_claude = Anthropic(model="claude-3-5-sonnet-20241022")

The tokenizer breaks text into tokens for the model to process. We apply it globally via `Settings` for consistency. Then, we instantiate the Claude 3.5 Sonnet model (as of October 2024)—a powerful LLM for reasoning and code generation.

## Set Up the Agent

Create an agent to bridge the tools and the LLM.

agent \= FunctionCallingAgent.from\_tools(  
    tools,  
    llm\=llm\_claude,  
    verbose\=True,  
    allow\_parallel\_tool\_calls\=False,  
)

The `FunctionCallingAgent` uses the tools and LLM to handle tasks. `verbose=True` logs the agent's steps for transparency and debugging. Setting `allow_parallel_tool_calls=False` ensures sequential execution to avoid potential issues.

This agent acts as the “brain,” deciding when to call the code interpreter based on the prompt.

## Craft a Prompt to Analyze Stock Data

Define a stock ticker and a prompt that instructs the agent to generate (and potentially execute) Python code for fetching and analyzing historical stock data.

ticker = "PLTR"  
  
prompt = f"""  
Write Python code to:  
\- Detect the current date.  
\- Based on this date, fetch historical prices for {ticker} from one year ago until today.  
\- Analyze the prices over the past year, including key metrics like average price, volatility, and trends.  
"""  
  
resp = agent.chat(prompt)

The prompt guides the LLM to produce executable Python code. It starts by determining the current date (e.g., using `datetime`), fetches data (likely via `yfinance` or similar), and performs analysis.

When you run this, the agent may:

1.  Reason about the task.
2.  Generate code.
3.  Use the code interpreter tool to execute it and retrieve results.

Example output (LLMs are probabilistic, so yours may vary):

\> Running step ...  
Step inputs: \[...\]  
  
Thought: I need to write Python code for this task. I'll use datetime for the date, yfinance for data, and pandas for analysis.  
Action: code\_interpreter  
Action Input: {"code": "import yfinance as yf\\nimport pandas as pd\\nimport datetime\\n\\ntoday = datetime.date.today()\\none\_year\_ago = today - datetime.timedelta(days=365)\\n\\n\# Fetch data dynamically\\ndata = yf.download('PLTR', start=one\_year\_ago, end=today)\\n\\nif data.empty:\\n    raise ValueError("No data fetched for PLTR in the given date range")\\n\\naverage\_price = data\['Close'\].mean()\\nvolatility = data\['Close'\].std()\\n\\nlast\_close = float(data\['Close'\].iloc\[-1\])\\nfirst\_close = float(data\['Close'\].iloc\[0\])\\n\\ntrend = 'Upward' if last\_close > first\_close else 'Downward'\\n\\nprint(f'Average Price: {average\_price}\\nVolatility: {volatility}\\nTrend: {trend}')\\n"}  
  
Observation: Average Price: 101.135  
Volatility: 41.604  
Trend: Upward  
\[...\]  
  
Final Answer: The analysis shows an average closing price of $101.135 over the past year, with a volatility (standard deviation) of 41.604. The overall trend is upward.

This demonstrates the agent is generating code, running it, and summarizing results.

## True Integration

You can bring in your fully working python scripts to the agent and integrate it for bug free continuation of your workflow. Open the script as a simple text file and feed it in as code to be used by LLM. This way you can modularize and debug your code and once you feel confident you can integrate it in your main workflow.

## Your Next Steps

Now with the setup out of the way, experiment to customize:

-   **Modify the prompt**: Change the ticker (e.g., “AAPL”), time range (e.g., “past 5 years”), or analysis (e.g., “calculate moving averages and plot a chart”).
-   **Add more tools**: Explore LlamaIndex’s other tools for data ingestion or querying.
-   **Handle errors**: If the generated code fails (e.g., missing libraries), install them or refine the prompt to specify imports.
-   **Scale up**: Integrate this into a larger workflow, like combining with financial statements for comprehensive reports.

This agent streamlines stock analysis, saving time and enabling deeper insights. If you run into issues, check the verbose logs or LlamaIndex docs for troubleshooting.

Happy coding! If you’d like to adapt this to other LLMs or expand it, let me know.

## Embedded Content

---