# main.py
# langchain agent that summarizes stock movement and headlines
# consulting-style output using real-time finance + news data

# -- #
# Was tempted to use NVIDIA's CUDA here, but can’t unless we run a local LLM (e.g., via HuggingFace or vLLM with GPU).
# This demo uses OpenAI’s API, so inference happens remotely.
# CUDA acceleration is best reserved for on-premise workloads or when business needs demand full model control.
# -- #

# nb: we ditched initialize_agent, which is unstable under strict schema rules in latest LangChain versions

import os
import datetime
import yfinance as yf
from dotenv import load_dotenv
from news_utils import get_business_news
from langchain.chat_models import ChatOpenAI

# Load key securely
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
if not key:
    raise RuntimeError("Missing OPENAI_API_KEY.")

# Get stock price movement summary
def get_price_summary(ticker: str = "MSFT") -> str:
    today = datetime.date.today()
    past = today - datetime.timedelta(days=7)
    data = yf.download(ticker, start=past, end=today)
    if data.empty:
        return f"Could not retrieve data for {ticker}."
    delta = data['Close'][-1] - data['Close'][0]
    pct = (delta / data['Close'][0]) * 100
    return f"{ticker} moved {delta:.2f} USD over the past week ({pct:.2f}%)."

# Run the final pipeline manually
if __name__ == "__main__":
    question = "Summarize Microsoft’s weekly stock movement and news. What might a client want to do next?"

    # Get data
    price = get_price_summary("MSFT")
    news = get_business_news("Microsoft")

    # Combine into prompt
    prompt = f"""You are a financial analyst.
Stock: {price}
News: {news}
What should a client consider next?
"""

    # Run LLM and print result
    llm = ChatOpenAI(openai_api_key=key, temperature=0)
    response = llm.predict(prompt)
    print("\nInsight:\n", response)
