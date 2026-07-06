import sys
import langchain_google_vertexai
import langchain_community.chat_models

langchain_community.chat_models.vertexai = langchain_google_vertexai
sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai

from evaluate_faq_agent_retrieval import main

if __name__ == "__main__":
    raise SystemExit(main())