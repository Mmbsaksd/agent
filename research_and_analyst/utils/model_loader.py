import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from research_and_analyst.utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_groq import ChatGroq
from research_and_analyst.logger import GLOBAL_LOGGER as log
from research_and_analyst.exception.custom_exception import ResearchAnalystException


class ApiKeyManager:
    """
    Loads and manages all environment-based API keys.
    """

    def __init__(self):
        load_dotenv()

        self.api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        }

        log.info("Initializing ApiKeyManager")

        # Log loaded key statuses without exposing secrets
        for key, val in self.api_keys.items():
            if val:
                log.info(f"{key} loaded successfully from environment")
            else:
                log.warning(f"{key} is missing in environment variables")

    def get(self, key: str):
        """
        Retrieve a specific API key.

        Args:
            key (str): Name of the API key.

        Returns:
            str | None: API key value if found.
        """
        return self.api_keys.get(key)


class ModelLoader:
    """
    Loads embedding models and LLMs dynamically based on YAML configuration and environment settings.
    """

    def __init__(self):
        """
        Initialize the ModelLoader and load configuration.
        """
        try:
            self.api_key_mgr = ApiKeyManager()
            self.config = load_config()
            log.info("YAML configuration loaded successfully", config_keys=list(self.config.keys()))
        except Exception as e:
            log.error("Error initializing ModelLoader", error=str(e))
            raise ResearchAnalystException("Failed to initialize ModelLoader", sys)

    # ----------------------------------------------------------------------
    # Embedding Loader
    # ----------------------------------------------------------------------
    def load_embeddings(self):
        """
        Load and return an embedding model based on configured provider.

        Supported providers:
            - Google Generative AI
            - Azure OpenAI

        Returns:
            Embedding model instance.
        """
        try:
            embedding_config = self.config["embedding_model"]
            provider = embedding_config.get("provider", "google")
            model_name = embedding_config.get("model_name")

            log.info("Loading embedding model", provider=provider, model=model_name)

            # Ensure event loop exists for gRPC-based embedding API
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            if provider == "google":
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                )

            elif provider == "azure":
                embeddings = AzureOpenAIEmbeddings(
                    model=model_name,
                    api_key=self.api_key_mgr.get("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=embedding_config.get("azure_endpoint"),
                    api_version=embedding_config.get("api_version"),
                )

            else:
                log.error("Unsupported embedding provider", provider=provider)
                raise ValueError(f"Unsupported embedding provider: {provider}")

            log.info("Embedding model loaded successfully", provider=provider, model=model_name)
            return embeddings

        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ResearchAnalystException("Failed to load embedding model", sys)

    # ----------------------------------------------------------------------
    # LLM Loader
    # ----------------------------------------------------------------------
    def load_llm(self):
        """
        Load and return a chat-based LLM according to the configured provider.

        Supported providers:
            - OpenAI
            - Google (Gemini)
            - Groq
            - Azure OpenAI

        Returns:
            ChatOpenAI | ChatGoogleGenerativeAI | ChatGroq | AzureChatOpenAI
        """
        try:
            llm_block = self.config["llm"]
            provider_key = os.getenv("LLM_PROVIDER", llm_block.get("default_provider", "openai"))
            providers = llm_block.get("providers", {})

            if provider_key not in providers:
                log.error("LLM provider not found in configuration", provider=provider_key)
                raise ValueError(f"LLM provider '{provider_key}' not found in configuration")

            llm_config = providers[provider_key]
            provider = llm_config.get("provider")
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0.2)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            log.info("Loading LLM", provider=provider, model=model_name)

            if provider == "google":
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

            elif provider == "groq":
                llm = ChatGroq(
                    model=model_name,
                    api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                    temperature=temperature,
                )

            elif provider == "openai":
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                    temperature=temperature,
                )

            elif provider == "azure":
                llm = AzureChatOpenAI(
                    api_key=self.api_key_mgr.get("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=llm_config.get("azure_endpoint"),
                    api_version=llm_config.get("api_version"),
                    deployment_name=llm_config.get("deployment_name"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            else:
                log.error("Unsupported LLM provider encountered", provider=provider)
                raise ValueError(f"Unsupported LLM provider: {provider}")

            log.info("LLM loaded successfully", provider=provider, model=model_name)
            return llm

        except Exception as e:
            log.error("Error loading LLM", error=str(e))
            raise ResearchAnalystException("Failed to load LLM", sys)


# ----------------------------------------------------------------------
# Standalone Testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        loader = ModelLoader()

        # Test LLM
        llm = loader.load_llm()
        print(f"LLM Loaded: {llm}")
        result = llm.invoke("Hello, how are you?")
        print(f"LLM Result: {result.content[:200]}")

        log.info("ModelLoader test completed successfully")

    except ResearchAnalystException as e:
        log.error("Critical failure in ModelLoader test", error=str(e))
