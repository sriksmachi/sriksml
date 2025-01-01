from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from azure.search.documents import SearchClient
from rulesEngine import RuleData
import os


class RAGAgent:

    def __init__(self):

        self.llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"])
        self.embedding = AzureOpenAIEmbeddings(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"])
        self.vectorStore = SearchClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
            credential=os.environ["AZURE_SEARCH_CREDENTIAL"])

    def executeRule(self, rule: RuleData):
        '''
        Get response from the RAG model
        '''

        # Retrieve Content
        source_retrievalPrompt = PromptTemplate.from_template(
            template=source_retrievalPrompt)
        retrivedChunk = self.vectorStore.get_document(rule.rule_id)


        # Apply the Rule
prompt = source_retrievalPrompt.format(
    target_df_json=target_df_json, page_content=page_content, json_format=json_format)
llm_response = llm.predict(prompt)
response = self.llm.prompt(query)
return response
