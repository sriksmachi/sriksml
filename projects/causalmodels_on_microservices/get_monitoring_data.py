import requests
import pandas as pd
from dotenv import load_dotenv
import json
from datetime import timedelta
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from azure.core.exceptions import HttpResponseError
import os
import logging

class AzureMonitorClient:
    def __init__(self, workspace_id, credential=None):
        self.workspace_id = workspace_id
        self.credential = credential or DefaultAzureCredential()
        self.client = LogsQueryClient(self.credential)
        self.logger = self._configure_logging()

    def _configure_logging(self):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger

    def read_query_results(self, query, delta=15):
        try:
            td = timedelta(days=delta)
            response = self.client.query_workspace(workspace_id=self.workspace_id, query=query, timespan=td)
            if response.status == LogsQueryStatus.SUCCESS:
                data = response.tables
            else:
                error = response.partial_error
                data = response.partial_data
                self.logger.error(error)
            table = data[0]
            df = pd.DataFrame(data=table.rows, columns=table.columns)
            return df
        except HttpResponseError as err:
            self.logger.fatal("Something fatal happened")
            self.logger.fatal(err)

    def get_dependency_graph_by_operationId(self, operationId):
        try:
            query = f"""union AppRequests,AppDependencies, AppTraces, AppExceptions | where OperationId contains "{operationId}" | where  DependencyType !has "InProc" | project Id, TimeGenerated, Type, OperationId, ParentId, DurationMs, Name, Message, DependencyType, AppRoleInstance, AppRoleName| order by TimeGenerated asc"""
            results = self.client.query_workspace(self.workspace_id, query=query, timespan=timedelta(days=7))
            table = results.tables[0]
            df = pd.DataFrame(data=table.rows, columns=table.columns)
            return df
        except HttpResponseError as err:
            self.logger.fatal("Something fatal happened")
            self.logger.fatal(err)

class MonitoringDataProcessor:
    def __init__(self, config_file):
        load_dotenv()
        self.config = self._load_config(config_file)
        self.azure_monitor_client = AzureMonitorClient(workspace_id=os.getenv("AZURE_LOG_ANALYTICS_WORKSPACE_ID"))

    def _load_config(self, config_file):
        with open(config_file) as f:
            return json.load(f)

    def process_queries(self):
        queries = self.config['queries']
        delta = self.config["general"]['timedelta']
        for query in queries:
            title = query["title"]
            self.azure_monitor_client.logger.info(f"Querying {title}")
            df = self.azure_monitor_client.read_query_results(query["query"], delta)
            if len(df) > 0:
                df.to_csv(f"{title}.csv", index=False)
                self.azure_monitor_client.logger.info(f"Saved {title}.csv")
            else:
                self.azure_monitor_client.logger.info(f"No data found for {title}")

if __name__ == '__main__':
    processor = MonitoringDataProcessor('kql_queries.yaml')
    processor.process_queries()