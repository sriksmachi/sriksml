import pandas as pd
from langchain_community.graphs import Neo4jGraph
import os
import tqdm
import logging
from get_monitoring_data import AzureMonitorClient
from constants import *
from dotenv import load_dotenv
import argparse
import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def create_nodes(service_dep_graph, graph):
  logger.info('Creating nodes...')
  nodes = service_dep_graph['AppRoleName'].unique()
  nodes = list({'id': node, 'type': NODE_TYPE_APP_ROLE_NAME} for node in nodes if len(node) > 0)
  nodes.append({'id': ROOT_NODE_ID, 'type': NODE_TYPE_USER})
  for dep in AZURE_DEPENDENCIES:
    app_azure_dep = service_dep_graph[service_dep_graph['DependencyType'].str.contains(dep)]["Name"]
    # remove spaces from the names
    for node in app_azure_dep:
      nodes.append({'id': node, 'type': dep})
  for node in nodes:
    query = MATCH_NODE_QUERY.format(type=node['type'], id=node['id'])
    result = graph.query(query)
    if len(result) == 0:
      query = CREATE_NODE_QUERY.format(type=node['type'], id=node['id'])
      graph.query(query)
      logger.info(f'Node created: {node}')

def get_parent_node(service_dep_graph, operation_ParentId):
  parent_node_type, parent_node = None, None
  parents = service_dep_graph[service_dep_graph['Id'] == operation_ParentId]['AppRoleName']
  if len(parents) > 0:
    parent_node = parents.values[0] if parents.values[0] else ROOT_NODE_ID
    parent_node_type = NODE_TYPE_APP_ROLE_NAME
  else:
    parent_node = ROOT_NODE_ID
    parent_node_type = NODE_TYPE_USER
  return parent_node_type, parent_node

def create_or_update_app_traces(graph, row, child_node, operationId, parent_node, total_duration):
 
  query = MATCH_NODE_QUERY.format(type=APPTRACES, id=operationId)
  result = graph.query(query)
  trace = [row['Message']]
 
  if len(result) == 0:
    # create AppTraces node for the first time, with id as operation id and traces as message
    query = "CREATE (n:AppTraces {id: '%s', traces: %s})" % (operationId, trace)
    result = graph.query(query)
  
  # App Traces node already exists, check if relationship exists
  query = MATCH_APPTRACES_QUERY.format(node_type=NODE_TYPE_APP_ROLE_NAME, child_node=child_node, operation_id=operationId)
  result = graph.query(query)
  if len(result) == 0:
    # create relationship between AppTraces and AppRoleName
    query = MATCH_AND_CREATE_REL_TRACES_QUERY.format(parent_type=NODE_TYPE_APP_ROLE_NAME, parent_id=parent_node, child_type=APPTRACES, child_id=operationId, rel_name="Traces", id=operationId, traces=trace, total_duration=total_duration)
    result = graph.query(query)
  else:
    # update the traces in the AppTraces node
    query = "MATCH (n:AppTraces {id: '%s'}) SET n.traces = n.traces + %s" % (operationId, trace)
    result = graph.query(query)

def create_node_if_not_exists(graph, node_type, node_id):
  query = MATCH_NODE_QUERY.format(type=node_type, id=node_id)
  result = graph.query(query)
  if len(result) == 0:
    query = CREATE_NODE_QUERY.format(type=node_type, id=node_id)
    result = graph.query(query)
    logger.info(f'Node created: {node_id}')
  return result

def build_graph(service_dep_graph, graph, total_duration):
  """
  Builds a graph based on the service dependency graph provided.
  Args:
    service_dep_graph (DataFrame): A DataFrame containing service dependency information.
    graph (Graph): The graph object where nodes and relationships will be created.
    total_duration (int): The total duration for which the graph is being built.
  Returns:
    None
  The function iterates through each row in the service dependency graph DataFrame and performs the following:
  - Extracts necessary information such as Id, ParentId, DurationMs, DependencyType, Type, Name, AppRoleName, AppRoleInstance, OperationId, and TimeGenerated.
  - Creates a child node if it does not already exist in the graph.
  - Skips certain types of dependencies based on predefined conditions.
  - Determines the parent node type and parent node.
  - Updates or creates nodes and relationships in the graph based on the extracted information.
  - Logs warnings if parent or child nodes are not found.
  """
  counter = 0
  for _, row in service_dep_graph.iterrows():
    id = row['Id']
    operation_ParentId = row['ParentId']
    duration = row['DurationMs']
    dependencyType = row['DependencyType']
    # value of type can be AppTraces, AppRequests, AppDependencies
    type = row['Type']
    name = row['Name']
    # if AppRoleName is empty, then use the first two parts of AppRoleInstance
    child_node = row['AppRoleName'] if row['AppRoleName'] != '' else "-".join(row['AppRoleInstance'].split('-')[:2])
    operationId = row['OperationId']
    timeGenerated = row['TimeGenerated']
    
    is_inproc = "token" in name
        
    create_node_if_not_exists(graph, NODE_TYPE_APP_ROLE_NAME, child_node)
    
    if type == APPDEPENDENCIES and "InProc_TT" in dependencyType:
      continue
    
    # assigning default values
    parent_node_type = NODE_TYPE_APP_ROLE_NAME
    child_node_type = NODE_TYPE_APP_ROLE_NAME
    
    if is_inproc:
      parent_node = child_node
      parent_node_type = child_node_type
    else:
      parent_node_type, parent_node = get_parent_node(service_dep_graph, operation_ParentId)
    
    if dependencyType in AZURE_DEPENDENCIES:
      child_node = name
      child_node_type = dependencyType
      
    if type == APPTRACES and child_node:
      # get AppTraces node connect to the child node
      create_or_update_app_traces(graph, row, child_node, operationId, parent_node, total_duration=total_duration)
      continue
    
    # if both parent and child nodes are found
    if parent_node and child_node:
      
      rel_name = REL_NAME_DEPENDS_ON
      
      if dependencyType in AZURE_DEPENDENCIES:
        rel_name = REL_NAME_AZURE_DEPENDS_ON
      
      if type == NODE_TYPE_HTTP_TRACKED_COMPONENT or child_node == parent_node:
        continue
      
      query = MATCH_AND_CREATE_REL_QUERY.format(
        parent_type=parent_node_type, parent_id=parent_node,
        child_type=child_node_type, child_id=child_node,
        rel_name=rel_name, id=id, duration=duration,
        item_type=dependencyType, counter=counter, name=name,
        type=type, operation_id=operationId, timeGenerated=timeGenerated, total_duration=total_duration
      )
      
      result = graph.query(query)
      
      if len(result) == 0:
        logger.info(f'Error: {query}')
      
      counter += 1
      
    else:
      logger.warning(f'Warning: Parent Node or Child Node not found for Id: {id}, child_node: {child_node}, parent_node: {parent_node}')


def main(graph, num_of_operations):
  raw_requests = pd.read_csv(CSV_FILE_REQUESTS)
  azuremonitor_client = AzureMonitorClient(workspace_id=os.getenv("AZURE_LOG_ANALYTICS_WORKSPACE_ID"))
  df_requests = raw_requests[['Id', 'TimeGenerated', 'Name', 'Url', 'DurationMs', 'OperationId', 'ParentId']]
  # POST Rules/Rules is the operationId for which we are building the graph
  df_requests = df_requests[df_requests['Name'] == POST_RULES_RULES]
  anomalus_operation_ids = df_requests[df_requests['DurationMs'] > DURATION_THRESHOLD]['OperationId'].unique()
  normal_operation_ids = df_requests[df_requests['DurationMs'] < DURATION_THRESHOLD]['OperationId'].unique()
  operation_ids = anomalus_operation_ids[:num_of_operations]
  operation_ids = np.append(operation_ids, normal_operation_ids[:num_of_operations])
  for op_id in tqdm.tqdm(operation_ids):
    logger.info(f'Processing Operation ID: {op_id}')
    duration = df_requests[df_requests['OperationId'] == op_id]['DurationMs'].values[0]
    raw_dependencies = azuremonitor_client.get_dependency_graph_by_operationId(op_id)
    create_nodes(raw_dependencies, graph)
    if len(raw_dependencies) == 0:
      logger.info(f'No data found for Operation ID: {op_id}')
      continue
    service_dep_graph = raw_dependencies[raw_dependencies['OperationId'] == op_id].sort_values('TimeGenerated', ascending=True)
    build_graph(service_dep_graph, graph, duration)
    logger.info(f'Graph built for Operation ID: {op_id}')
    logger.info('\n')

if __name__ == '__main__':
  load_dotenv()
  max_operations = 20
  # read command line arguments
  argparser = argparse.ArgumentParser()
  argparser.add_argument('--clear_graph', action='store_true', help='Clear the graph before building', default=True)
  argparser.add_argument('--num_of_operations', type=int, help='Number of operations to process', default=5)
  args = argparser.parse_args()
  
  if args.num_of_operations > max_operations:
    raise ValueError(f'Number of operations should be less than {max_operations}')
  
  os.environ['NEO4J_URI'] = os.getenv('NEO4J_URI')
  os.environ['NEO4J_USERNAME'] = os.getenv('NEO4J_USERNAME')
  os.environ['NEO4J_PASSWORD'] = os.getenv('NEO4J_PASSWORD')
  
  graph = Neo4jGraph()
  
  if args.clear_graph:
    graph.query(CLEAR_GRAPH_QUERY)
  
  main(graph, args.num_of_operations)
