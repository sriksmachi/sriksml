import pandas as pd
from langchain_community.graphs import Neo4jGraph
import os 
import tqdm

azure_dependencies = ['SQL']

def create_nodes(service_dep_graph, graph):
    print('Creating nodes...')
    nodes = service_dep_graph['cloud_RoleName'].unique()
    nodes = list({'id': node, 'type': 'cloudRoleName'} for node in nodes)
    nodes.append({'id': 'ROOT', 'type': 'cloudRoleName'})
    for dep in azure_dependencies:
        app_azure_dep = service_dep_graph[service_dep_graph['type'] == dep]['name'].unique()
        for node in app_azure_dep:
            nodes.append({'id': node, 'type': dep})
    for node in nodes:
      query = f"CREATE (n: {node['type']} {{id: '{node['id']}'}})"
      graph.query(query)
    print(f'{len(nodes)} nodes created')

def build_graph(service_dep_graph, graph):
    # Code to build the graph goes here
    # add edges
    counter = 0
    for _, row in service_dep_graph.iterrows():
      # get the node
      id = row['id']
      operation_ParentId = row['operation_ParentId']
      duration = row['duration']
      itemType = row['itemType']
      type = row['type']
      name = row['name']
      child_node = row['cloud_RoleName']
      operationId = row['operation_Id']
      
      # node types
      parent_node_type = "cloudRoleName"
      child_node_type = "cloudRoleName"
      
      # find the parent node
      parents = service_dep_graph[service_dep_graph['id'] == operation_ParentId]['cloud_RoleName'][:1]
      parent_node = None
      if len(parents) > 0:
        parent_node = parents.values[0]
      else:
        parent_node = "ROOT"
      
    #   print(f'Parent: {parent_node}, Child: {child_node}')
      # create edge with id as child and operation_ParentId as parent
      if parent_node and child_node:
        # if tracked component, add the time taken is by the parent node
        rel_name = 'DEPENDS_ON'
        if type in azure_dependencies:
          rel_name = 'AZURE_DEPENDS_ON'
          child_node = name
          child_node_type = type
        else:
          child_node_type = 'cloudRoleName'
    
        if type == 'Http (tracked component)' or child_node == parent_node:
          rel_name = 'INPROC_TT'
    
        query = f"""MATCH (a:{parent_node_type} {{id: '{parent_node}'}}), (b:{child_node_type} {{id: '{child_node}'}}) \
          CREATE (a)-[r:{rel_name} {{id: '{id}', duration: {duration}, itemType: '{itemType}', counter: {counter}, \
            operation_Name: '{name}', type: '{type}', operationId: '{operationId}'}}]->(b) RETURN r"""
        # print(query)
        result = graph.query(query)
        if len(result) == 0:
          print(f'Error: {query}')
        counter += 1
        
def main(graph):
    raw_dependencies = pd.read_csv('query_data (33).csv')
    raw_requests = pd.read_csv('rules_data.csv')
    
    # select only required columns
    df_requests = raw_requests[['id', 'timestamp [UTC]', 'name', 'url', 'duration', 'customDimensions', 'operation_Id', 'operation_ParentId', 'performanceBucket']]
    
    # measure at APIM level
    df_requests = df_requests[df_requests['name'] == 'POST Rules/Rules']
    
    operation_ids = df_requests[df_requests['duration'] > 1000]['operation_Id'].unique()[:5]
    print(f'Number of operations: {len(operation_ids)}')
    create_nodes(raw_dependencies, graph)
    print('-'*100)
    for op_id in operation_ids:
       print(f'Adding edges for Operation ID: {op_id}')
       service_dep_graph = raw_dependencies[raw_dependencies['operation_Id'] == op_id].sort_values('timestamp [UTC]', ascending=True)
       service_dep_graph.to_csv('service_dep_graph_temp.csv', index=False)
       build_graph(service_dep_graph, graph)
    
if __name__ == '__main__':
    os.environ['NEO4J_URI'] = 'bolt://localhost:7690'
    os.environ['NEO4J_USERNAME'] = 'neo4j'
    os.environ['NEO4J_PASSWORD'] = 'Password@123'
    graph = Neo4jGraph()
    # clear the graph
    graph.query('MATCH (n) DETACH DELETE n')
    main(graph)