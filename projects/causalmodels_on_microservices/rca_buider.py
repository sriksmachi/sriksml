import networkx as nx
from neo4j import GraphDatabase
import matplotlib
import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
driver = GraphDatabase.driver(os.environ['NEO4J_URI'], auth=("neo4j", os.environ['NEO4J_PASSWORD']))


def graph_from_cypher(operationId, services=None, filter=""):
    G = nx.MultiDiGraph()
    if services:
      query = f"MATCH (n:appRoleName)-[r:DEPENDS_ON]-(m:appRoleName) where (r.operationId = '{operationId}' and n.id in {services} and m.id in {services} ) RETURN n, r, m"
    else:
      # get all the nodes and relationships for the operationId
      query = f"MATCH (n)-[r]-(m) where r.operationId = '{operationId}' and {filter}  RETURN n, r, m"
    results = driver.session().run(query)
    nodes = list(results.graph()._nodes.values())
    for node in nodes:
        G.add_node(
            node._properties['id'], labels=node._labels, properties=node._properties)
    rels = list(results.graph()._relationships.values())
    for rel in rels:
        G.add_edge(rel.start_node._properties['id'], rel.end_node._properties['id'],
                   key=rel.element_id, type=rel.type, properties=rel._properties)
    return G, nodes, rels


def get_nodes_rels(operationIds, filter):
  graph, nodes, rels = None, [], []
  for opId in operationIds:
      g, n, r = graph_from_cypher(opId, filter=filter)
      nodes.extend(n)
      rels.extend(r)
      graph = nx.compose(graph, g) if graph else g
  return graph, nodes, rels

def get_total_duration(duration_df, operationId):
    total_duration = []
    for opId in operationId:
        total_duration.extend(duration_df[duration_df['operationId'] == opId]['total_duration'].unique()[:1]/1000)
    return total_duration

def get_data(operationIds, filter):
  graph, nodes, rels = get_nodes_rels(operationIds, filter)
  duration_df = pd.DataFrame.from_records([{
                                            "child": rel.nodes[0]['id'],
                                            "duration": rel._properties['duration']/1000,
                                            "operationId":rel._properties['operationId'],
                                            "total_duration": rel._properties["total_duration"]
                                        } for rel in rels])
  p_duration_df = duration_df.groupby(['operationId', 'child']).agg({
                                    'duration': 'sum'}).reset_index()
  p_duration_df = p_duration_df.pivot(
      index = 'operationId', columns = 'child', values = 'duration').reset_index()
  p_duration_df = p_duration_df.fillna(0)
  p_duration_df['ROOT'] = get_total_duration(duration_df, p_duration_df['operationId'].to_list())
  p_duration_df = p_duration_df.drop(columns=['operationId'])
  p_duration_df.index.name = None
  p_duration_df.columns.name = None
  return p_duration_df, graph

def get_data_from_graph(graph):
# get operation Ids part of the graph
   query = "match(n)-[r]-(m) return distinct r.operationId, r.total_duration"
   records = graph.query(query)
   operationIds = []
   for record in records:
       operationIds.append(
           {'operationId': record['r.operationId'], 'total_duration': record['r.total_duration']})
   df_operationIds = pd.DataFrame(operationIds)
   df_operationIds_anomaly = df_operationIds[df_operationIds['total_duration']
       > 2000]["operationId"].unique()
   df_operationIds_normal = df_operationIds[df_operationIds['total_duration']
       <= 2000]["operationId"].unique()
   return df_operationIds_anomaly, df_operationIds_normal
