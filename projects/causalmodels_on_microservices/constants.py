## Constants
APPTRACES = 'AppTraces'
APPREQUESTS = 'AppRequests'
APPDEPENDENCIES = 'AppDependencies'
AZURE_DEPENDENCIES = ['SQL']
NODE_TYPE_APP_ROLE_NAME = 'appRoleName'
NODE_TYPE_USER = 'User'
NODE_TYPE_HTTP_TRACKED_COMPONENT = 'Http (tracked component)'
REL_NAME_DEPENDS_ON = 'DEPENDS_ON'
REL_NAME_AZURE_DEPENDS_ON = 'AZURE_DEPENDS_ON'
REL_NAME_INPROC_TT = 'INPROC_TT'
ROOT_NODE_ID = 'ROOT'
CSV_FILE_REQUESTS = 'requests.csv'
CSV_FILE_SERVICE_DEP_GRAPH_TEMP = 'service_dep_graph_temp.csv'
MATCH_NODE_QUERY = "MATCH (n: {type}) WHERE n.id = '{id}' RETURN n"
CREATE_NODE_QUERY = "CREATE (n: {type} {{id: '{id}'}})"
MATCH_AND_CREATE_REL_QUERY = """MATCH (a:{parent_type} {{id: '{parent_id}'}}), (b:{child_type} {{id: '{child_id}'}}) \
CREATE (a)-[r:{rel_name} {{id: '{id}', duration: {duration}, dependencyType: '{item_type}', counter: {counter}, \
operation_Name: '{name}', type: '{type}', operationId: '{operation_id}'}}]->(b) RETURN r"""
CLEAR_GRAPH_QUERY = 'MATCH (n) DETACH DELETE n'
POST_RULES_RULES = 'POST Rules/Rules'
DURATION_THRESHOLD = 2000
MATCH_AND_CREATE_REL_TRACES_QUERY = """MATCH (a:{parent_type} {{id: '{parent_id}'}}), (b:{child_type} {{id: '{child_id}'}}) CREATE (a)-[r:{rel_name} {{id: '{id}', traces: {traces}}}]->(b) RETURN r"""
MATCH_APPTRACES_QUERY = "MATCH(n:AppTraces)<-[r:Traces]-(m:{node_type}) WHERE m.id = '{child_node}' and n.id = '{operation_id}' RETURN n"