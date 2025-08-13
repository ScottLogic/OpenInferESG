from src.utils.graph_db_utils import execute_query
import logging
from src.prompts import PromptEngine
import json
import asyncio

semantic_layer_ready = asyncio.Event()

logger = logging.getLogger(__name__)

engine = PromptEngine()
relationship_property_query = engine.load_prompt("relationship-property-cypher-query")

node_property_query = engine.load_prompt("node-property-cypher-query")

neo4j_graph_why_prompt = engine.load_prompt("neo4j-graph-why")

neo4j_nodes_understanding_prompt = engine.load_prompt(
    "neo4j-nodes-understanding", neo4j_graph_why_prompt=neo4j_graph_why_prompt
)

neo4j_relationship_property_prompt = engine.load_prompt(
    "neo4j-property-intent-prompt", neo4j_graph_why_prompt=neo4j_graph_why_prompt
)

neo4j_node_property_prompt = engine.load_prompt("neo4j-node-property", neo4j_graph_why_prompt=neo4j_graph_why_prompt)
relationship_query = engine.load_prompt("relationships-query")

neo4j_relationships_understanding_prompt = engine.load_prompt(
    "neo4j-relationship-understanding", neo4j_graph_why_prompt=neo4j_graph_why_prompt
)


async def get_semantic_layer(llm, model, agent):
    finalised_graph_structure = {"nodes": {}, "properties": {}}

    relationship_result = execute_query(relationship_query)
    payload = relationship_result[0]

    nodes = []
    relationships_dict = {}

    # Convert nodes
    for node in payload["nodes"]:
        nodes.append(
            {
                "cypher_representation": f"(:{node['name']})",
                "label": node["name"],
                "indexes": node.get("indexes", []),
                "constraints": node.get("constraints", []),
            }
        )

    # Convert relationships
    for relationship in payload["relationships"]:
        start_node = relationship[0]["name"]
        relationship_type = relationship[1]
        end_node = relationship[2]["name"]
        path = f"(:{start_node})-[:{relationship_type}]->(:{end_node})"

        if relationship_type not in relationships_dict:
            relationships_dict[relationship_type] = {
                "cypher_representation": f"[:{relationship_type}]",
                "type": relationship_type,
                "paths": [],
            }

        relationships_dict[relationship_type]["paths"].append({"path": path, "detail": ""})
    # Convert relationships_dict to a list
    relationships = list(relationships_dict.values())

    finalised_graph_structure = {"nodes": nodes, "relationships": relationships}
    json.dumps(finalised_graph_structure)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(enrich_relationships(llm, model, finalised_graph_structure, agent))
        tg.create_task(enrich_nodes(llm, model, finalised_graph_structure, agent))
        tg.create_task(enriched_rel_properties(llm, model, finalised_graph_structure, agent))
        tg.create_task(enrich_nodes_properties(llm, model, finalised_graph_structure, agent))
    semantic_layer_ready.set()
    logger.info("Semantic layer initialized successfully.")
    return finalised_graph_structure


async def enrich_relationships(llm, model, finalised_graph_structure, agent):
    relationships = finalised_graph_structure["relationships"]
    enriched_relationships_list = []

    for relationship in relationships:
        enriched_relationship = await llm.chat(
            model, neo4j_relationships_understanding_prompt, str(relationship), agent=agent, return_json=True
        )
        enriched_relationships_list.append(json.loads(enriched_relationship))

        finalised_graph_structure["relationships"] = enriched_relationships_list
    logger.debug(f"finalised graph structure with enriched relationships: {finalised_graph_structure}")


async def enrich_nodes(llm, model, finalised_graph_structure, agent):
    neo4j_data = finalised_graph_structure["nodes"]
    print(f"neo4j data: {neo4j_data}")
    enriched_nodes = await llm.chat(model, neo4j_nodes_understanding_prompt, str(neo4j_data), agent=agent, return_json=True)
    enriched_nodes = json.loads(enriched_nodes)
    json.dumps(enriched_nodes)
    finalised_graph_structure["nodes"] = enriched_nodes
    logger.debug(f"finalised graph structure: {finalised_graph_structure}")
    print(f"finalised_graph_structure with nodes: {finalised_graph_structure}")


async def enriched_rel_properties(llm, model, finalised_graph_structure, agent):
    properties_result = execute_query(relationship_property_query)
    rel_properties_neo4j = properties_result[0]
    cleaned_rel_properties = []

    for rel_property in rel_properties_neo4j["relProperties"]:
        cleaned_properties = [prop for prop in rel_property["properties"] if prop["name"] is not None]
        if cleaned_properties:
            rel_property["properties"] = cleaned_properties
            cleaned_rel_properties.append(rel_property)

    rel_properties_neo4j = {"relProperties": cleaned_rel_properties}
    json.dumps(rel_properties_neo4j)

    enriched_rel_properties = await llm.chat(
        model, neo4j_relationship_property_prompt, str(rel_properties_neo4j), agent=agent ,return_json=True
    )
    enriched_rel_properties = json.loads(enriched_rel_properties)

    # Merge properties
    for new_rel in enriched_rel_properties["relProperties"]:
        # FS-67 - This change was done as relType is unavailable here for mistral.
        relationship_type = new_rel.get("relType", None)
        if relationship_type is None:
            continue  # Skip if there's no relationship type specified
        properties_to_add = new_rel.get("property", None)

        for rel in finalised_graph_structure["relationships"]:
            if rel["cypher_representation"] == relationship_type:
                if "properties" not in rel:
                    rel["property"] = []
                rel["property"] = properties_to_add

    logger.debug(f"finalised graph structure with enriched properties: {finalised_graph_structure}")


async def enrich_nodes_properties(llm, model, finalised_graph_structure, agent):
    node_properties_neo4j_result = execute_query(node_property_query)
    node_properties_neo4j = node_properties_neo4j_result[0]
    filtered_payload = {
        "nodeProperties": [
            node
            for node in node_properties_neo4j["nodeProperties"]
            if all(prop["data_type"] is not None and prop["name"] is not None for prop in node["properties"])
        ]
    }
    enriched_node_properties = await llm.chat(
        model, neo4j_node_property_prompt, str(filtered_payload), agent=agent, return_json=True
    )
    enriched_node_properties = json.loads(enriched_node_properties)

    for new_node in enriched_node_properties["nodeProperties"]:
        label = new_node["label"]
        properties_to_add = new_node["properties"]
        # FS-67 - Determine if the structure has nested 'nodes' or not
        # This is a temporary code as we will change it in future to have LLM return the same structure for both Mistral
        # and OpenAI.
        if isinstance(finalised_graph_structure["nodes"], dict):
            # Nodes are nested under a "nodes" key in a dictionary in the case of OpenAI
            node_list = finalised_graph_structure["nodes"]["nodes"]
        else:
            # Nodes are directly in the list in the case of Mistral
            node_list = finalised_graph_structure["nodes"]
        for node in node_list:
            if node["label"] == label:
                if "properties" not in node:
                    node["properties"] = []
                node["properties"] = properties_to_add
    logger.debug(f"finalised graph structure with enriched nodes: {finalised_graph_structure}")
