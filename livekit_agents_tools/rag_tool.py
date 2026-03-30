"""
RAG tools for LiveKit agents.
"""

"""
Get the collection ID from the config of the agent
User query will be provided during the runtime 
There can be multiple collections - search the query in parallel across all collections
Return the most relevant results

Reranking will be done twice for the content (only if there are multiple collections) - first time for each collection, then again for all results
Return the KBID and the content
"""