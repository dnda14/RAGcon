# Consultas Cypher para crear el Grafo a partir de las tripletas extraídas

MERGE_ENTITY_QUERY = """
MERGE (e:Entidad {nombre: $nombre})
ON CREATE SET e.tipo = 'Desconocido', e.creado_en = datetime()
RETURN e
"""

MERGE_RELATIONSHIP_QUERY = """
MATCH (s:Entidad {nombre: $sujeto})
MATCH (o:Entidad {nombre: $objeto})
MERGE (s)-[r:RELACION_EXTRAIDA {tipo_relacion: $relacion}]->(o)
RETURN r
"""

# Podríamos mejorar la consulta en una sola, pero separarla
# nos permite asegurar que los nodos existan incluso si falla la arista.
CREATE_TRIPLET_QUERY = """
MERGE (s:Entidad {nombre: $sujeto})
MERGE (o:Entidad {nombre: $objeto})
MERGE (s)-[r:RELACION {tipo: $relacion}]->(o)
RETURN s, r, o
"""
