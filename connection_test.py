from neo4j import GraphDatabase
import logging

# Enable verbose logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neo4j")

def test_direct_connection():
    uri = "neo4j+ssc://8f1f4951.databases.neo4j.io"
    auth = ("neo4j", "K1U2AcP3FgGf5eUwcSaT55atcLxAupNGVCEx1rCjei0")
    
    try:
        driver = GraphDatabase.driver(
            uri,
            auth=auth,
            max_connection_lifetime=30,
            keep_alive=True
        )
        
        with driver.session() as session:
            result = session.run("""
                CALL dbms.components()
                YIELD name, versions
                RETURN name, versions[0] as version
                """)
            print("Neo4j Version:", [record for record in result])
            
        driver.close()
        return True
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        return False

test_direct_connection()