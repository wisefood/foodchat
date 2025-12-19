import os

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path)

URI = os.getenv("URI")
AUTH = os.getenv('AUTH')
AUTH = (os.getenv('NEO4J_USERNAME'),os.getenv('NEO4J_PASSWORD'))

def run_cypher_query(driver, query, params=None):
    """
    Runs a Cypher query against the database and returns the results.
    """
    with driver.session(database="neo4j") as session:
        result = session.run(query, params)
        # The result object is an iterable of Record objects
        # We convert it to a more friendly list of dictionaries
        return [record.data() for record in result]

def filter_allergens_diet(driver, allergens, diet, meal_course, exclude_ids): 
        query = """
        MATCH (r:Recipe)
        WHERE ALL (allerg IN $allergens
            WHERE NOT EXISTS { 
                (r)-[:CONTAINS_ALLERGEN]->(a:Allergen)
                WHERE toLower(a.name) = toLower(allerg) })
        AND (
            $diet IS NULL 
            OR size($diet) = 0
            OR EXISTS {
                MATCH (r)-[:HAS_DIET]->(d:Diet)
                WHERE any(dietName IN $diet WHERE toLower(d.name) = toLower(dietName))
            }
        )

        AND (
            EXISTS {
            MATCH (r)-[:IS_MEAL_COURSE]->(mc:MealCourse)
            WHERE toLower(mc.name) = toLower($meal_course)
            }
        )
        AND  (
            $exclude_ids IS NULL
            OR size($exclude_ids) = 0
            OR NOT r.Id IN $exclude_ids
        )
        RETURN r.Id as recipe_id
        """
        params = {"allergens": allergens, "diet" : diet, "meal_course" : meal_course, "exclude_ids" : exclude_ids}
        results = run_cypher_query(driver, query, params)
        # print("RESULTS: ", results)
        return results


def get_filtered_recipe_ids(allergens : list, diet : str): 

    res_per_courses = {'breakfast' : [], 'lunch' : [], 'dinner' : []}
    exclude_ids  = []
    print("URI: ", URI, "AUTH: ", AUTH)
    for course in res_per_courses.keys() :
        print("COURSE: ", course)
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
            print("Successfully connected to Neo4j.\n")
            print("PARAMETERS TYPE: ", allergens, diet)
            results = filter_allergens_diet(driver, allergens, diet, course, exclude_ids)
            exclude_ids.extend(pd.DataFrame(results)['recipe_id'].values.tolist())
            res_per_courses[course] = pd.DataFrame(results)['recipe_id'].values.tolist()
    # print("List of filtered recipes: ", res_per_courses)
    return res_per_courses