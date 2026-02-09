import logging
import os

import httpx
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path)

logger = logging.getLogger(__name__)

RECIPE_SOURCE = os.getenv("RECIPE_SOURCE", "neo4j")  # "neo4j" or "api"
RECIPEWRANGLER_API_URL = os.getenv("RECIPEWRANGLER_API_URL", "http://localhost:8000")

# Set up Neo4j connection (only required when RECIPE_SOURCE=neo4j)
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = None

if RECIPE_SOURCE == "neo4j":
    # Support NEO4J_AUTH="username/password", NEO4J_AUTH="password" (defaults user to neo4j),
    # or separate NEO4J_USERNAME + NEO4J_PASSWORD.
    neo4j_auth = os.getenv("NEO4J_AUTH")
    if neo4j_auth:
        if "/" in neo4j_auth:
            username, password = neo4j_auth.split("/", 1)
        else:
            username, password = "neo4j", neo4j_auth
    else:
        username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not username or not password:
            raise ValueError(
                "Set NEO4J_AUTH (username/password) or NEO4J_USERNAME + NEO4J_PASSWORD."
            )
    AUTH = (username, password)

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
        RETURN r.Id as recipe_id, r.title as recipe_title, r.Ingredients as recipe_ingredients, 
               r.Directions as recipe_directions
        """
        params = {"allergens": allergens, "diet" : diet, "meal_course" : meal_course, "exclude_ids" : exclude_ids}
        results = run_cypher_query(driver, query, params)
        return results


def get_filtered_recipe_ids_neo4j(allergens: list, diet: str):
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
            res_per_courses[course] = pd.DataFrame(results).values.tolist()[:4]
    return res_per_courses


def _fetch_recipe_details(client, recipe_id):
    """Fetch full recipe details (ingredients, instructions) by ID."""
    base = RECIPEWRANGLER_API_URL.rstrip("/")
    resp = client.get(f"{base}/api/v1/recipes/{recipe_id}", timeout=15.0)
    resp.raise_for_status()
    return resp.json()


def _format_ingredients(ingredients):
    """Convert ingredients list-of-dicts to a readable string."""
    if isinstance(ingredients, str):
        return ingredients
    if isinstance(ingredients, list):
        parts = []
        for ing in ingredients:
            if isinstance(ing, dict):
                name = ing.get("name", "")
                measurement = ing.get("measurement", "")
                parts.append(f"{measurement} {name}".strip() if measurement else name)
            else:
                parts.append(str(ing))
        return ", ".join(parts)
    return str(ingredients)


def _format_instructions(instructions):
    """Convert instructions list to a readable string."""
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, list):
        return " ".join(str(step) for step in instructions)
    return str(instructions)


def get_filtered_recipe_ids_api(allergens: list, diet: str):
    """Fetch candidate recipes from the RecipeWrangler API instead of Neo4j.

    1. POST /api/v1/recipes/search  -> list of {recipe_id, title, ...}
    2. GET  /api/v1/recipes/{id}    -> full details (ingredients, instructions)
    """
    res_per_courses = {'breakfast': [], 'lunch': [], 'dinner': []}
    exclude_ids = []
    base = RECIPEWRANGLER_API_URL.rstrip("/")

    with httpx.Client() as client:
        for course in res_per_courses.keys():
            logger.info("Querying RecipeWrangler API for course: %s", course)
            payload = {
                "question": f"Find me a {course} recipe",
                "exclude_allergens": allergens or [],
            }
            response = client.post(
                f"{base}/api/v1/recipes/search",
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            # The search endpoint wraps results under "result.results"
            if isinstance(data, dict):
                results_list = (
                    data.get("result", {}).get("results")
                    or data.get("results")
                    or data.get("recipes")
                    or []
                )
            else:
                results_list = data

            course_recipes = []
            for r in results_list:
                recipe_id = r.get("recipe_id") or r.get("id")
                if recipe_id in exclude_ids:
                    continue

                # Fetch full recipe details
                details = _fetch_recipe_details(client, recipe_id)
                title = details.get("title", r.get("title", ""))
                ingredients = _format_ingredients(details.get("ingredients", ""))
                directions = _format_instructions(
                    details.get("instructions") or details.get("directions", "")
                )

                course_recipes.append([recipe_id, title, ingredients, directions])
                exclude_ids.append(recipe_id)
                if len(course_recipes) >= 4:
                    break

            res_per_courses[course] = course_recipes
            logger.info("Got %d recipes for %s", len(course_recipes), course)

    return res_per_courses


def get_filtered_recipe_ids(allergens: list, diet: str):
    """Dispatcher: use RecipeWrangler API or Neo4j based on RECIPE_SOURCE env var."""
    if RECIPE_SOURCE == "api":
        return get_filtered_recipe_ids_api(allergens, diet)
    return get_filtered_recipe_ids_neo4j(allergens, diet)