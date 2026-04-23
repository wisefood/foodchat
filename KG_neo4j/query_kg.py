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
RECIPEWRANGLER_API_URL = os.getenv("RECIPEWRANGLER_API_URL", "http://recipewrangler:8001")

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


def get_filtered_recipe_ids_neo4j(allergens: list, diet: str, limit: int = 5):
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
            if results:
                df = pd.DataFrame(results)
                exclude_ids.extend(df['recipe_id'].values.tolist())
                res_per_courses[course] = df.values.tolist()[:limit]
    return res_per_courses


# Valid dietary tags as stored on RecipeWrangler recipe nodes (confirmed against API)
VALID_RW_DIET_TAGS = {
    "gluten_free", "high-protein", "low-carb", "low-fat",
    "pescatarian", "pescatarian_safe", "vegan", "dairy_free", "nut_free", "vegetarian",
}

# Map common user profile diet values → RecipeWrangler tag string
DIET_TAG_MAP = {
    "gluten_free": "gluten_free",
    "gluten-free": "gluten_free",
    "high_protein": "high-protein",
    "high-protein": "high-protein",
    "low_carb": "low-carb",
    "low-carb": "low-carb",
    "low_fat": "low-fat",
    "low-fat": "low-fat",
    "pescatarian": "pescatarian",
    "pescatarian_safe": "pescatarian_safe",
    "vegan": "vegan",
    "dairy_free": "dairy_free",
    "dairy-free": "dairy_free",
    "nut_free": "nut_free",
    "nut-free": "nut_free",
    "vegetarian": "vegetarian",
    # Non-restrictive / cuisine labels that have no RW tag — silently dropped
    "omnivore": None,
    "mediterranean": None,
    "balanced": None,
    "healthy": None,
}


def _normalize_diet_tags(diet) -> list:
    """Convert user profile diet values to valid RecipeWrangler diet tags.

    Unknown or non-restrictive values (e.g. 'omnivore', 'mediterranean') are
    dropped so they don't accidentally produce empty result sets.
    """
    raw = diet if isinstance(diet, list) else ([diet] if diet else [])
    tags = []
    for d in raw:
        normalized = DIET_TAG_MAP.get(d.lower().strip())
        if normalized is None and d.lower().strip() not in DIET_TAG_MAP:
            # Unknown tag — keep only if it's already a valid RW tag
            if d.lower().strip() in VALID_RW_DIET_TAGS:
                tags.append(d.lower().strip())
            else:
                logger.warning("Dropping unrecognized diet tag %r — not in RecipeWrangler schema", d)
        elif normalized is not None:
            tags.append(normalized)
        # else: mapped to None → intentionally dropped (non-restrictive label)
    return tags


def get_filtered_recipe_ids_api(
    allergens: list,
    diet,
    include_ingredients: list = None,
    exclude_ingredients: list = None,
    limit: int = 5,
    exclude_ids: list = None,
    nutrition_profile: dict = None,
    randomize: bool = True,
):
    """Fetch candidate recipes from the RecipeWrangler foodchat_candidates endpoint.

    Returns recipes already grouped by meal slot (breakfast/lunch/dinner) with
    server-side dish-type tagging — no client-side keyword classification needed.
    """
    base = RECIPEWRANGLER_API_URL.rstrip("/")
    valid_diet_tags = _normalize_diet_tags(diet)

    payload = {
        "user_profile": {
            "allergies": allergens or [],
            "diet": valid_diet_tags,
        },
        "constraints": {
            "include_ingredients": include_ingredients or [],
            "exclude_ingredients": exclude_ingredients or [],
            "exclude_recipe_ids": exclude_ids or [],
            "nutrition_profile": nutrition_profile,
        },
        "quotas": {
            "breakfast": limit,
            "lunch": limit,
            "dinner": limit,
        },
        "randomize": randomize,
    }

    with httpx.Client() as client:
        logger.info("Querying RecipeWrangler foodchat_candidates (limit=%d per slot)", limit)
        response = client.post(
            f"{base}/api/v1/recipes/foodchat_candidates",
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

    slot_results = data.get("results", {}) if isinstance(data, dict) else {}

    res_per_courses = {"breakfast": [], "lunch": [], "dinner": []}
    for slot in res_per_courses:
        for r in slot_results.get(slot, []):
            recipe_entry = [
                r.get("recipe_id", ""),
                r.get("title", ""),
                r.get("ingredients", ""),
                r.get("directions", ""),
            ]
            res_per_courses[slot].append(recipe_entry)
        logger.info("Got %d recipes for %s", len(res_per_courses[slot]), slot)

    return res_per_courses


def get_filtered_recipe_ids(
    allergens: list,
    diet,
    include_ingredients: list = None,
    exclude_ingredients: list = None,
    limit: int = 5,
    exclude_ids: list = None,
    nutrition_profile: dict = None,
    randomize: bool = True,
):
    """Dispatcher: use RecipeWrangler API or Neo4j based on RECIPE_SOURCE env var."""
    if RECIPE_SOURCE == "api":
        return get_filtered_recipe_ids_api(
            allergens, diet,
            include_ingredients=include_ingredients,
            exclude_ingredients=exclude_ingredients,
            limit=limit,
            exclude_ids=exclude_ids,
            nutrition_profile=nutrition_profile,
            randomize=randomize,
        )
    return get_filtered_recipe_ids_neo4j(allergens, diet, limit)