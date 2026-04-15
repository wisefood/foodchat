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


BREAKFAST_KEYWORDS = {
    # direct meal references
    "breakfast", "brunch", "morning",
    # pancakes / waffles / crepes
    "pancake", "pancakes", "waffle", "waffles", "crepe", "crepes",
    # eggs
    "scramble", "scrambled", "omelet", "omelette", "frittata",
    "eggs benedict", "poached egg", "sunny side", "over easy",
    "hard boiled egg", "soft boiled egg", "egg muffin",
    # toast / bread
    "french toast", "toast", "bagel", "english muffin", "crumpet",
    # cereal / oats / grains
    "oatmeal", "porridge", "cereal", "granola", "muesli", "overnight oats",
    # baked breakfast
    "muffin", "muffins", "scone", "scones", "croissant", "danish", "cinnamon roll",
    "coffee cake", "banana bread",
    # potatoes
    "hash brown", "hash browns", "home fries", "breakfast potato",
    # bowls / smoothies
    "smoothie bowl", "acai bowl", "smoothie", "parfait", "yogurt bowl",
    # other breakfast items
    "bacon and eggs", "sausage and eggs", "breakfast burrito", "breakfast sandwich",
    "breakfast wrap", "breakfast casserole", "shakshuka", "chilaquiles",
    "huevos rancheros", "quiche",
}

LUNCH_KEYWORDS = {
    # direct meal references
    "lunch", "midday",
    # sandwiches
    "sandwich", "sandwiches", "sub", "hoagie", "club sandwich", "grilled cheese",
    "panini", "blt", "po boy", "monte cristo", "open face",
    # wraps / flatbreads
    "wrap", "wraps", "pita", "flatbread", "gyro", "falafel wrap",
    # salads
    "salad", "slaw", "coleslaw", "caesar salad", "cobb salad",
    "greek salad", "nicoise", "grain salad", "chopped salad",
    # soups
    "soup", "bisque", "chowder", "gazpacho", "minestrone", "broth",
    # bowls
    "bowl", "grain bowl", "poke", "buddha bowl", "noodle bowl", "rice bowl",
    # tacos / mexican light
    "taco", "tacos", "quesadilla", "burrito", "tostada", "nachos", "enchilada",
    # burgers
    "burger", "burgers", "slider", "sliders",
    # other lunch items
    "crostini", "bruschetta", "hummus plate", "mezze",
    "spring roll", "spring rolls", "summer roll", "summer rolls",
    "empanada", "empanadas", "samosa", "samosas",
    "quiche lorraine", "croque monsieur", "croque madame",
}

DINNER_KEYWORDS = {
    # direct meal references
    "dinner", "supper", "entree",
    # roasts / grilled
    "roast", "roasted", "grilled", "broiled", "seared",
    "prime rib", "rack of lamb", "pot roast", "rotisserie",
    # steak / chops
    "steak", "fillet", "filet mignon", "ribeye", "sirloin", "t-bone",
    "pork chop", "pork chops", "lamb chop", "lamb chops", "veal chop",
    # stews / braised
    "stew", "braised", "slow cooker", "slow-cooker", "pot pie",
    "beef bourguignon", "coq au vin", "goulash", "tagine",
    # casseroles / bakes
    "casserole", "lasagna", "lasagne", "baked ziti", "moussaka",
    "shepherd's pie", "shepherds pie", "gratin", "au gratin",
    # pasta / noodles
    "pasta", "spaghetti", "fettuccine", "penne", "linguine",
    "carbonara", "bolognese", "alfredo", "puttanesca", "primavera",
    "mac and cheese", "macaroni", "ramen", "lo mein", "pad thai", "chow mein",
    # rice dishes
    "risotto", "paella", "biryani", "jambalaya", "pilaf", "fried rice",
    # curries / asian
    "curry", "tikka masala", "korma", "vindaloo", "rendang",
    "teriyaki", "katsu", "tempura", "kung pao", "general tso",
    "sweet and sour", "orange chicken", "szechuan",
    # stir fry
    "stir fry", "stir-fry",
    # other dinner items
    "meatloaf", "meatball", "meatballs", "stroganoff", "chili", "con carne",
    "stuffed pepper", "stuffed peppers", "stuffed chicken",
    "wellington", "piccata", "marsala", "parmesan crusted",
    "fish and chips", "salmon fillet", "shrimp scampi", "crab cake",
    "lobster", "cioppino", "gumbo", "etouffee",
}


def _classify_course(title, ingredients):
    """Classify a recipe into breakfast, lunch, or dinner using keyword matching."""
    text = f"{title} {ingredients}".lower()
    for keyword in BREAKFAST_KEYWORDS:
        if keyword in text:
            return "breakfast"
    for keyword in LUNCH_KEYWORDS:
        if keyword in text:
            return "lunch"
    for keyword in DINNER_KEYWORDS:
        if keyword in text:
            return "dinner"
    return None


def get_filtered_recipe_ids_api(
    allergens: list, diet: str,
    include_ingredients: list = None, exclude_ingredients: list = None,
    limit: int = 5
):
    """Fetch candidate recipes from the RecipeWrangler param_search API.

    Recipes are classified into breakfast/lunch/dinner using keyword heuristics
    since the API does not support filtering by meal course.
    """
    recipes_per_course = limit
    res_per_courses = {'breakfast': [], 'lunch': [], 'dinner': []}
    base = RECIPEWRANGLER_API_URL.rstrip("/")

    payload = {
        "exclude_allergens": allergens or [],
        "diet_tags": diet if isinstance(diet, list) else ([diet] if diet else []),
        "include_ingredients": include_ingredients or [],
        "exclude_ingredients": exclude_ingredients or [],
        "limit": max(50, limit * 4),
    }

    with httpx.Client() as client:
        logger.info("Querying RecipeWrangler param_search")
        response = client.post(
            f"{base}/api/v1/recipes/param_search",
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        logger.debug("RecipeWrangler param_search response: %s", str(data)[:500])

        results_list = data.get("results", []) if isinstance(data, dict) else []

        unclassified = []
        for r in results_list:
            recipe_id = r.get("recipe_id") or r.get("id")

            details = _fetch_recipe_details(client, recipe_id)
            r_title = r.get("title", "")
            title = details.get("title", r_title) if isinstance(details, dict) else r_title
            ingredients = _format_ingredients(
                details.get("ingredients", "") if isinstance(details, dict) else ""
            )
            directions = _format_instructions(
                (details.get("instructions") or details.get("directions", ""))
                if isinstance(details, dict) else ""
            )

            recipe_entry = [recipe_id, title, ingredients, directions]
            course = _classify_course(title, ingredients)

            if course and len(res_per_courses[course]) < recipes_per_course:
                res_per_courses[course].append(recipe_entry)
            else:
                unclassified.append(recipe_entry)

            # Stop early if all courses are full
            if all(len(v) >= recipes_per_course for v in res_per_courses.values()):
                break

        # Fill any courses that are still short with unclassified recipes
        for course in res_per_courses:
            while len(res_per_courses[course]) < recipes_per_course and unclassified:
                res_per_courses[course].append(unclassified.pop(0))

    for course, recipes in res_per_courses.items():
        logger.info("Got %d recipes for %s", len(recipes), course)

    return res_per_courses


def get_filtered_recipe_ids(
    allergens: list, diet: str,
    include_ingredients: list = None, exclude_ingredients: list = None,
    limit: int = 5
):
    """Dispatcher: use RecipeWrangler API or Neo4j based on RECIPE_SOURCE env var."""
    if RECIPE_SOURCE == "api":
        return get_filtered_recipe_ids_api(
            allergens, diet, include_ingredients, exclude_ingredients, limit
        )
    return get_filtered_recipe_ids_neo4j(allergens, diet, limit)