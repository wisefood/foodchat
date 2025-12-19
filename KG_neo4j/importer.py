import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from neo4j import GraphDatabase

URI = os.getenv('URI')
#AUTH = os.getenv('AUTH')
AUTH = (os.getenv("NEO4J_USERNAME"),os.getenv("NEO4J_PASSWORD"))
# Create the driver instance
print('AUTH: ', AUTH)
driver = GraphDatabase.driver(URI, auth=AUTH)




def clean_db(driver):
    """Deletes all nodes and relationships in the database."""
    print("Clearing the database...")
    with driver.session(database="neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")


def create_constraints(driver):
    """Creates unique constraints for nodes to ensure data integrity and faster lookups."""
    print("Creating constraints...")
    with driver.session(database="neo4j") as session: # opening sessions - with keyword closes the session automatically
        # Unique constraint on Recipe title for faster MERGE operations
        session.run("CREATE CONSTRAINT recipe_title IF NOT EXISTS FOR (r:Recipe) REQUIRE r.title IS UNIQUE")
        # Unique constraints for related nodes
        session.run("CREATE CONSTRAINT allergen_name IF NOT EXISTS FOR (a:Allergen) REQUIRE a.name IS UNIQUE")
        session.run("CREATE CONSTRAINT meal_course_name IF NOT EXISTS FOR (m:MealCourse) REQUIRE m.name IS UNIQUE")
        session.run("CREATE CONSTRAINT ingredient_name IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE")
        session.run("CREATE CONSTRAINT diet_name IF NOT EXISTS FOR (d:Diet) REQUIRE d.name IS UNIQUE")
        session.run("CREATE CONSTRAINT dish_type_name IF NOT EXISTS FOR (dt:DishType) REQUIRE dt.name IS UNIQUE")
    print("Constraints created.")


def import_recipes(driver, recipes_data):
    """
    Imports recipe data into Neo4j using a batched transaction.
    This query will:
    1. UNWIND a list of recipe data rows.
    2. MERGE (create or find) the Recipe node based on its unique title.
    3. SET its properties.
    4. UNWIND the list of allergens for that recipe.
    5. MERGE the Allergen node and create a relationship to the Recipe.
    6. UNWIND the list of meal types for that recipe.
    7. MERGE the MealCourse node and create a relationship to the Recipe.
    """
    print(f"Importing {len(recipes_data)} recipes...")
    
    # This is the powerful, batched Cypher query
    query = """
    UNWIND $rows AS row
    
    MERGE (r:Recipe {title: row.title})
    SET r.Id = row.recipe_id,
        r.Duration = row.duration,
        r.Directions = row.directions,
        r.Ingredients = row.ingredients

    FOREACH (allergen_name IN row.allergens |
        MERGE (a:Allergen {name: allergen_name})
        MERGE (r)-[:CONTAINS_ALLERGEN]->(a)
    )

    FOREACH (meal_course_name IN row.meal_course |
        MERGE (mt:MealCourse {name: meal_course_name})
        MERGE (r)-[:IS_MEAL_COURSE]->(mt)
    )

    FOREACH (ingredient_name IN row.ingredient_list |
        MERGE (i:Ingredient {name: ingredient_name})
        MERGE (r)-[:CONTAINS_INGREDIENT]->(i)
    )
    
    FOREACH (diet_name IN [row.diet] |
        MERGE (d:Diet {name: diet_name})
        MERGE (r)-[:HAS_DIET]->(d)
    )

    FOREACH (dish_type_name IN [row.dish_type] |
        MERGE (dt:DishType {name: dish_type_name})
        MERGE (r)-[:IS_DISH_TYPE]->(dt)
    )
    """

    with driver.session(database="neo4j") as session:
        # Run the query with the entire dataset as a parameter
        result = session.run(query, rows=recipes_data)
        summary = result.consume()
        print(f"Import complete. {summary.counters}")

if __name__ == "__main__":
    # Load the CSV file
    try:
        df = pd.read_csv('import/data_kg.csv')
        

    except FileNotFoundError:
        print("Error: data_kg.csv not found. Make sure the path is correct.")
        exit()
    
    # prepare dataset
    df.meal_course = df.meal_course.apply(eval)
    df.allergens = df.allergens.apply(eval)
    df.ingredient_list = df.ingredient_list.apply(eval)

    # Convert the DataFrame to a list of dictionaries, which is the format the driver needs
    recipes_list = df.to_dict('records')

    # Create the driver instance
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")

        # Run the import process
        clean_db(driver)
        create_constraints(driver)
        import_recipes(driver, recipes_list)
    print("\nKnowledge graph creation complete!")


# # Function to run a Cypher query
# def run_query(query, parameters=None):
#     with driver.session() as session:
#         result = session.run(query, parameters)  # Pass parameters directly
#         return list(result) #list(result)


# def create_recipe_nodes(df):
#     query = """
#     MERGE (r:Recipe {title: $title})
#     SET r.Id = $recipe_id,
#         r.Duration = $duration,
#         r.Instructions = $instructions,
#         r.Ingredients = $ingredients,
#         r.Allergens = $allergens, 
#         r.Meal = $meal_types
#     RETURN r
#     """
    
#     for _, row in df.iterrows():
#         parameters = {
#             "title": row['title'],
#             "recipe_id": row['recipe_id'],
#             "duration": row['duration'],
#             "instructions": row['directions'],
#             "ingredients": row['ingredients'],
#             "allergens": row['allergens'],
#             "meal_types": row['meal_types'],
#         }
#         run_query(query, parameters)

# def import_data(driver, query):
#     """Function to run the import query."""
#     with driver.session(database="neo4j") as session:
#         print("Clearing database...")
#         session.run("MATCH (n) DETACH DELETE n") # Optional: Clears the DB before import
        
#         print("Importing data from CSV...")
#         result = session.run(query)
#         summary = result.consume()
#         print(f"Import complete. Query returned {summary.counters}.")