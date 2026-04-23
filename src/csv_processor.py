import os
import pandas as pd

def _get_csv_path(env_var: str, default_name: str) -> str:
    # 1. Check environment variable
    env_path = os.getenv(env_var)
    if env_path:
        return env_path
    
    # 2. Check local data directory (inside src)
    local_path = os.path.join(os.path.dirname(__file__), "data", default_name)
    if os.path.exists(local_path):
        return local_path
        
    # 3. Check parent data directory (root data)
    parent_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", default_name)
    if os.path.exists(parent_path):
        return parent_path
        
    # Default to local data path if nothing else found
    return f"./data/{default_name}"

CSV_HUMMUS_PATH = _get_csv_path("CSV_HUMMUS_PATH", "data_kg.csv")
CSV_CULINARY_PATH = _get_csv_path("CSV_CULINARY_PATH", "sample_CulinaryDB.csv")




class CSVProcessor: 
    def __init__(self, data_path, embedding_model): 
        self.data_path = data_path
        # self.embeddings = OllamaEmbeddings(model = embedding_model)
        self.embeddings = embedding_model


    def load_csv_data(self, source_name) -> pd.DataFrame : 
        """Process csv data: load the data
        Returns:
            _type_: Pandas DataFrame 
        """
        if source_name == 'hummus' : 
            data = pd.read_csv(CSV_HUMMUS_PATH)[['title', 'ingredients', 'directions',
                                                 'recipe_id', 'allergens', 'meal_course',
                                                 'diet', 'dish_type']] # select the important features

            for col in data.columns: 
                data[col] = data[col].apply(lambda x : x.lower().strip() if type(x) == str else x)

            data['combined_text'] = (
                "Title: " + data['title'].astype(str) + 
                " Ingredients: " + data['ingredients'].astype(str) + 
                " Directions: " + data['directions'].astype(str)
            )
        elif source_name == 'culinary' : 
            data = pd.read_csv(CSV_CULINARY_PATH)[['title', 'ingredients']]
            
            for col in data.columns:
                data[col] = data[col].apply(lambda x : x.lower().strip())

            
            data['combined_text'] = (
                "Title: " + data['title'].astype(str) + 
                " Ingredients: " + data['ingredients'].astype(str)
            )
        return data
        



