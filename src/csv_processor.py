import pandas as pd

CSV_HUMMUS_PATH = f"../data/data_kg.csv"
CSV_CULINARY_PATH = f"./data/sample_CulinaryDB.csv"




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
                "Title: " + data['title'] + 
                " Ingredients: " + data['ingredients'] + 
                " Directions: " + data['directions']
            )
        elif source_name == 'culinary' : 
            data = pd.read_csv(CSV_CULINARY_PATH)[['title', 'ingredients']]
            
            for col in data.columns:
                data[col] = data[col].apply(lambda x : x.lower().strip())

            
            data['combined_text'] = (
                "Title: " + data['title'] + 
                " Ingredients: " +data['ingredients']
            )
        return data
        



