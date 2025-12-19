import asyncio
import time
from typing import Any, Generator, List, Tuple

import gradio as gr
from colorama import Fore, Style

from agents import QueryClassifier, RAGReadyPreparator, SimpleChatBot
from custom_js import pop_up
from user_profiles import (get_user_profiles, save_user_profile,
                           setup_user_session, update_user_profile_history)
from utils import write_to_file

# Colors from your logo (approx)
PRIMARY_COLOR = "#8C1675"  # Purple
ACCENT_COLOR = "#D4375B"   # Pink

import re

def pretty_format_recipes(recipe_data):
    """
    Parses and formats a tuple of raw recipe strings into a clean,
    human-readable format.

    Args:
        recipe_data (tuple): A tuple of strings, where each string
                             contains a recipe with 'Title:', 'Ingredients:',
                             and 'Directions:' sections.
    """

    result = ''
    result += "Here is your meal plan:"
    print("Your Meal Plan:")
    print("=" * 50)

    # Assign meal types based on order
    meal_types = {
        0: "Breakfast",
        1: "Lunch",
        2: "Dinner"
    }

    for i, raw_recipe in enumerate(recipe_data):
        try:
            # Clean up known data issues, like the stray characters in the second recipe
            # and extra whitespace.
            clean_recipe = raw_recipe.strip().replace(",t.',", "").replace("',", "")

            # Use regular expressions to robustly find the sections,
            # ignoring case and handling varied spacing.
            title_match = re.search(r'Title:(.*?)\s*Ingredients:', clean_recipe, re.IGNORECASE | re.DOTALL)
            ingredients_match = re.search(r'Ingredients:(.*?)\s*Directions:', clean_recipe, re.IGNORECASE | re.DOTALL)
            directions_match = re.search(r'Directions:(.*)', clean_recipe, re.IGNORECASE | re.DOTALL)

            if not all([title_match, ingredients_match, directions_match]):
                print(f"\n--- Could not parse Recipe {i+1} ---")
                print(f"Original data: {raw_recipe}\n")
                continue

            title = title_match.group(1).strip().title()
            ingredients_raw = ingredients_match.group(1).strip()
            directions_raw = directions_match.group(1).strip()

            # --- FORMATTING AND PRINTING ---

            # Print Header
            meal_name = meal_types.get(i, f"Recipe {i+1}")
            result += f"\n\n--- {meal_name}: {title} ---\n\n"
            print(f"\n--- {meal_name}: {title} ---\n")

            # Print Ingredients
            result += "Ingredients:"
            print("Ingredients:")
            # Split ingredients by comma and clean up each item
            ingredient_list = [ing.strip().capitalize() for ing in ingredients_raw.split(',') if ing.strip()]
            for item in ingredient_list:
                result += f"\n  - {item}"
                print(f"  - {item}")

            # Print Directions
            result += "\n\nDirections:"
            print("\nDirections:")
            # Split directions by sentences (periods) and clean up each step
            steps = [step.strip().capitalize() for step in directions_raw.split('.') if step.strip()]
            for j, step in enumerate(steps, 1):
                result += f"\n  {j}. {step}."
                print(f"  {j}. {step}.")

            #result += "\n" + "=" * 50
            print("\n" + "=" * 50)

        except Exception as e:
            print(f"\n--- An error occurred while parsing Recipe {i+1} ---")
            print(f"Error: {e}")
            print(f"Original data: {raw_recipe}\n")
            print("=" * 50)

    return result


# Main execution block to demonstrate the function
if __name__ == "__main__":
    # The data provided in the prompt
    data = (
        'Title: low carb low sodium breakfast burritos Ingredients: 2 lbs ground turkey, 2 teaspoons oregano, 2 teaspoons garlic powder, 2 teaspoons sage, 2 teaspoons black pepper, 2 teaspoons fennel seeds, 2 teaspoons thyme, 2 teaspoons cinnamon, 1 tablespoon brown sugar, 1 teaspoon white pepper, 1/2 teaspoon ground ginger, 1/4 teaspoon ground cloves, 1/4 teaspoon nutmeg, 1/4 teaspoon allspice, 1/4 teaspoon marjoram, 1/8 teaspoon red pepper flakes, 32 ounces egg whites, 12 eggs, 4 cups low-sodium swiss cheese shredded, 17 la tortilla factory low-carb flour tortillas large whole wheat Directions: mix meat and spices together and refrigerate for 24 to 48 hours. cook meat and drain off fat. while meat is cooking cook eggs and egg whites together in another pan. when meat and eggs are done cooking place in large mixing bowl with the shredded cheese and mix till well blended. warm tortillas and then scoop approximately 1/2 cup of filling into tortilla and roll. the burritos can then be placed in separate baggies and frozen for future use.',
        'Title: all purpose freezer meatballs Ingredients: 1 lb fine ground beef at least 85 even better grind your own, 1 medium yellow onion cut to fit through the grinder, 1/2 cup panko breadcrumbs or 1/2 cup fine unseasoned breadcrumbs plus some to roll them in, 1 small egg, 1 teaspoon powdered ginger, 1 teaspoon allspice, 1 tablespoon old fashioned tasty and smelly blackstrap/sulphered molasses, salt to taste as you would a meat loaf Directions: i mix the panko ginger allspice salt and egg in a small bowl. ginger, allspice, and molasses may and likely should be increased run it all, with the beef, through the fine die on your food grinder. if you do not have a grinder, purchase the beef fine grind, at least twice through the grinder. and chop the onion fine in your food processor. i\'m looking here for a very dense end product. mix well by hand. form into no-larger-than-pingpong ball size. roll lightly in that excess panko or bread crumbs i forgot to tell you to put in a plate. place on a grill rack, hopefully fine grate enough so they don\'t fall through. oven at 400 (preheat). bake about 20 min or until that instant thermometer reads 145+. let them cool completely and put what you\'re not going to use now in a freezer bag. that extra roll in panko keeps them loose in the bag. for use? put the balls and some (jarred) pasta sauce, gravy, whatever you fancy in the microwave to heat. place in/on bun, spaghetti, hash browned or mashed potatoes, rice, or whatever the spirit moves you to. it\'s a handy meal-in-a-hurry. enjoy and let me know any other uses you can find.",t.',
        'Title: busy day meatloaf Ingredients: 1 cup herbed croutons, 1/2 cup milk, 1 egg, 2 teaspoons worcestershire sauce, 1/4 cup finely chopped onion, 1 teaspoon salt, 1 lb ground beef, catsup Directions: combine croutons & milk in a large mixing bowl. let stand about 5 minutes or till croutons are softened. add egg, worcestershire sauce, onion and salt. beat well. add ground beef and mix until combined. shape into loaf and place in greased or foil lined shallow pan. score loaf by making several grooves across top. fill with catsup (i frost the whole loaf with catsup). bake in oven at 350° about 45 minutes to an hour or till well glazed.'
    )

    # Call the function with the data
    pretty_format_recipes(data)


class UI: 
    def __init__(self, foodchat, args): 
        self.foodchat = foodchat
        # self.rag_chain = rag_chain
        self.args = args
        self.user_profiles = get_user_profiles()
        self.current_state = "welcome"
        self.temp_data = {}
        self.current_profile = None
        self.interaction_generator: Generator[str, str, Any] | None
        self.pending_rag_data: dict | None = None
        self.rag_preparator = RAGReadyPreparator()
        self.rag_chain_part1 = self.foodchat.create_pre_clarification_chain(args)
        self.rag_chain_part2 = self.foodchat.create_post_clarification_chain()
        # self.waiting_for_answer = False
        # self.current_question = None
        # self.user_answer = None
        # self.rag_chain = foodchat.create_chain(
        #     args,
        #     ask_user_fn=self.ask_user_question  # Inject our method
        # )

    def handle_message(self, message: str, chat_history: List[Tuple[str, str]]): 

        message = message.strip().lower()
        print(Fore.RED + "Current State: " + self.current_state +  Style.RESET_ALL)

        state_handlers = {
            "welcome" : self._handle_welcome_state,
            "has_id" : self._handle_has_id_state, 
            "get_id" : self._handle_get_id_state, 
            "create_choice" : self._handle_create_choice_state, 
            "create_profile" : self._handle_create_profile_state, 
            "ready" : self._handle_ready_state, 
            "clarifying_question" : self._handle_clarifying_question_state, 
            "feedback" : self._handle_ask_user_feedback,
        }

        handler = state_handlers.get(self.current_state)
        if handler: 
            return handler(message, chat_history)
        return chat_history, ""
    
    def _handle_welcome_state(self, chat_history: List[Tuple[str, str]]):
        new_history = chat_history.copy()
        new_history.append((None, "WELCOME TO FOODCHAT!" ))
        new_history.append((None, "Do you have a user ID? (yes/no)"))
        self.current_state = "has_id"
        return new_history
    
    def _handle_has_id_state(self, message: str, chat_history: List[Tuple[str, str]]): 

        if message.lower() not in ["yes", "no"]: 
            chat_history.append((message, "Please answer 'yes' or 'no'."))
            return chat_history, "" 
        
        if message == "yes": 
            chat_history.append((message, "Please enter your user ID: "))
            self.current_state = "get_id"
        else: 
            self.current_state = 'create_profile'
            chat_history.append((message, "Let's create a new profile. Choose a unique user id (this will be your username):"))
        
        return chat_history, ""
    
    def _handle_get_id_state(self, message: str, chat_history: List[Tuple[str, str]]):

        if message in self.user_profiles: 
            self.current_profile = self.user_profiles[message]
            welcome_msg = f"Welcome back {self.current_profile.get('user_id', 'User')}!"
            chat_history.append((message, welcome_msg))
            invite_for_query = f"How can I help you today?"
            chat_history.append((None, invite_for_query ))
            self.current_state = 'ready'
        else: 
            chat_history.append((message, f"User ID: '{message}' not found. Create a new Profile?"))
            # self.current_state = "has_id"
        
            self.temp_data['missing_data'] = message
            self.current_state = "create_choice"

        return chat_history, ""
    
    def _handle_create_choice_state(self, message:str, chat_history: List[Tuple[str, str]]): 
        if message == 'yes': 
            self.current_state = "create_profile"
            chat_history.append((message, "Great! Enter a username: "))
        else: 
            self.current_profile = self.user_profiles['default_user']
            chat_history.append((message, "Using default profile."))
            self.current_state = "ready"
        
        return chat_history, ""
    
    def _handle_create_profile_state(self, message:str, chat_history:List[Tuple[str, str]]):

        step = self.temp_data.get('profile_step', 'username')
        #new_profile = {"username" : "",
        #               "diet" : [],
        #               "allergies" : [],
        #               "preferences" : [],
        #               "location" : ""
        #               }
        if step == 'username' :
            username = message
            if not username: 
                chat_history.append((message, 'Username can not be empty.'))
                self.current_state = 'create_profile'
                return chat_history , ""
            elif username in self.user_profiles.keys():
                chat_history.append((message, 'Username  already exists, please choose another one.'))
                self.current_state = 'create_profile'
                return chat_history, ""
            else: 
                # new_profile["user_id"] = username

                self.current_profile = {
                    "user_id" : username
                }
                self.temp_data['profile_step'] = 'allergies'
                chat_history.append((None, f"Great! Your username is {username}."))
                chat_history.append((None, "Do you have any allergies? If yes, can you list them: "))
                return chat_history, ""

        elif step == "allergies" :
            allergies = message
            if not allergies: 
                chat_history.append((None, "No allergies? Perfect!"))
                self.current_profile['allergies'] = []
                chat_history.append((None, "Do you have one or several diet restrictions? If yes, can you list them: "))
                self.temp_data['profile_step'] = "diet"

                return chat_history, ""
            else: 
                chat_history.append((allergies, f"Ok! I registered that."))
                self.current_profile['allergies'] = [x.strip() for x in allergies.lower().split(',')]
                self.temp_data['profile_step'] = "diet"
                chat_history.append((None, "Do you have one or several diet restrictions? If yes, can you list them: "))
                return chat_history, ""
        
        elif step == "diet":
            diet = message
            if not diet: 
                chat_history.append((None, "No diet restriction? Perfect!"))
                self.temp_data['profile_step'] = "preferences"
                self.current_profile['diet'] = []
                chat_history.append((None, "Do you have any particular preferences, like a particular cuisine or ingredient? If yes, can you list them:  "))
                return chat_history, ""
            else: 
                chat_history.append((diet, 'I have recorded your diet preferences'))
                self.temp_data['profile_step'] = "preferences"
                self.current_profile['diet'] = [x.strip() for x in diet.lower().split(',')]
                chat_history.append((None, "Do you have any particular preferences, like a particular cuisine or ingredient? If yes, can you list them:  "))
                return chat_history, ""
        
        elif step == "preferences":
            preferences = message
            if not preferences: 
                chat_history.append((None, "No particular preferences? Perfect!"))
                self.temp_data['profile_step'] = "end"
                self.current_profile['preferences'] = []
                return chat_history, ""
            else:
                chat_history.append((preferences, 'I have recorded your preferences'))
                self.temp_data['profile_step'] = "end"
                self.current_profile['preferences'] = [x.strip() for x in preferences.lower().split(',')]
                return chat_history, ""

        elif step == 'end': 
            chat_history.append((None,f"So just to summarise, your username is {self.current_profile['user_id']},\
                                    You are allergic to {self.current_profile['allergies'] if len(self.current_profile['allergies'])>0 else "nothing"}\
                                    Your diet preferences are: {self.current_profile["diet"] if len(self.current_profile['diet'])>0 else "nothing"}\
                                    And your preferences in terms of food are {self.current_profile['preferences'] if len(self.current_profile['preferences'])>0 else "nothing"}"))
                
        # self.user_profiles[new_profile['user_id']] = new_profile
        save_user_profile(self.current_profile['user_id'], self.current_profile)
        # self.current_profile = new_profile
        chat_history.append((message, f"Profile created! Welcome {self.current_profile['user_id']}!"))
        self.current_state = 'ready'
        return chat_history, ""
    
    def _run_post_clarification_chain(self, final_query: str, chat_history: List[Tuple[str, str]]): 
        print("Executing post-clarification chain... ")

        final_chain_input = {
            **self.pending_rag_data, 
            "reformulated_query" : final_query
        }

        response = self.rag_chain_part2.invoke(final_chain_input)
        meal_plan = response["context"][0][0]
        reasoning = response["context"][0][1]['reasoning']
        print('Debug: ', meal_plan, reasoning)
        response = pretty_format_recipes(meal_plan) + "\n\n" + reasoning


        print(Fore.RED + "Here is the response: " + response + Style.RESET_ALL)

        chat_history.append((None, response))
        self._update_system_state(self.pending_rag_data['question'], response)

        self.current_state = "feedback"
        self.pending_rag_data = None
        self.interaction_generator = None

        return chat_history, ""
    
    def _handle_ask_user_feedback(self, message: str, chat_history:List[Tuple[str, str]]):
        self.current_state == 'ready'
        return chat_history, ""



    def _handle_ready_state(self, message:str, chat_history:List[Tuple[str, str]]): 
        if message.lower() == 'exit' : 
            chat_history.append((message, "Session ended. Refresh to start new chat"))
            return chat_history, ""
        
        routing = QueryClassifier().router(message)

        print("Routed towards: ", routing)

        #chat_history.append((message, None)) # I would like a meal plan ...
        
        if routing["source"] == "vectorstore":
            # response = self.rag_chain.invoke({
            #                 "question": message, 
            #                 "user_id": self.current_profile["user_id"] # type: ignore
            #             })
            self.pending_rag_data = self.rag_chain_part1.invoke({
                "question" : message, 
                "user_id" : self.current_profile['user_id']
            })
            print('Debugging: ', self.pending_rag_data)
            self.interaction_generator = self.rag_preparator.query_check(
                self.pending_rag_data['question'],
                self.pending_rag_data['modified_user_data']
            )
            try: 
                first_question = next(self.interaction_generator)
                
                print(f"Generator yielded a question: ", {first_question})
                chat_history.append((None, first_question))
                # print(Fore.GREEN + "Just checking the format: " + str(chat_history) + Style.RESET_ALL)
                self.current_state = "clarifying_question"
                return chat_history, ""
            except StopIteration as e: 
                print("Generator finished without interaction. Running post-chain")
                final_query = e.value # get reformulated or original query

                chat_history, _ = self._run_post_clarification_chain(final_query, chat_history)

            # chat_history.append((message, response))
        else: 
            chatbot = SimpleChatBot()
            response = chatbot.chat(message)
            chat_history.append((message, response)) # type: ignore
            self._update_system_state(message, response) # type: ignore

        return chat_history, ""
    
    def _handle_clarifying_question_state(self, message: str, chat_history: List[Tuple[str, str]]):
        
        if not self.interaction_generator: 
            chat_history.append((None, "An error occured. Please start over."))
            self.current_state = "ready"
            return chat_history
        
        chat_history.append((message, None))
        
        try: 
            # send the user question to the generator
            next_question = self.interaction_generator.send(message)
            # if there is no error, then we have aa question. 
            chat_history.append((None, next_question))

            return chat_history, ""
        
        except StopIteration as e: 
            print("Clarification Finished")
            final_conversation = e.value

            # final_response = f"final RAG call based on the context: {final_conversation}"
            # chat_history.append((None, final_response))

            # self.interaction_generator = None
            # self.current_state = "ready"

            chat_history, _ = self._run_post_clarification_chain(final_conversation, chat_history)

        return chat_history, ""
    
    def ask_user_question(self, question:str):
        self.waiting_for_answer = True
        self.current_question = question

        self.answer_event = asyncio.Event()
        self.user_answer = None
        return self.user_answer



    def _update_system_state(self, query:str, response: str): 
        # self.foodchat.memory.save_context({'question':query}, {'answer': response})
        #update_user_profile_history(self.current_profile['user_id'], response) # type: ignore
        #write_to_file(response, response =True) #todo: rewrite user db with feedback
        return
    
    

def launch_interface(ui, args):
    # ui = UI(rag_chain, args)

    with gr.Blocks(title='FoodChat', js = pop_up) as interface: 
        chatbot = gr.Chatbot(label="Conversation", height = 600)
        msg = gr.Textbox(label="Your message", placeholder="Type here...", type="text")

        interface.load(
            fn=lambda: ui._handle_welcome_state([]), 
            inputs = None, 
            outputs = [chatbot]
        )

        msg.submit(
            ui.handle_message, 
            [msg, chatbot],
            [chatbot, msg]
        )
    
    interface.launch()