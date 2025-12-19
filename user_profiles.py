import json
import re
from pathlib import Path

from colorama import Fore, Style

from agents import *

PATH = Path(__file__).parent.resolve()
USER_DB_PATH = PATH / 'user_profiles_db.json'

def get_user_profile(user_id : str) : 
    with open(USER_DB_PATH, "r", encoding='utf-8') as f:
        user_db = json.load(f)
    user = user_db[user_id]
    return user
    
def get_user_profiles() : 
    with open(USER_DB_PATH, "r", encoding="utf-8") as f: 
        user_db = json.load(f)
    db = {str(k) : v for k, v in user_db.items()}
    return db

def save_user_profile(user_id, user_data) :
    with open(USER_DB_PATH, "r", encoding="utf-8") as f:
        user_db = json.load(f)
    user_db[user_id] = user_data
    with open(USER_DB_PATH, "w", encoding = "utf-8") as f:
        json.dump(user_db, f, indent=4, ensure_ascii=False)
    f.close()

def format_user_profile(user_profile): 
    user_info = ["--- User Information ---"]
    if user_profile.get("diet"): 
        user_info.append(f"Dietary Requirements: {', '.join(user_profile['diet'])}")
    if user_profile.get("allergies"): 
        user_info.append(f"Allergies: {', '.join(user_profile['allergies'])}")
    if user_profile.get("preferences"): 
        user_info.append(f"Preferences: {', '.join(user_profile['preferences'])}")
    if user_profile.get("location") : 
        user_info.append(f"Location: {user_profile['location']}")
    if user_profile.get("history") : 
        user_info.append(f"History: {user_profile['history']}")
    user_info = "\n".join(user_info)
    return user_info

def get_user_input(prompt_message, required = True) : 

    while True: 
        value = input(f"{prompt_message}(separate items with ','): ").strip()
        if not value : 
            return []
        return [item.strip() for item in value.split(",")]
    
def create_profile(current_users): 
    print(Fore.CYAN + "Create Your Profile" + Style.RESET_ALL)

    while True : 
        
        username = get_user_input("Choose a unique username (this will be your User ID)", required = True)[0]
        user_id = re.sub(r'\s+', '_', username)

        if not user_id: 
            print("Username cannot be empty")
            continue
        if user_id in current_users.keys(): 
            print(f"Username: '{username}' is already taken. Please choose another.")
        else: 
            break #Unique ID processed
    
    print(Fore.CYAN + f"Your user id will be {user_id}" + Style.RESET_ALL)

    new_profile = {
        "user_id" : user_id, 
        "username" : username, 
        "diet" : get_user_input("Enter dietary requirements (e.g. Vegetarian, Vegan, ...)"), 
        "allergies" : get_user_input("Enter allergies (e.g. Peanuts, Shellfish, ...)"),
        "preferences" : get_user_input("Enter your preferences (e.g. Quick meals, Italian cuisine, ...)"),
        "location" : get_user_input("Enter location: (e.g Paris, France)"), 
        "history" : ""
    }

    print(Fore.CYAN + f'Dear {user_id}, your profile has been created!' + Style.RESET_ALL)

    return new_profile

def setup_user_session(): 

    user_profiles = get_user_profiles()
    print(Fore.LIGHTMAGENTA_EX + "WELCOME TO FOODCHAT" + Style.RESET_ALL)

    while True: 
        has_id = get_user_input("Do you have a user ID? (yes/no)", required = True)[0]

        if has_id == "yes" :
            user_id = get_user_input("Please enter your user ID: ", required=True)[0]

            if user_id in user_profiles: 
                print(f"Welcome Back {user_profiles[user_id].get('user_id')}!")
                return user_profiles[user_id]
            else: 
                print(f"User id '{user_id}' not found")
                is_new_profile = get_user_input("Would you like to create a new profile? (yes/no)")[0]
                if is_new_profile == "yes": 
                    new_profile = create_profile(user_profiles)
                    user_profiles[new_profile["user_id"]] = new_profile
                    save_user_profile(user_profiles)
                    return new_profile
                else :
                    print("Profile Creation Failure. Fallback to default user.")
                    return user_profiles["default_user"]
        elif has_id == "no" : 
            new_profile = create_profile(user_profiles)
            user_profiles[new_profile["user_id"]] = new_profile
            save_user_profile(user_profiles)
            return new_profile
        else : 
            print("Please answer with 'yes' or 'no'.")
            
def get_user_feedback() :
    
    while True : 
        feedback = input(f"Are you satisfied by this answer? Please answer by 'yes' or 'no': ")

        if feedback == "yes" : 
            print(Fore.LIGHTBLUE_EX + "Great! Wish you a happy meal!" + Style.RESET_ALL)
            return ""

        elif feedback == "no" : 
            explanation = input(f"We’re sorry to hear that! What didn’t work for you? Just a quick sentence helps us improve: ")
            return explanation

        else: 
            continue

def update_user_profile_history(user_id: str, llm_response : str): 

    user_profiles = get_user_profiles()
    user_profile = user_profiles[user_id]

    feedback = get_user_feedback()
    if feedback != "" : 
    # if key != "history" : 
    #     print(Fore.Red+ "USER PROFILE MODIFICATION FOR THIS ELEMENT HAS NOT BEEN IMPLEMENTED YET" + Style.RESET_ALL)
    #     return None
    
    # else : 

        current_data = user_profile["history"]
        feedback = current_data + " " + feedback + '.'
        print(Fore.LIGHTGREEN_EX + "RAW FEEDBACK: " + str(feedback) + Style.RESET_ALL)
        refined_feedback = FeedBackRewriter().rewrite(response=llm_response, feedback= feedback)
        user_profile["history"] = refined_feedback
        user_profiles[user_id] = user_profile

        save_user_profile(user_profiles)





