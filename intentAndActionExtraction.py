import openai
from APIs import MY_API_KEY
import ast

openai.api_key = MY_API_KEY

caption = "Three stuffed bears hugging and sitting on a blue pillow"


def getIntent(caption):
    intentPrompt = f"Analyze the given sentence '{caption}' and determine the subject and object in it. Important NOTE:- Objects must not inlcude verbs or actions. Your response must be a list of subjects and objects in the sentence"
    intent = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=intentPrompt,
        temperature=0.1,
        max_tokens=30,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    intent = intent["choices"][0]["text"]
    intent = intent.replace('\n', '')
    #print(intent)
    return intent

def getActions(caption):

    actionPrompt = f"Extract actions being performed in the following sentence: '{caption}'. Your response must be an array of single word 'actions' only"

    actions = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=actionPrompt,
        temperature=0.0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    actions = actions["choices"][0]["text"]
    try:
        actions = actions.replace('\n', '')
        action_list = ast.literal_eval(actions)
    except:
        return []

    #print(actions)
    return action_list

# print(getActions("Three stuffed bears hugging and sitting on a blue pillow"))
# print(getActions("A teddy bear is dancing"))
