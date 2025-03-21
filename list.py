import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

models = list(genai.list_models())  # Convert generator to list
for model in models:
    print(model.name)  # Print available model names

