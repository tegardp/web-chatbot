from flask import Flask, render_template, request
from bot.bot import Chatbot

import json
# Load Data
with open('bot/data/intents.json') as file:
    data = json.load(file)

app = Flask(__name__)

bot = Chatbot('Chatbot BPS')
bot.set_training(data)

@app.route("/")
def home():    
    return render_template("home.html") 
    
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')    
    return bot.get_response(userText)
if __name__ == "__main__":    
    app.run()