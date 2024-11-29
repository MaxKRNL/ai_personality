# main.py
from my_bot import MyBot

bot = MyBot()

user_input = "Hello, how are you?"
response = bot.generate_response(user_input)
print("Bot:", response)
