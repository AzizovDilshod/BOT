import rasa_core
import rasa_nlu
import spacy
import sys
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_core.policies import FallbackPolicy, MemoizationPolicy,KerasPolicy
from rasa_core.agent import Agent
from rasa_core.agent import Agent
python = sys.executable


training_data = load_data("nlu.md")
trainer = Trainer(config.load("config.yml"))

interpreter = trainer.train(training_data) 
model_directory = trainer.persist("./models/nlu", fixed_model_name="current")


agent = Agent('domain.yml', policies=[MemoizationPolicy(), KerasPolicy(), ])
training_data = agent.load_data('stories.md')
agent.train(training_data)
agent.persist('models/dialogue')

agent = Agent.load('models/dialogue', interpreter=model_directory)

print("The bot is ready to talk! Say 'hello' to start or send 'stop' to end the conversation" + "\n\n" + "You:")
while True:
    a = input()
    if a == 'stop':
        break
    responses = agent.handle_message(a)
    for response in responses:
        print(response["text"])
