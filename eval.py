import pickle

with open("eps_agent_classifier.pkl", "rb") as f:
    model = pickle.load(f)

pred = model.predict(["please start eps agent for customer xyz"])
print(pred)

pred = model.predict(["what is api"])
print(pred)
