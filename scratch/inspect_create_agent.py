import inspect
from langchain.agents import create_agent

print("File:", inspect.getfile(create_agent))
print("Source:")
print(inspect.getsource(create_agent))
