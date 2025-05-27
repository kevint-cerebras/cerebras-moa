from agent import MOAgent

# Configure agent
layer_agent_config = {
    'layer_agent_1' : {'system_prompt': "Think through your response with step by step {helper_response}", 'model_name': 'llama-3.3-70b'},
    'layer_agent_2' : {'system_prompt': "Respond with a thought and then your response to the question {helper_response}", 'model_name': 'qwen-3-32b'},
    'layer_agent_3' : {'model_name': 'llama3.1-8b'},
    'layer_agent_4' : {'model_name': 'llama-4-scout-17b-16e-instruct'},
    'layer_agent_5' : {'model_name': 'llama-3.3-70b'},
}
agent = MOAgent.from_config(
    main_model='llama-3.3-70b',
    layer_agent_config=layer_agent_config
)

while True:
    inp = input("\nAsk a question: ")
    stream = agent.chat(inp, output_format='json')
    for chunk in stream:
        print(chunk)