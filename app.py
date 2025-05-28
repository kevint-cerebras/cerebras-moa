import copy
import json
import os
from typing import Iterable, Dict, Any, Generator

import streamlit as st
from streamlit_ace import st_ace
from cerebras.cloud.sdk import Cerebras

from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk, MOAgentConfig
from moa.agent.prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

# Default configuration
default_main_agent_config = {
    "main_model": "llama-3.3-70b",
    "cycles": 3,
    "temperature": 0.1,
    "system_prompt": SYSTEM_PROMPT,
    "reference_system_prompt": REFERENCE_SYSTEM_PROMPT
}

default_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama-4-scout-17b-16e-instruct",
        "temperature": 0.3
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "qwen-3-32b",
        "temperature": 0.7
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "llama3.1-8b",
        "temperature": 0.1
    },
}

# Recommended Configuration
rec_main_agent_config = {
    "main_model": "llama-3.3-70b",
    "cycles": 2,
    "temperature": 0.1,
    "system_prompt": SYSTEM_PROMPT,
    "reference_system_prompt": REFERENCE_SYSTEM_PROMPT
}

rec_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama-4-scout-17b-16e-instruct",
        "temperature": 0.1
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "llama3.1-8b",
        "temperature": 0.2,
        "max_tokens": 2048
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "qwen-3-32b",
        "temperature": 0.4,
        "max_tokens": 2048
    },
    "layer_agent_4": {
        "system_prompt": "You are an expert planner agent. Create a plan for how to answer the human's query. {helper_response}",
        "model_name": "llama-3.3-70b",
        "temperature": 0.5
    },
}

# Helper functions
def json_to_moa_config(config_file) -> Dict[str, Any]:
    config = json.load(config_file)
    moa_config = MOAgentConfig( # To check if everything is ok
        **config
    ).model_dump(exclude_unset=True)
    return {
        'moa_layer_agent_config': moa_config.pop('layer_agent_config', None),
        'moa_main_agent_config': moa_config or None
    }

def stream_response(messages: Iterable[ResponseChunk]):
    layer_outputs = {}
    for message in messages:
        if message['response_type'] == 'intermediate':
            layer = message['metadata']['layer']
            agent = message['metadata'].get('agent', '')
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append((agent, message['delta']))
        else:
            # Display accumulated layer outputs
            for layer, outputs in layer_outputs.items():
                st.write(f"Layer {layer}")
                cols = st.columns(len(outputs))
                for i, (agent, output) in enumerate(outputs):
                    with cols[i]:
                        agent_label = f"Agent {i+1}" if not agent else f"{agent}"
                        st.expander(label=agent_label, expanded=False).write(output)
            
            # Clear layer outputs for the next iteration
            layer_outputs = {}
            
            # Yield the main agent's output
            yield message['delta']

def set_moa_agent(
    moa_main_agent_config = None,
    moa_layer_agent_config = None,
    override: bool = False
):
    moa_main_agent_config = copy.deepcopy(moa_main_agent_config or default_main_agent_config)
    moa_layer_agent_config = copy.deepcopy(moa_layer_agent_config or default_layer_agent_config)

    if "moa_main_agent_config" not in st.session_state or override:
        st.session_state.moa_main_agent_config = moa_main_agent_config

    if "moa_layer_agent_config" not in st.session_state or override:
        st.session_state.moa_layer_agent_config = moa_layer_agent_config

    if override or ("moa_agent" not in st.session_state):
        st_main_copy = copy.deepcopy(st.session_state.moa_main_agent_config)
        st_layer_copy = copy.deepcopy(st.session_state.moa_layer_agent_config)
        
        # Create the MOAgent with the new direct Cerebras implementation
        st.session_state.moa_agent = MOAgent.from_config(
            main_model=st_main_copy.pop('main_model'),
            system_prompt=st_main_copy.pop('system_prompt', SYSTEM_PROMPT),
            reference_system_prompt=st_main_copy.pop('reference_system_prompt', REFERENCE_SYSTEM_PROMPT),
            cycles=st_main_copy.pop('cycles', 1),
            temperature=st_main_copy.pop('temperature', 0.1),
            max_tokens=st_main_copy.pop('max_tokens', None),
            layer_agent_config=st_layer_copy,
            **st_main_copy
        )

        del st_main_copy
        del st_layer_copy

    del moa_main_agent_config
    del moa_layer_agent_config

# App
st.set_page_config(
    page_title="Mixture-Of-Agents Powered by Cerebras",
    page_icon='static/favicon.ico',
        menu_items={
        'About': "## Cerebras Mixture-Of-Agents \n Powered by [Cerebras](https://cerebras.net)"
    },
    layout="wide"
)

# For Cerebras, we use a predefined list of models
valid_model_names = ["llama-3.3-70b", "llama3.1-8b", "llama-4-scout-17b-16e-instruct", "qwen-3-32b"]

# Display banner directly using st.image with absolute path
import os
banner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "banner.png")
if os.path.exists(banner_path):
    st.image(banner_path, width=500)
    st.markdown("[Powered by Cerebras](https://cerebras.net)")
else:
    st.error(f"Banner image not found at {banner_path}")
    st.markdown("# Mixture-Of-Agents Powered by Cerebras")
    st.markdown("[Powered by Cerebras](https://cerebras.net)")

st.write("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

set_moa_agent()

# Sidebar for configuration
with st.sidebar:
    st.title("MOA Configuration")
    # upl_col, load_col = st.columns(2)
    st.download_button(
        "Download Current MoA Configuration as JSON", 
        data=json.dumps({
            **st.session_state.moa_main_agent_config,
            'moa_layer_agent_config': st.session_state.moa_layer_agent_config
        }, indent=2),
        file_name="moa_config.json"
    )

    # moa_config_upload = st.file_uploader("Choose a JSON file", type="json")
    # submit_config_file = st.button("Update config")
    # if moa_config_upload and submit_config_file:
    #     try:
    #         moa_config = json_to_moa_config(moa_config_upload)
    #         set_moa_agent(
    #             moa_main_agent_config=moa_config['moa_main_agent_config'],
    #             moa_layer_agent_config=moa_config['moa_layer_agent_config']
    #         )
    #         st.session_state.messages = []
    #         st.success("Configuration updated successfully!")
    #     except Exception as e:
    #         st.error(f"Error loading file: {str(e)}")

    with st.form("Agent Configuration", border=False):
        # Load and Save moa config file
             
        if st.form_submit_button("Use Recommended Config"):
            try:
                set_moa_agent(
                    moa_main_agent_config=rec_main_agent_config,
                    moa_layer_agent_config=rec_layer_agent_config,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

        # Main model selection
        main_model = st.session_state.moa_main_agent_config['main_model']
        # Ensure the model is in our valid_model_names list
        if main_model not in valid_model_names:
            # Default to first model if current one isn't in list
            default_index = 0
            st.warning(f"Model '{main_model}' not in valid models list. Defaulting to {valid_model_names[0]}")
        else:
            default_index = valid_model_names.index(main_model)
            
        new_main_model = st.selectbox(
            "Select Main Model",
            options=valid_model_names,
            index=default_index
        )



        # Cycles input
        new_cycles = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=10,
            value=st.session_state.moa_main_agent_config['cycles']
        )

        # Main Model Temperature
        main_temperature = st.number_input(
            label="Main Model Temperature",
            value=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.1
        )

        # Layer agent configuration
        tooltip = "Agents in the layer agent configuration run in parallel _per cycle_. Each layer agent supports all initialization parameters of [Cerebras' Cloud SDK](https://cloud.cerebras.ai) class as valid dictionary fields."
        st.markdown("Layer Agent Config", help=tooltip)
        new_layer_agent_config = st_ace(
            key="layer_agent_config",
            value=json.dumps(st.session_state.moa_layer_agent_config, indent=2),
            language='json',
            placeholder="Layer Agent Configuration (JSON)",
            show_gutter=False,
            wrap=True,
            auto_update=True
        )

        with st.expander("Optional Main Agent Params"):
            tooltip_str = """\
Main Agent configuration that will respond to the user based on the layer agent outputs.
Valid fields:
- ``system_prompt``: System prompt given to the main agent. \
**IMPORTANT**: it should always include a `{helper_response}` prompt variable.
- ``reference_prompt``: This prompt is used to concatenate and format each layer agent's output into one string. \
This is passed into the `{helper_response}` variable in the system prompt. \
**IMPORTANT**: it should always include a `{responses}` prompt variable. 
- ``main_model``: Which Cerebras powered model to use. Will overwrite the model given in the dropdown.\
"""
            tooltip = tooltip_str
            st.markdown("Main Agent Config", help=tooltip)
            new_main_agent_config = st_ace(
                key="main_agent_params",
                value=json.dumps(st.session_state.moa_main_agent_config, indent=2),
                language='json',
                placeholder="Main Agent Configuration (JSON)",
                show_gutter=False,
                wrap=True,
                auto_update=True
            )

        if st.form_submit_button("Update Configuration"):
            try:
                new_layer_config = json.loads(new_layer_agent_config)
                new_main_config = json.loads(new_main_agent_config)
                # Configure conflicting params
                # If param in optional dropdown == default param set, prefer using explicitly defined param
                if new_main_config.get('main_model', default_main_agent_config['main_model']) == default_main_agent_config["main_model"]:
                    new_main_config['main_model'] = new_main_model
                
                if new_main_config.get('cycles', default_main_agent_config['cycles']) == default_main_agent_config["cycles"]:
                    new_main_config['cycles'] = new_cycles

                if new_main_config.get('temperature', default_main_agent_config['temperature']) == default_main_agent_config['temperature']:
                    new_main_config['temperature'] = main_temperature
                
                set_moa_agent(
                    moa_main_agent_config=new_main_config,
                    moa_layer_agent_config=new_layer_config,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

    st.markdown("---")
    st.markdown("""
    ### Credits
    - MOA: [Together AI](https://www.together.ai/blog/together-moa)
    - LLMs: [Cerebras](https://cerebras.ai/)
    - Paper: [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
    """)

# Main app layout
st.header("Mixture of Agents", anchor=False)
st.write("A demo of the Mixture of Agents architecture proposed by Together AI, Powered by Cerebras LLMs.")

# Display current configuration
with st.status("Current MOA Configuration", expanded=True, state='complete') as config_status:
    # Use absolute path for the SVG image
    svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "moa.svg")
    if os.path.exists(svg_path):
        st.image(svg_path, caption="Mixture of Agents Workflow", use_container_width=True)
    else:
        st.error(f"SVG image not found at {svg_path}")
        st.markdown("### Mixture of Agents Workflow Diagram")
        st.markdown("*Image not available*")
    st.markdown(f"**Main Agent Config**:")
    new_layer_agent_config = st_ace(
        value=json.dumps(st.session_state.moa_main_agent_config, indent=2),
        language='json',
        placeholder="Layer Agent Configuration (JSON)",
        show_gutter=False,
        wrap=True,
        readonly=True,
        auto_update=True
    )
    st.markdown(f"**Layer Agents Config**:")
    new_layer_agent_config = st_ace(
        value=json.dumps(st.session_state.moa_layer_agent_config, indent=2),
        language='json',
        placeholder="Layer Agent Configuration (JSON)",
        show_gutter=False,
        wrap=True,
        readonly=True,
        auto_update=True
    )

if st.session_state.get("message", []) != []:
    st.session_state['expand_config'] = False
# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question"):
    try:
        # Add debug info
        debug_placeholder = st.empty()
        debug_placeholder.info("Processing your question...")
        
        config_status.update(expanded=False)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Debug info about MOA agent
        debug_placeholder.info(f"Using main model: {st.session_state.moa_agent.main_model}\nCycles: {st.session_state.moa_agent.cycles}")
        
        moa_agent: MOAgent = st.session_state.moa_agent
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                debug_placeholder.info("Calling MOAgent chat function...")
                ast_mess = stream_response(moa_agent.chat(query, output_format='json'))
                debug_placeholder.info("Got response stream, now writing...")
                response = st.write_stream(ast_mess)
                debug_placeholder.success("Response generated successfully!")
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                debug_placeholder.error(error_msg)
                st.error(error_msg)
                import traceback
                st.code(traceback.format_exc(), language="python")
                response = "I encountered an error while processing your request. Please check the error message above."
                message_placeholder.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")