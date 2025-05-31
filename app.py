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

# Import competition system
try:
    from competition_ui import render_competition_page
    COMPETITION_AVAILABLE = True
    print("✅ Competition system imported successfully")
except ImportError as e:
    COMPETITION_AVAILABLE = False
    print(f"❌ Competition system import failed: {e}")
except Exception as e:
    COMPETITION_AVAILABLE = False
    print(f"❌ Competition system import error: {e}")

# App configuration - must be first
st.set_page_config(
    page_title="Mixture-Of-Agents Powered by Cerebras",
    page_icon='static/favicon.ico',
    menu_items={
        'About': "## Cerebras Mixture-Of-Agents \n Powered by [Cerebras](https://cerebras.net)"
    },
    layout="wide"
)

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

# For Cerebras, we use a predefined list of models
valid_model_names = ["llama-3.3-70b", "llama3.1-8b", "llama-4-scout-17b-16e-instruct", "qwen-3-32b"]

# Display banner directly using st.image with absolute path
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

def update_moa_config(update_main_only=False):
    """Auto-save the current configuration to the MOA agent
    
    Args:
        update_main_only: If True, only update the main agent config, not the layer agents
    """
    try:
        # Update the main config with the current number of layers
        new_main_config = copy.deepcopy(st.session_state.moa_main_agent_config)
        new_main_config['cycles'] = st.session_state.num_layers
        
        if update_main_only:
            # Only update the main agent config, not the layer agents
            set_moa_agent(
                moa_main_agent_config=new_main_config,
                override=False  # Don't override existing layer config
            )
        else:
            # Convert agent configs to layer agent config format
            new_layer_agent_config = {}
            for agent in st.session_state.agent_configs:
                new_layer_agent_config[agent["name"]] = {
                    "system_prompt": agent["system_prompt"],
                    "model_name": agent["model_name"],
                    "temperature": float(agent["temperature"]),  # Ensure proper type
                    "max_tokens": int(agent["max_tokens"])      # Ensure proper type
                }
            
            # Set the MOA agent with the new configuration
            set_moa_agent(
                moa_main_agent_config=new_main_config,
                moa_layer_agent_config=new_layer_agent_config,
                override=True
            )
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

def set_moa_agent(
    moa_main_agent_config = None,
    moa_layer_agent_config = None,
    override: bool = False
):
    moa_main_agent_config = copy.deepcopy(moa_main_agent_config or default_main_agent_config)
    moa_layer_agent_config = copy.deepcopy(moa_layer_agent_config or default_layer_agent_config)

    # Only update the main agent config if explicitly provided or not initialized
    if moa_main_agent_config is not None and ("moa_main_agent_config" not in st.session_state or override):
        st.session_state.moa_main_agent_config = moa_main_agent_config

    # Only update the layer agent config if explicitly provided or not initialized
    if moa_layer_agent_config is not None and ("moa_layer_agent_config" not in st.session_state or override):
        st.session_state.moa_layer_agent_config = moa_layer_agent_config
        
        # When layer agent config is updated, also update the agent_configs list to stay in sync
        if "agent_configs" in st.session_state:
            # Only update agent_configs if we're explicitly overriding with a new layer config
            if override:
                # Create a new agent_configs list from the provided layer_agent_config
                st.session_state.agent_configs = []
                for agent_name, agent_config in moa_layer_agent_config.items():
                    st.session_state.agent_configs.append({
                        "name": agent_name,
                        "system_prompt": agent_config.get("system_prompt", "Think through your response step by step. {helper_response}"),
                        "model_name": agent_config.get("model_name", "llama3.1-8b"),
                        "temperature": float(agent_config.get("temperature", 0.7)),
                        "max_tokens": int(agent_config.get("max_tokens", 2048))
                    })

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
        
        # Print debug info
        print("=== set_moa_agent called ===")
        print(f"num_layers: {st.session_state.moa_main_agent_config.get('cycles')}")
        print(f"Successfully created new MOAgent instance")

        del st_main_copy
        del st_layer_copy

    del moa_main_agent_config
    del moa_layer_agent_config

# Initialize MOA system first
set_moa_agent()

def render_moa_chat_page():
    """Render the MOA chat interface"""
    # Sidebar for configuration
    with st.sidebar:
        st.title("MOA Configuration")
        st.download_button(
            "Download Current MoA Configuration as JSON", 
            data=json.dumps({
                **st.session_state.moa_main_agent_config,
                'moa_layer_agent_config': st.session_state.moa_layer_agent_config
            }, indent=2),
            file_name="moa_config.json"
        )

        # Agent management section with improved UI
        st.markdown("## Agent Management")
        
        # Initialize config in session state if not present
        if "agent_configs" not in st.session_state:
            # Convert the existing config to the new format
            st.session_state.agent_configs = []
            for agent_name, agent_config in st.session_state.moa_layer_agent_config.items():
                st.session_state.agent_configs.append({
                    "name": agent_name,
                    "system_prompt": agent_config.get("system_prompt", "Think through your response step by step. {helper_response}"),
                    "model_name": agent_config.get("model_name", "llama3.1-8b"),
                    "temperature": agent_config.get("temperature", 0.7),
                    "max_tokens": agent_config.get("max_tokens", 2048)
                })
        
        # Initialize tracking variables in session state
        if "agent_to_remove" not in st.session_state:
            st.session_state.agent_to_remove = None
        
        if "num_layers" not in st.session_state:
            st.session_state.num_layers = st.session_state.moa_main_agent_config.get('cycles', 1)
        
        # Create a more intuitive layout with tabs for different sections
        tab1, tab2 = st.tabs(["📊 Configuration", "🧠 Agents"])
        
        with tab1:
            # Main configuration controls
            st.markdown("### Main Configuration")
            
            # Create a clean layout with columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Number of layers input (separate from number of agents)
                num_layers = st.number_input(
                    "Number of Layers",
                    min_value=1,
                    max_value=20,  # Increased max value
                    value=st.session_state.num_layers,
                    help="The number of processing layers in the MOA architecture. Can exceed the number of agents."
                )
                if num_layers != st.session_state.num_layers:
                    st.session_state.num_layers = num_layers
                    # Auto-save the configuration, but only update the main config
                    update_moa_config(update_main_only=True)
            
            with col2:
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
                    "Main Model",
                    options=valid_model_names,
                    index=default_index,
                    help="The model used for the final response generation"
                )
                
                # Auto-save if main model changes
                if new_main_model != main_model:
                    new_main_config = copy.deepcopy(st.session_state.moa_main_agent_config)
                    new_main_config['main_model'] = new_main_model
                    # Only update the main config, not the layer agents
                    set_moa_agent(
                        moa_main_agent_config=new_main_config,
                        override=False  # Don't override existing layer config
                    )
            
            # Main Model Temperature with a more intuitive slider
            main_temp = st.session_state.moa_main_agent_config.get('temperature', 0.1)
            new_main_temp = st.slider(
                label="Main Model Temperature",
                min_value=0.0,
                max_value=1.0,
                value=main_temp,
                step=0.1,
                help="Controls randomness in the main model's output. Higher values make output more random."
            )
            
            # Auto-save if temperature changes
            if new_main_temp != main_temp:
                new_main_config = copy.deepcopy(st.session_state.moa_main_agent_config)
                new_main_config['temperature'] = float(new_main_temp)  # Ensure proper type
                # Only update the main config, not the layer agents
                set_moa_agent(
                    moa_main_agent_config=new_main_config,
                    override=False  # Don't override existing layer config
                )
            
            # Quick configuration options
            st.markdown("### Quick Configuration Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Use Recommended Config", use_container_width=True):
                    try:
                        # Reset agent configs
                        st.session_state.agent_configs = []
                        for agent_name, agent_config in rec_layer_agent_config.items():
                            st.session_state.agent_configs.append({
                                "name": agent_name,
                                "system_prompt": agent_config.get("system_prompt", "Think through your response step by step. {helper_response}"),
                                "model_name": agent_config.get("model_name", "llama3.1-8b"),
                                "temperature": agent_config.get("temperature", 0.7),
                                "max_tokens": agent_config.get("max_tokens", 2048)
                            })
                        
                        # Set number of layers
                        st.session_state.num_layers = rec_main_agent_config.get('cycles', 2)
                        
                        # Update the configuration
                        set_moa_agent(
                            moa_main_agent_config=rec_main_agent_config,
                            moa_layer_agent_config=rec_layer_agent_config,
                            override=True
                        )
                        st.session_state.messages = []
                        st.success("Configuration updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating configuration: {str(e)}")
            
            with col2:
                if st.button("Reset to Default Config", use_container_width=True):
                    try:
                        # Reset agent configs
                        st.session_state.agent_configs = []
                        for agent_name, agent_config in default_layer_agent_config.items():
                            st.session_state.agent_configs.append({
                                "name": agent_name,
                                "system_prompt": agent_config.get("system_prompt", "Think through your response step by step. {helper_response}"),
                                "model_name": agent_config.get("model_name", "llama3.1-8b"),
                                "temperature": agent_config.get("temperature", 0.7),
                                "max_tokens": agent_config.get("max_tokens", 2048)
                            })
                        
                        # Set number of layers
                        st.session_state.num_layers = default_main_agent_config.get('cycles', 3)
                        
                        # Update the configuration
                        set_moa_agent(
                            moa_main_agent_config=default_main_agent_config,
                            moa_layer_agent_config=default_layer_agent_config,
                            override=True
                        )
                        st.session_state.messages = []
                        st.success("Reset to default configuration!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error resetting configuration: {str(e)}")
        
        with tab2:
            # Agent management section
            st.markdown("### Agent Management")
            
            # Display current agents count and add button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Current Agents: {len(st.session_state.agent_configs)} | Active Layers: {st.session_state.num_layers}**")
            with col2:
                if st.button("+ Add Agent", key="add_agent_btn", use_container_width=True):
                    # Create a unique agent ID to avoid duplicates
                    agent_id = len(st.session_state.agent_configs) + 1
                    agent_name = f"agent_{agent_id}"
                    
                    # Check if an agent with this name already exists
                    existing_names = [agent["name"] for agent in st.session_state.agent_configs]
                    while agent_name in existing_names:
                        agent_id += 1
                        agent_name = f"agent_{agent_id}"
                    
                    # Add the new agent with a unique name
                    st.session_state.agent_configs.append({
                        "name": agent_name,
                        "system_prompt": "Think through your response step by step. {helper_response}",
                        "model_name": "llama3.1-8b",
                        "temperature": 0.7,
                        "max_tokens": 2048
                    })
                    
                    # Auto-save the configuration
                    if update_moa_config():
                        st.success(f"Added new agent: {agent_name}")
                        st.rerun()
            
            # Process agent removal if needed
            if st.session_state.agent_to_remove is not None:
                if 0 <= st.session_state.agent_to_remove < len(st.session_state.agent_configs):
                    st.session_state.agent_configs.pop(st.session_state.agent_to_remove)
                    # Auto-save the configuration
                    update_moa_config()
                st.session_state.agent_to_remove = None
                st.rerun()
            
            # Display current agents with their settings
            if not st.session_state.agent_configs:
                st.info("No agents configured. Add an agent below to get started.")
            else:
                for i, agent in enumerate(st.session_state.agent_configs):
                    with st.expander(f"🤖 {agent['name']}", expanded=False):
                        # Agent name input
                        agent_name = st.text_input(
                            "Agent Name",
                            value=agent["name"],
                            key=f"tab1_agent_name_{i}",
                            help="A descriptive name for this agent"
                        )
                        if agent_name != agent["name"]:
                            agent["name"] = agent_name
                            # Auto-save the configuration
                            update_moa_config()
                        
                        # System prompt with a more intuitive editor
                        system_prompt = st.text_area(
                            "System Prompt",
                            value=agent["system_prompt"],
                            key=f"tab1_system_prompt_{i}",
                            height=100,
                            help="The system prompt for this agent. MUST include {helper_response} placeholder."
                        )
                        if system_prompt != agent["system_prompt"]:
                            agent["system_prompt"] = system_prompt
                            # Auto-save the configuration
                            update_moa_config()
                        
                        # Model and settings in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Model selection
                            current_model = agent["model_name"]
                            if current_model not in valid_model_names:
                                default_model_index = 0
                                st.warning(f"Model '{current_model}' not in valid models list.")
                            else:
                                default_model_index = valid_model_names.index(current_model)
                                
                            new_model = st.selectbox(
                                "Model",
                                options=valid_model_names,
                                index=default_model_index,
                                key=f"tab1_model_{i}",
                                help="The LLM model to use for this agent"
                            )
                            if new_model != current_model:
                                agent["model_name"] = new_model
                                # Auto-save the configuration
                                update_moa_config()
                        
                        with col2:
                            # Temperature slider
                            new_temp = st.slider(
                                "Temperature",
                                min_value=0.0,
                                max_value=1.0,
                                value=float(agent["temperature"]),
                                step=0.1,
                                key=f"tab1_temperature_{i}",
                                help="Controls randomness. Higher = more creative, Lower = more deterministic"
                            )
                            if new_temp != agent["temperature"]:
                                agent["temperature"] = new_temp
                                # Auto-save the configuration
                                update_moa_config()
                        
                        # Max tokens input
                        new_tokens = st.number_input(
                            "Max Tokens",
                            min_value=100,
                            max_value=8192,
                            value=int(agent["max_tokens"]),
                            step=100,
                            key=f"tab1_max_tokens_{i}",
                            help="Maximum number of tokens this agent can generate"
                        )
                        if new_tokens != agent["max_tokens"]:
                            agent["max_tokens"] = int(new_tokens)  # Ensure max_tokens is an integer
                            
                            # Auto-save the configuration
                            update_moa_config()
                        
                        # Remove button at the bottom of the expander
                        if st.button("Remove Agent", key=f"tab1_remove_{i}", use_container_width=True):
                            st.session_state.agent_to_remove = i
                            st.rerun()
        
        # Advanced configuration options
        with st.expander("Advanced Configuration Options", expanded=False):
            # Optional main agent parameters editor
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
            st.markdown("### Advanced Main Agent Config", help=tooltip_str)
            new_main_agent_config = st_ace(
                key="tab1_main_agent_params",
                value=json.dumps(st.session_state.moa_main_agent_config, indent=2),
                language='json',
                placeholder="Main Agent Configuration (JSON)",
                show_gutter=False,
                wrap=True,
                auto_update=True,
                height=200
            )
            
            # Auto-save if JSON is valid
            try:
                new_config = json.loads(new_main_agent_config)
                if new_config != st.session_state.moa_main_agent_config:
                    # Preserve the number of layers
                    new_config['cycles'] = st.session_state.num_layers
                    set_moa_agent(
                        moa_main_agent_config=new_config,
                        override=True
                    )
            except json.JSONDecodeError:
                # Don't auto-save if JSON is invalid
                pass
            
            # Apply JSON configuration button
            if st.button("Apply JSON Configuration", key="tab1_apply_json", use_container_width=True):
                try:
                    # Get the main agent config from the JSON editor
                    new_main_config = json.loads(new_main_agent_config)
                    
                    # Always preserve the number of layers
                    new_main_config['cycles'] = st.session_state.num_layers
                    
                    # Set the MOA agent with the new configuration
                    set_moa_agent(
                        moa_main_agent_config=new_main_config,
                        override=True
                    )
                    st.success("Configuration updated successfully!")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON in Main Agent Configuration. Please check your input.")
                except Exception as e:
                    st.error(f"Error updating configuration: {str(e)}")

    # Credits section
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

    # Chat interface - removed the big configuration display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question"):
        try:
            # Add debug info
            debug_placeholder = st.empty()
            debug_placeholder.info("Processing your question...")
            
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

def main():
    """Main function to control page navigation"""
    # Page selection in sidebar
    st.sidebar.title("Navigation")
    if COMPETITION_AVAILABLE:
        page = st.sidebar.selectbox(
            "Choose a page",
            ["🤖 MOA Chat", "🏆 Code Competition"],
            index=0
        )
    else:
        page = "🤖 MOA Chat"
        st.sidebar.info("Competition system not available")

    # Main content area based on page selection
    if page == "🏆 Code Competition" and COMPETITION_AVAILABLE:
        # Render competition page ONLY - no MOA chat interface
        render_competition_page()
    else:
        # Render MOA Chat page with sidebar configuration
        render_moa_chat_page()

# Run the main function
main()