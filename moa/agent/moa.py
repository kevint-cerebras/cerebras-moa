"""
MOAgent implementation using direct Cerebras SDK
"""
import os
import time
import json
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from cerebras.cloud.sdk import Cerebras

from .prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT



class MOAgentConfig(BaseModel):
    main_model: Optional[str] = None
    system_prompt: Optional[str] = None
    cycles: int = Field(...)
    layer_agent_config: Optional[Dict[str, Any]] = None
    reference_system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    class Config:
        extra = "allow"  # This allows for additional fields not explicitly defined

load_dotenv()
valid_model_names = Literal[
    'llama3.3-70b',
    'llama3.1-8b',
    'llama-4-scout-17b-16e-instruct',
    'qwen-3-32b'
]

class Message(BaseModel):
    role: str
    content: str

class ResponseChunk(TypedDict):
    delta: str
    response_type: Literal['intermediate', 'output']
    metadata: Dict[str, Any]
    
class ConversationMemory:
    def __init__(self):
        self.messages = []
        
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
    def get_messages(self):
        return self.messages
    
    def clear(self):
        self.messages = []


class MOAgent:
    def __init__(
        self,
        main_model: str,
        layer_agent_config: Dict[str, Dict[str, Any]],
        reference_system_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        cycles: Optional[int] = None,
        temperature: Optional[float] = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> None:
        self.client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
        self.reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.main_model = main_model
        self.layer_agent_config = layer_agent_config
        self.cycles = cycles or 1
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.memory = ConversationMemory()

    def concat_responses(
        self,
        responses: Dict[str, str]
    ) -> Dict[str, Union[str, List[str]]]:
        """Format the responses from layer agents into a single prompt."""
        responses_text = ""
        res_list = []
        for i, out in enumerate(responses.values()):
            responses_text += f"{i}. {out}\n"
            res_list.append(out)

        formatted_prompt = self.reference_system_prompt.format(responses=responses_text)
        return {
            'formatted_response': formatted_prompt,
            'responses': res_list
        }

    @classmethod
    def from_config(
        cls,
        main_model: Optional[valid_model_names] = 'llama3.3-70b',
        system_prompt: Optional[str] = None,
        cycles: int = 1,
        layer_agent_config: Optional[Dict] = None,
        reference_system_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.1,
        max_tokens: Optional[int] = None,
        **main_model_kwargs
    ):
        """Create a MOAgent from configuration parameters."""
        reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        system_prompt = system_prompt or SYSTEM_PROMPT
        
        if not layer_agent_config:
            layer_agent_config = {
                'layer_agent_1': {
                    'system_prompt': 'Think through your response step by step. {helper_response}',
                    'model_name': 'llama3.1-8b',
                    'temperature': 0.3
                },
                'layer_agent_2': {
                    'system_prompt': 'Respond with a thought and then your response to the question. {helper_response}',
                    'model_name': 'llama-4-scout-17b-16e-instruct',
                    'temperature': 0.7
                },
                'layer_agent_3': {
                    'system_prompt': 'You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}',
                    'model_name': 'llama3.1-8b',
                    'temperature': 0.1
                }
            }
            
        return cls(
            main_model=main_model,
            layer_agent_config=layer_agent_config,
            reference_system_prompt=reference_system_prompt,
            system_prompt=system_prompt,
            cycles=cycles,
            temperature=temperature,
            max_tokens=max_tokens,
            **main_model_kwargs
        )

    def _generate_completion(
        self, 
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Generate a completion from the Cerebras API."""
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Debug info
        print(f"\n===== CEREBRAS API CALL =====")
        print(f"Model: {model}")
        print(f"Temperature: {temperature}")
        print(f"Max Tokens: {max_tokens}")
        print(f"Stream: {stream}")
        print(f"Messages: {json.dumps(messages, indent=2)}")
        
        # Combine kwargs with any instance defaults
        all_kwargs = {**self.kwargs}
        if kwargs:
            all_kwargs.update(kwargs)
            
        # Add temperature and max_tokens if specified
        # Ensure temperature is a float and max_tokens is an integer
        if temperature is not None:
            all_kwargs['temperature'] = float(temperature)
        if max_tokens is not None:
            all_kwargs['max_tokens'] = int(max_tokens)
            
        print(f"Additional kwargs: {all_kwargs}")
        
        try:
            print("Making API call to Cerebras...")
            if stream:
                print("Streaming mode enabled")
                response_stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    **all_kwargs
                )
                
                # Return a generator that yields chunks
                def response_generator():
                    print("Starting response stream")
                    try:
                        for chunk in response_stream:
                            print(f"Got chunk: {chunk}")
                            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                print(f"Yielding content: {content}")
                                yield content
                    except Exception as e:
                        print(f"Error in stream processing: {e}")
                        yield f"[Error in stream: {str(e)}]"
                
                return response_generator()
            else:
                print("Non-streaming mode")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **all_kwargs
                )
                print(f"Got response: {response}")
                return response.choices[0].message.content
                
        except Exception as e:
            error_str = str(e)
            error_msg = f"Error calling Cerebras API: {error_str}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            if stream:
                def error_generator():
                    yield f"[Error: {error_str}]"
                return error_generator()
            else:
                return f"[Error: {error_str}]"

    def _create_chat_messages(
        self,
        system_prompt: str,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
        helper_response: str = ""
    ) -> List[Dict[str, str]]:
        """Create formatted messages for the chat completion API."""
        # Apply the helper_response to the system prompt if present
        formatted_system_prompt = system_prompt.format(helper_response=helper_response)
        
        messages = [
            {"role": "system", "content": formatted_system_prompt}
        ]
        
        # Add conversation history if available
        if history and len(history) > 0:
            messages.extend(history)
            
        # Add the current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages

    def chat(
        self, 
        input: str,
        messages: Optional[List[Dict[str, str]]] = None,
        cycles: Optional[int] = None,
        save: bool = True,
        output_format: Literal['string', 'json'] = 'string'
    ) -> Generator[Union[str, ResponseChunk], None, None]:
        """
        Generate a response using the MOA architecture.
        
        Args:
            input: User input query
            messages: Optional message history (will use internal memory if not provided)
            cycles: Number of cycle iterations to run
            save: Whether to save the conversation in memory
            output_format: Format of the output ('string' or 'json')
            
        Yields:
            Response chunks from the model(s)
        """
        cycles = cycles or self.cycles
        history = messages or self.memory.get_messages()
        helper_response = ""
        
        # Use the requested number of cycles (layers) regardless of how many agents are configured
        num_layers = cycles or self.cycles
        
        # Run through each cycle
        for cyc in range(num_layers):
            layer_responses = {}
            
            # If we have more layers than agents, we'll reuse agents in a round-robin fashion
            # Convert the layer_agent_config dictionary items to a list for indexing
            layer_agents = list(self.layer_agent_config.items())
            num_agents = len(layer_agents)
            
            # Run each layer agent in parallel (in a real production system, 
            # we'd use threading or asyncio here for true parallelism)
            for i, (layer_name, layer_config) in enumerate(layer_agents):
                # If we're in a layer beyond the number of agents, create a unique name for logging
                if cyc >= num_agents:
                    layer_name = f"{layer_name}_layer{cyc}"
                system_prompt = layer_config.get('system_prompt', self.system_prompt)
                model_name = layer_config.get('model_name', self.main_model)
                temperature = layer_config.get('temperature', self.temperature)
                max_tokens = layer_config.get('max_tokens', self.max_tokens)
                
                # Create messages for this layer
                layer_messages = self._create_chat_messages(
                    system_prompt=system_prompt,
                    user_input=input,
                    history=history,
                    helper_response=helper_response
                )
                
                # Get completion from this layer
                layer_response = self._generate_completion(
                    model=model_name,
                    messages=layer_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Store the layer's response
                layer_responses[layer_name] = layer_response
                
                # Yield the layer's response if in JSON format
                if output_format == 'json':
                    yield ResponseChunk(
                        delta=layer_response,
                        response_type='intermediate',
                        metadata={'layer': cyc + 1, 'agent': layer_name}
                    )
            
            # Concat the layer responses
            formatted_output = self.concat_responses(layer_responses)
            helper_response = formatted_output['formatted_response']
        
        # Create messages for the main agent
        main_messages = self._create_chat_messages(
            system_prompt=self.system_prompt,
            user_input=input,
            history=history,
            helper_response=helper_response
        )
        
        # Get streaming completion from the main agent
        stream = self._generate_completion(
            model=self.main_model,
            messages=main_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        
        # Stream the response
        full_response = ""
        for chunk in stream:
            if output_format == 'json':
                yield ResponseChunk(
                    delta=chunk,
                    response_type='output',
                    metadata={}
                )
            else:
                yield chunk
            full_response += chunk
        
        # Save the conversation if requested
        if save:
            self.memory.add_message("user", input)
            self.memory.add_message("assistant", full_response)