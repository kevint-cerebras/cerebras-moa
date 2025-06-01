#!/usr/bin/env python3
"""
Comprehensive Guide: Making LLM Outputs Maximally Deterministic

This guide explains why LLMs aren't perfectly deterministic and provides
practical strategies to achieve the highest level of consistency possible.
"""

import json
import random
import numpy as np
import os
from typing import Dict, Any, List
from moa.agent import MOAgent
from moa.agent.moa import MOAgentConfig

def make_deterministic_config() -> Dict[str, Any]:
    """
    Create the most deterministic LLM configuration possible.
    
    Returns:
        Configuration dictionary optimized for deterministic output
    """
    return {
        # === CORE DETERMINISTIC SETTINGS ===
        "temperature": 0.0,           # Greedy decoding (most deterministic)
        "top_p": 1.0,                # Consider full distribution (no nucleus sampling)
        "top_k": -1,                 # No top-k filtering
        "seed": 42,                  # Fixed seed (if supported by provider)
        "max_tokens": 2048,          # Fixed max tokens
        
        # === AVOID THESE FOR DETERMINISM ===
        # "frequency_penalty": 0.0,  # Don't use penalty parameters
        # "presence_penalty": 0.0,   # They can introduce variability
        # "n": 1,                    # Always request single output
        
        # === CEREBRAS-SPECIFIC OPTIMIZATIONS ===
        "stream": False,             # Disable streaming for consistency
    }

def set_deterministic_environment():
    """
    Set environment variables and random seeds for maximum reproducibility.
    
    Note: This only controls client-side randomness, not server-side model behavior.
    """
    # Set Python random seeds
    random.seed(42)
    np.random.seed(42)
    
    # Set environment variables for deterministic behavior
    os.environ["PYTHONHASHSEED"] = "0"
    
    # TensorFlow/PyTorch determinism (if using local models)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def create_deterministic_moa_agent() -> MOAgent:
    """
    Create an MOAgent configured for maximum determinism.
    
    Returns:
        MOAgent with deterministic settings
    """
    # Most deterministic configuration
    config = MOAgentConfig(
        main_model="llama-3.3-70b",  # Dense model (avoid MoE if possible)
        cycles=1,                    # Minimize cycles to reduce variability
        temperature=0.0,             # Greedy decoding
        max_tokens=2048,
        system_prompt="You are a helpful assistant that provides consistent, deterministic responses.",
        layer_agent_config={}       # No layer agents to reduce complexity
    )
    
    # Apply deterministic settings
    deterministic_kwargs = make_deterministic_config()
    
    return MOAgent(config, **deterministic_kwargs)

class DeterministicLLMManager:
    """
    Manager class for handling deterministic LLM interactions with validation.
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b"):
        self.model_name = model_name
        self.config = make_deterministic_config()
        self.agent = create_deterministic_moa_agent()
        
        # Track consistency
        self.call_history = []
        
    def generate_deterministic_response(self, prompt: str, validation_runs: int = 3) -> Dict[str, Any]:
        """
        Generate response with deterministic validation.
        
        Args:
            prompt: Input prompt
            validation_runs: Number of times to run for consistency check
            
        Returns:
            Dictionary with response and consistency metrics
        """
        responses = []
        
        for run in range(validation_runs):
            try:
                # Generate response with deterministic settings
                response = ""
                for chunk in self.agent.chat(prompt):
                    if isinstance(chunk, str):
                        response += chunk
                    elif hasattr(chunk, 'delta'):
                        response += chunk.delta
                
                responses.append(response.strip())
                
            except Exception as e:
                responses.append(f"ERROR: {str(e)}")
        
        # Analyze consistency
        unique_responses = list(set(responses))
        consistency_rate = responses.count(responses[0]) / len(responses) if responses else 0
        
        result = {
            "primary_response": responses[0] if responses else "",
            "all_responses": responses,
            "unique_responses": unique_responses,
            "consistency_rate": consistency_rate,
            "is_deterministic": len(unique_responses) == 1,
            "validation_runs": validation_runs
        }
        
        # Log for analysis
        self.call_history.append({
            "prompt": prompt,
            "result": result,
            "config": self.config.copy()
        })
        
        return result

def why_llms_arent_perfectly_deterministic():
    """
    Educational function explaining why even temperature=0 doesn't guarantee determinism.
    """
    print("🤖 Why LLMs Aren't Perfectly Deterministic (Even at Temperature=0)\n")
    
    reasons = [
        {
            "category": "🏗️ Architecture Issues",
            "problems": [
                "Mixture-of-Experts (MoE) routing: GPT-4 and similar models route tokens to different 'expert' networks",
                "Batch processing: Your request gets mixed with others, affecting expert routing",
                "Model parallelism: Large models split across multiple GPUs introduce synchronization issues"
            ]
        },
        {
            "category": "🔢 Hardware & Numerical Issues", 
            "problems": [
                "Floating-point precision: GPU calculations have tiny rounding differences",
                "Parallel computation: Non-deterministic order of operations on GPU threads",
                "Hardware differences: Different GPU models or configurations produce slightly different results"
            ]
        },
        {
            "category": "🌐 API & Infrastructure Issues",
            "problems": [
                "Server-side batching: Multiple users' requests processed together",
                "Load balancing: Requests routed to different servers/hardware",
                "Model updates: Backend changes that aren't user-visible"
            ]
        },
        {
            "category": "🎲 Edge Cases",
            "problems": [
                "Probability ties: When multiple tokens have nearly identical probabilities",
                "Seed implementation: Not all providers implement seeds consistently",
                "System fingerprint changes: OpenAI's term for when their backend system changes"
            ]
        }
    ]
    
    for reason in reasons:
        print(f"{reason['category']}:")
        for problem in reason['problems']:
            print(f"  • {problem}")
        print()

def best_practices_for_determinism():
    """
    Practical strategies for maximizing LLM determinism.
    """
    print("✅ Best Practices for Maximum LLM Determinism\n")
    
    strategies = [
        {
            "category": "🎯 Core Parameters",
            "tips": [
                "Always set temperature=0.0 (greedy decoding)",
                "Set top_p=1.0 (don't use nucleus sampling)",
                "Avoid top_k sampling (set to -1 or disable)",
                "Use seed parameter if available (e.g., OpenAI's seed)",
                "Request single outputs only (n=1)"
            ]
        },
        {
            "category": "🏛️ Model Selection",
            "tips": [
                "Prefer dense models over Mixture-of-Experts when possible",
                "Use smaller models for critical deterministic tasks",
                "Stick to the same model version/endpoint",
                "Avoid beta or preview models for production determinism"
            ]
        },
        {
            "category": "📝 Prompt Engineering",
            "tips": [
                "Use identical prompts (watch for hidden characters)",
                "Avoid asking for 'random' or 'creative' outputs",
                "Be specific rather than open-ended",
                "Use structured output formats (JSON, lists)",
                "Include explicit instructions for consistency"
            ]
        },
        {
            "category": "🔧 Implementation",
            "tips": [
                "Set client-side random seeds (Python, numpy)",
                "Cache responses when possible",
                "Implement retry logic with consistency checking",
                "Log all parameters and responses for debugging",
                "Use local models for critical deterministic needs"
            ]
        },
        {
            "category": "🏢 For This Codebase (Cerebras)",
            "tips": [
                "Use single-agent configs (avoid multi-layer MOA)",
                "Set cycles=1 to minimize variability",
                "Choose dense models like llama-3.3-70b over MoE models",
                "Disable streaming for consistency",
                "Implement validation runs to check consistency"
            ]
        }
    ]
    
    for strategy in strategies:
        print(f"{strategy['category']}:")
        for tip in strategy['tips']:
            print(f"  ✓ {tip}")
        print()

def test_determinism_levels():
    """
    Test different levels of determinism with various configurations.
    """
    print("🧪 Testing Determinism Levels\n")
    
    configs = [
        {
            "name": "Maximum Determinism",
            "config": {"temperature": 0.0, "top_p": 1.0, "seed": 42},
            "expected": "Highest consistency, but not 100% guaranteed"
        },
        {
            "name": "High Determinism", 
            "config": {"temperature": 0.1, "top_p": 0.9, "seed": 42},
            "expected": "Good consistency with slight creativity"
        },
        {
            "name": "Moderate Determinism",
            "config": {"temperature": 0.3, "top_p": 0.8, "seed": 42},
            "expected": "Balanced consistency and creativity"
        },
        {
            "name": "Low Determinism",
            "config": {"temperature": 0.7, "top_p": 0.7},
            "expected": "High creativity, low consistency"
        }
    ]
    
    for config_info in configs:
        print(f"📊 {config_info['name']}:")
        print(f"   Config: {config_info['config']}")
        print(f"   Expected: {config_info['expected']}")
        print()

def optimize_cerebras_for_determinism():
    """
    Specific optimizations for this Cerebras-based codebase.
    """
    print("🚀 Optimizing Cerebras MOA for Maximum Determinism\n")
    
    # Update the default configuration
    optimized_config = {
        "main_config": {
            "main_model": "llama-3.3-70b",  # Dense model, not MoE
            "cycles": 1,                     # Single cycle
            "temperature": 0.0,              # Greedy decoding
            "system_prompt": "You are a helpful assistant that provides consistent, deterministic responses.",
            "reference_system_prompt": "Synthesize the responses in a consistent, deterministic manner."
        },
        "layer_config": {}  # Disable layer agents for determinism
    }
    
    print("Recommended config.json updates:")
    print(json.dumps(optimized_config, indent=2))
    print()
    
    print("Code modifications for maximum determinism:")
    print("""
# In competitive_programming.py, modify agent creation:
def create_deterministic_agents(self) -> Dict[str, MOAgent]:
    agents = {}
    
    # Use single deterministic agent instead of multiple specialists
    deterministic_config = MOAgentConfig(
        main_model="llama-3.3-70b",
        cycles=1,
        temperature=0.0,
        system_prompt="You are a code analysis expert that provides consistent, deterministic feedback.",
        layer_agent_config={}
    )
    
    agents['deterministic_analyzer'] = MOAgent(
        deterministic_config, 
        temperature=0.0,
        top_p=1.0,
        seed=42
    )
    
    return agents
""")

if __name__ == "__main__":
    print("🎯 Comprehensive Guide: Making LLM Outputs Maximally Deterministic\n")
    
    # Set deterministic environment
    set_deterministic_environment()
    
    # Educational sections
    why_llms_arent_perfectly_deterministic()
    best_practices_for_determinism()
    test_determinism_levels()
    optimize_cerebras_for_determinism()
    
    # Practical example
    print("💡 Practical Example: Testing Determinism")
    manager = DeterministicLLMManager()
    
    test_prompt = "What is 2+2? Provide only the numeric answer."
    result = manager.generate_deterministic_response(test_prompt, validation_runs=3)
    
    print(f"Prompt: {test_prompt}")
    print(f"Consistency Rate: {result['consistency_rate']*100:.1f}%")
    print(f"Is Deterministic: {result['is_deterministic']}")
    print(f"Unique Responses: {len(result['unique_responses'])}")
    
    if not result['is_deterministic']:
        print("⚠️ Note: Even with optimal settings, perfect determinism isn't guaranteed!")
        print("This is due to the factors explained above.")
    else:
        print("✅ Achieved deterministic output!")
    
    print("\n🎯 Key Takeaway:")
    print("Perfect LLM determinism is theoretically impossible with current technology,")
    print("but you can achieve very high consistency (95%+) with proper configuration.")
    print("For critical applications requiring 100% determinism, consider rule-based")
    print("systems or local models with fixed environments.") 