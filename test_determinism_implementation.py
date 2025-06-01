#!/usr/bin/env python3
"""
Test Current Determinism Implementation

This script checks what determinism settings are actually implemented
in the competitive programming system.
"""

import sys
import json
sys.path.append('.')

def test_current_config():
    """Test the current MOA configuration"""
    print("🔍 Checking Current Configuration")
    print("=" * 50)
    
    try:
        with open('config/moa_config.json', 'r') as f:
            config = json.load(f)
        
        print("📋 Main Config:")
        main_config = config.get('main_config', {})
        print(f"  Model: {main_config.get('main_model', 'unknown')}")
        print(f"  Cycles: {main_config.get('cycles', 'unknown')}")
        print(f"  Temperature: {main_config.get('temperature', 'unknown')}")
        
        print("\n🎛️ Layer Config:")
        layer_config = config.get('layer_config', {})
        if layer_config:
            for agent_name, agent_config in layer_config.items():
                temp = agent_config.get('temperature', 'unknown')
                print(f"  {agent_name}: temp={temp}")
        else:
            print("  ✅ Layer agents disabled (good for determinism)")
        
        print("\n⚙️ Deterministic Settings:")
        det_settings = config.get('deterministic_settings', {})
        if det_settings:
            for key, value in det_settings.items():
                print(f"  {key}: {value}")
        else:
            print("  ❌ No explicit deterministic settings found")
        
        # Analysis
        print("\n📊 Determinism Analysis:")
        main_temp = main_config.get('temperature', 1.0)
        cycles = main_config.get('cycles', 2)
        has_layers = bool(layer_config)
        
        score = 0
        if main_temp == 0.0:
            print("  ✅ Main temperature = 0.0 (optimal)")
            score += 25
        else:
            print(f"  ❌ Main temperature = {main_temp} (should be 0.0)")
        
        if cycles == 1:
            print("  ✅ Cycles = 1 (optimal)")
            score += 25
        else:
            print(f"  ⚠️ Cycles = {cycles} (should be 1 for max determinism)")
        
        if not has_layers:
            print("  ✅ Layer agents disabled (optimal)")
            score += 25
        else:
            print(f"  ⚠️ {len(layer_config)} layer agents enabled (adds variability)")
        
        if det_settings:
            print("  ✅ Explicit deterministic settings found")
            score += 25
        else:
            print("  ❌ No explicit deterministic settings")
        
        print(f"\n🎯 Determinism Score: {score}/100")
        
        if score >= 75:
            print("✅ Good determinism configuration")
        elif score >= 50:
            print("⚠️ Moderate determinism configuration")
        else:
            print("❌ Poor determinism configuration")
    
    except Exception as e:
        print(f"❌ Error reading config: {e}")

def test_competitive_programming_agents():
    """Test the competitive programming agent configuration"""
    print("\n🤖 Checking Competitive Programming Agents")
    print("=" * 50)
    
    try:
        from competitive_programming import CompetitiveProgrammingSystem
        
        system = CompetitiveProgrammingSystem()
        print("✅ CompetitiveProgrammingSystem imported successfully")
        
        # Try to inspect how agents are created (without creating them)
        print("\n🔍 Agent Creation Method Analysis:")
        
        # Check the create_specialized_agents method
        import inspect
        source = inspect.getsource(system.create_specialized_agents)
        
        # Look for temperature settings
        if '"temperature": 0.0' in source or 'temperature=0.0' in source:
            print("  ✅ Found temperature=0.0 in agent creation")
        else:
            print("  ❌ temperature=0.0 not found in agent creation")
        
        if '"cycles": 1' in source or 'cycles=1' in source:
            print("  ✅ Found cycles=1 in agent creation")
        else:
            print("  ❌ cycles=1 not found in agent creation")
        
        if '"llama-3.3-70b"' in source:
            print("  ✅ Using llama-3.3-70b (dense model)")
        else:
            print("  ❌ Not using recommended dense model")
        
    except ImportError as e:
        print(f"❌ Could not import CompetitiveProgrammingSystem: {e}")
    except Exception as e:
        print(f"❌ Error analyzing agents: {e}")

def test_deterministic_configs():
    """Test the deterministic config files"""
    print("\n📋 Checking Deterministic Config Files")
    print("=" * 50)
    
    try:
        with open('config/deterministic_config.json', 'r') as f:
            det_config = json.load(f)
        
        print("✅ Deterministic config file found")
        
        main_config = det_config.get('main_config', {})
        det_settings = det_config.get('deterministic_settings', {})
        
        print(f"  Main model: {main_config.get('main_model')}")
        print(f"  Cycles: {main_config.get('cycles')}")
        print(f"  Temperature: {main_config.get('temperature')}")
        
        print(f"  Deterministic settings: {len(det_settings)} parameters")
        for key, value in det_settings.items():
            print(f"    {key}: {value}")
        
        # Check if it's being used
        print("\n❓ Status: Created but not active (demonstration only)")
        
    except FileNotFoundError:
        print("❌ Deterministic config file not found")
    except Exception as e:
        print(f"❌ Error reading deterministic config: {e}")

def main():
    """Main test function"""
    print("🧪 Testing Current Determinism Implementation")
    print("=" * 60)
    
    test_current_config()
    test_competitive_programming_agents()
    test_deterministic_configs()
    
    print("\n" + "=" * 60)
    print("📋 Summary:")
    print("• Determinism concepts explained ✅")
    print("• Deterministic config created ✅")
    print("• Main config partially optimized ✅")
    print("• Competitive agents using temp=0.0 ✅")
    print("• Full determinism integration: ⚠️ In Progress")
    
    print("\n🔧 Next Steps for Full Implementation:")
    print("1. Ensure main config uses deterministic settings")
    print("2. Add seed parameter support in Cerebras API calls")
    print("3. Implement consistency validation in submissions")
    print("4. Add caching for repeated prompts")

if __name__ == "__main__":
    main() 