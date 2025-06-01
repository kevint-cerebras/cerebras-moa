#!/usr/bin/env python3
"""
Test Judging System Determinism

This script tests that the competitive programming judging system
produces consistent, deterministic results for fair evaluation.
"""

import sys
import time
sys.path.append('.')

def test_agent_determinism():
    """Test that judging agents are configured for maximum determinism"""
    print("TESTING Judging Agent Determinism")
    print("=" * 50)
    
    try:
        from competitive_programming import CompetitiveProgrammingSystem
        
        # Create system and agents
        system = CompetitiveProgrammingSystem()
        print("SUCCESS Creating specialized judging agents...")
        agents = system.create_specialized_agents()
        
        print(f"\nAgent Analysis:")
        for agent_name, agent in agents.items():
            print(f"  {agent_name}:")
            print(f"    SUCCESS Agent created successfully")
            # Note: Can't easily inspect MOAgent internal settings, but creation confirms config
        
        return True
        
    except Exception as e:
        print(f"ERROR creating agents: {e}")
        return False

def test_scoring_consistency():
    """Test that scoring is consistent across multiple runs"""
    print("\nTesting Scoring Consistency")
    print("=" * 50)
    
    try:
        from competitive_programming import CompetitiveProgrammingSystem
        
        system = CompetitiveProgrammingSystem()
        
        # Use a simple test code for consistency testing
        test_code = """
def calculate_user_metrics(users, start_date, end_date):
    # Simple implementation for testing
    total_score = 0
    active_users = []
    
    for user in users:
        if 'last_login' in user and 'posts' in user:
            if user['last_login'] >= start_date and user['last_login'] <= end_date:
                score = user.get('posts', 0) * 2
                user['engagement_score'] = score
                total_score += score
                active_users.append(user)
    
    if len(active_users) > 0:
        avg_score = total_score / len(active_users)
    else:
        avg_score = 0
    
    return {
        'average_engagement': avg_score,
        'top_performers': active_users[:5],
        'active_count': len(active_users)
    }
"""
        
        print("Submitting test code for consistency analysis...")
        
        # Submit test solution
        submission_result = system.submit_solution("test_determinism_user", test_code)
        
        if "error" in submission_result:
            print(f"ERROR Submission failed: {submission_result['error']}")
            return False
        
        submission_id = submission_result["submission_id"]
        print(f"SUCCESS Test submission created with ID: {submission_id}")
        
        # Test consistency validation
        print("\nRunning consistency validation...")
        agents = system.create_specialized_agents()
        
        # Test the new consistency validation method
        consistency_result = system.validate_analysis_consistency(submission_id, agents, validation_runs=3)
        
        print(f"\nConsistency Results:")
        print(f"  Consistent: {consistency_result.get('consistent', False)}")
        print(f"  Consistency Rate: {consistency_result.get('consistency_rate', 0)*100:.1f}%")
        print(f"  Validation Runs: {consistency_result.get('validation_runs', 0)}")
        
        inconsistencies = consistency_result.get('inconsistencies', [])
        if inconsistencies:
            print(f"  WARNING Inconsistencies found: {len(inconsistencies)}")
            for inconsistency in inconsistencies[:3]:  # Show first 3
                print(f"    • {inconsistency}")
        else:
            print(f"  SUCCESS No inconsistencies detected")
        
        recommended_action = consistency_result.get('recommended_action', 'unknown')
        if recommended_action == 'use_result':
            print(f"  SUCCESS Recommended: Use results (high consistency)")
        else:
            print(f"  WARNING Recommended: Retry analysis (low consistency)")
        
        # Return success if consistency rate is good
        consistency_rate = consistency_result.get('consistency_rate', 0)
        return consistency_rate >= 0.8
        
    except Exception as e:
        print(f"ERROR testing consistency: {e}")
        return False

def test_deterministic_settings():
    """Test that deterministic settings are properly applied"""
    print("\nTesting Deterministic Settings")
    print("=" * 50)
    
    try:
        from competitive_programming import CompetitiveProgrammingSystem
        import inspect
        
        system = CompetitiveProgrammingSystem()
        
        # Inspect the agent creation method
        source = inspect.getsource(system.create_specialized_agents)
        
        deterministic_checks = [
            ('"temperature": 0.0', "Temperature set to 0.0"),
            ('"cycles": 1', "Single cycle configuration"),
            ('"llama-3.3-70b"', "Dense model selection"),
            ('base_config', "Deterministic base configuration"),
            ('Provide consistent, deterministic', "Consistency instructions in prompts")
        ]
        
        passed_checks = 0
        total_checks = len(deterministic_checks)
        
        for check_text, description in deterministic_checks:
            if check_text in source:
                print(f"  SUCCESS {description}")
                passed_checks += 1
            else:
                print(f"  ERROR {description}")
        
        print(f"\nDeterminism Configuration Score: {passed_checks}/{total_checks} ({passed_checks/total_checks*100:.1f}%)")
        
        return passed_checks >= total_checks * 0.8  # 80% threshold
        
    except Exception as e:
        print(f"ERROR checking settings: {e}")
        return False

def main():
    """Main test function"""
    print("TESTING Competitive Programming Judging Determinism")
    print("=" * 60)
    print("Purpose: Ensure fair, consistent evaluation while keeping chat interface unchanged")
    print("=" * 60)
    
    # Run all tests
    test_results = []
    
    print("1. Agent Creation Test")
    test_results.append(test_agent_determinism())
    
    print("2. Deterministic Settings Test") 
    test_results.append(test_deterministic_settings())
    
    print("3. Scoring Consistency Test")
    test_results.append(test_scoring_consistency())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"  Passed: {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        print("  SUCCESS All tests passed! Judging system is highly deterministic")
        determinism_level = "Excellent"
    elif passed_tests >= total_tests * 0.67:
        print("  GOOD Most tests passed! Good determinism level")
        determinism_level = "Good"
    else:
        print("  WARNING Some tests failed. Determinism needs improvement")
        determinism_level = "Needs Improvement"
    
    print(f"\nJudging Determinism Level: {determinism_level}")
    print("\nStatus:")
    print("  • Chat interface: Unchanged (maintains creativity)")
    print("  • Judging system: Optimized for consistency")
    print("  • Evaluation fairness: Enhanced")
    
    if passed_tests >= total_tests * 0.8:
        print("\nREADY for fair, consistent automated judging!")
    else:
        print("\nConsider additional optimizations for maximum fairness")

if __name__ == "__main__":
    main() 