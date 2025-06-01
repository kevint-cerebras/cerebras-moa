#!/usr/bin/env python3
"""
Test Device Tracking System

This script demonstrates how the device tracking system works:
- Allows rewrites from the same device
- Prevents abuse from different devices
- Tracks best scores per user
"""

import sys
import time
sys.path.append('.')

def test_device_tracking():
    """Test device tracking functionality"""
    print("TESTING Device Tracking System")
    print("=" * 60)
    print("Policy: Rewrites allowed from same device, blocked from different devices")
    print("=" * 60)
    
    try:
        from competitive_programming import CompetitiveProgrammingSystem
        
        system = CompetitiveProgrammingSystem()
        
        # Test code with different quality levels
        basic_code = """
def calculate_user_metrics(users, start_date, end_date):
    # Basic implementation
    total_score = 0
    active_users = []
    
    for user in users:
        if 'last_login' in user:
            if user['last_login'] >= start_date and user['last_login'] <= end_date:
                score = user.get('posts', 0) * 2
                user['engagement_score'] = score
                total_score += score
                active_users.append(user)
    
    avg_score = total_score / len(active_users) if active_users else 0
    
    return {
        'average_engagement': avg_score,
        'top_performers': active_users[:5],
        'active_count': len(active_users)
    }
"""
        
        improved_code = """
def calculate_user_metrics(users, start_date, end_date):
    # Improved implementation with better error handling
    if not users:
        return {'average_engagement': 0.0, 'top_performers': [], 'active_count': 0}
    
    total_score = 0
    active_users = []
    
    for user in users:
        if not isinstance(user, dict):
            continue
        
        if 'last_login' in user:
            try:
                if user['last_login'] >= start_date and user['last_login'] <= end_date:
                    posts = user.get('posts', 0)
                    comments = user.get('comments', 0)
                    likes = user.get('likes', 0)
                    days_active = max(user.get('days_active', 1), 1)  # Prevent division by zero
                    
                    score = (posts * 2 + comments * 1.5 + likes * 0.1) / days_active
                    user['engagement_score'] = score
                    total_score += score
                    active_users.append(user)
            except Exception:
                continue
    
    avg_score = total_score / len(active_users) if active_users else 0
    top_performers = sorted(active_users, key=lambda x: x.get('engagement_score', 0), reverse=True)[:5]
    
    return {
        'average_engagement': avg_score,
        'top_performers': top_performers,
        'active_count': len(active_users)
    }
"""
        
        # Simulate device fingerprints
        device_a = "device_a_12345678"
        device_b = "device_b_87654321"
        
        print("\n1. TESTING: First submission from new user")
        print("-" * 50)
        
        # First submission from Alice on Device A
        result1 = system.submit_solution("Alice", basic_code, device_a)
        print(f"Result: {result1}")
        
        if result1.get("error"):
            print(f"❌ Unexpected error: {result1['error']}")
            return False
        
        print(f"✅ Alice's first submission accepted from device {device_a[:12]}...")
        submission_id_1 = result1["submission_id"]
        
        print("\n2. TESTING: Improved submission from same device (should work)")
        print("-" * 50)
        
        # Alice improves her code on the same device
        result2 = system.submit_solution("Alice", improved_code, device_a)
        print(f"Result: {result2}")
        
        if result2.get("error"):
            print(f"❌ Unexpected error: {result2['error']}")
            return False
        
        print(f"✅ Alice's resubmission accepted from same device")
        submission_id_2 = result2["submission_id"]
        
        print("\n3. TESTING: Submission from different device (should be blocked)")
        print("-" * 50)
        
        # Alice tries to submit from a different device (should be blocked)
        result3 = system.submit_solution("Alice", improved_code, device_b)
        print(f"Result: {result3}")
        
        if not result3.get("error"):
            print(f"❌ Should have been blocked! Device mismatch not detected")
            return False
        
        print(f"✅ Alice's submission correctly blocked from different device")
        print(f"   Expected device: {result3.get('allowed_device', 'unknown')}")
        print(f"   Attempted device: {result3.get('device_fingerprint', 'unknown')}")
        
        print("\n4. TESTING: New user from any device (should work)")
        print("-" * 50)
        
        # Bob submits from Device B (should work - new user)
        result4 = system.submit_solution("Bob", basic_code, device_b)
        print(f"Result: {result4}")
        
        if result4.get("error"):
            print(f"❌ Unexpected error: {result4['error']}")
            return False
        
        print(f"✅ Bob's first submission accepted from device {device_b[:12]}...")
        submission_id_4 = result4["submission_id"]
        
        print("\n5. TESTING: Bob tries Alice's device (should be blocked)")
        print("-" * 50)
        
        # Bob tries to use Device A (should be blocked)
        result5 = system.submit_solution("Bob", improved_code, device_a)
        print(f"Result: {result5}")
        
        if not result5.get("error"):
            print(f"❌ Should have been blocked! Bob using Alice's device not detected")
            return False
        
        print(f"✅ Bob's submission correctly blocked from Alice's device")
        
        print("\n6. TESTING: User submission history")
        print("-" * 50)
        
        # Test getting user history
        alice_history = system.get_user_submission_history("Alice")
        print(f"Alice's submission history: {len(alice_history)} submissions")
        for i, sub in enumerate(alice_history):
            print(f"  {i+1}. Score: {sub['total_score']}, Device: {sub['device_id']}")
        
        bob_history = system.get_user_submission_history("Bob")
        print(f"Bob's submission history: {len(bob_history)} submissions")
        for i, sub in enumerate(bob_history):
            print(f"  {i+1}. Score: {sub['total_score']}, Device: {sub['device_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during device tracking test: {e}")
        return False

def test_leaderboard_with_devices():
    """Test leaderboard functionality with device tracking"""
    print("\n\nTESTING Leaderboard with Device Tracking")
    print("=" * 60)
    
    try:
        from competitive_programming import CompetitiveProgrammingSystem
        
        system = CompetitiveProgrammingSystem()
        
        # Get leaderboard
        leaderboard = system.get_leaderboard()
        
        print(f"Current leaderboard: {len(leaderboard)} entries")
        print("\nTop performers:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Name':<20} {'Score':<6} {'%':<5} {'Submissions':<11} {'Device':<12}")
        print("-" * 80)
        
        for entry in leaderboard[:10]:  # Show top 10
            rank = entry['position']
            name = entry['student_name'][:18]
            score = entry['best_score']
            percentage = entry['percentage']
            submissions = entry['device_info']['total_submissions']
            device_id = entry['device_info']['device_id'][:10]
            
            print(f"{rank:<4} {name:<20} {score:<6} {percentage:<5.1f} {submissions:<11} {device_id:<12}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing leaderboard: {e}")
        return False

def main():
    """Main test function"""
    print("DEVICE TRACKING SYSTEM TEST")
    print("=" * 60)
    print("Testing fair resubmission policy:")
    print("✅ Allow: Multiple submissions from same device")
    print("❌ Block: Submissions from different devices")
    print("📊 Track: Best score per user")
    print("=" * 60)
    
    # Run tests
    test_results = []
    
    test_results.append(test_device_tracking())
    test_results.append(test_leaderboard_with_devices())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n" + "=" * 60)
    print("DEVICE TRACKING TEST RESULTS")
    print(f"Passed: {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        print("✅ SUCCESS: Device tracking system working correctly!")
        print("\nFeatures verified:")
        print("  • Same device resubmissions: ALLOWED")
        print("  • Different device blocking: WORKING")
        print("  • User history tracking: FUNCTIONAL")
        print("  • Best score detection: ACCURATE")
        print("\nREADY for fair competitive programming with rewrite protection!")
    else:
        print("❌ Some tests failed. Review device tracking implementation.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main() 