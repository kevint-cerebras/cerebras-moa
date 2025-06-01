#!/usr/bin/env python3
"""Test script to verify the expanded import allowlist works correctly"""

from competitive_programming import validate_and_execute_code, SecureExecutionError

def test_expanded_imports():
    """Test that the newly allowed imports work"""
    print("🧪 Testing Expanded Import Allowlist\n")
    
    # Test code with the previously blocked imports
    test_code = '''
import logging
from typing import List, Dict, Optional, Any
import sys
from dateutil import parser
import heapq
import math

def calculate_user_metrics(users: List[Dict], start_date: str, end_date: str) -> Dict[str, Any]:
    """Calculate user engagement metrics with expanded imports"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting user metrics calculation")
    
    # Use typing for better code clarity
    active_users: List[Dict] = []
    total_score: float = 0.0
    
    # Use sys for version checking (safe attributes only)
    logger.info(f"Python version: {sys.version}")
    
    # Handle empty input
    if not users:
        logger.warning("No users provided")
        return {'average_engagement': 0, 'top_performers': [], 'active_count': 0}
    
    for user in users:
        # Use dateutil for robust date parsing
        try:
            user_login = parser.parse(user.get('last_login', '1900-01-01'))
            start_dt = parser.parse(start_date) 
            end_dt = parser.parse(end_date)
            
            if start_dt <= user_login <= end_dt:
                # Calculate engagement score with math functions
                posts = user.get('posts', 0)
                comments = user.get('comments', 0)
                likes = user.get('likes', 0)
                
                # Use math.log for diminishing returns calculation
                score = posts * 2 + comments * 1.5 + math.log(max(1, likes)) * 0.5
                
                # Handle division by zero
                days_active = user.get('days_active', 1)
                if days_active > 0:
                    user['engagement_score'] = score / days_active
                else:
                    user['engagement_score'] = 0
                    
                active_users.append(user)
                total_score += score
                
        except Exception as e:
            logger.error(f"Error processing user: {e}")
            continue
    
    # Handle no active users
    if not active_users:
        logger.warning("No active users found")
        return {'average_engagement': 0, 'top_performers': [], 'active_count': 0}
    
    # Calculate results with heapq for efficiency
    avg_engagement = total_score / len(active_users)
    top_performers = heapq.nlargest(5, active_users, key=lambda x: x.get('engagement_score', 0))
    
    logger.info(f"Processed {len(active_users)} active users")
    
    return {
        'average_engagement': avg_engagement,
        'top_performers': top_performers,
        'active_count': len(active_users)
    }
'''
    
    print("📝 Testing code with expanded imports:")
    print("=" * 60)
    print("Key imports being tested:")
    print("- logging (for debugging and monitoring)")
    print("- typing (for type hints and code clarity)")  
    print("- sys (safe read-only attributes)")
    print("- dateutil (for robust date parsing)")
    print("- math (expanded functions)")
    print("=" * 60)
    
    try:
        # Test security validation and execution
        func, _ = validate_and_execute_code(test_code)
        print("✅ Security validation passed!")
        print("✅ All imports allowed successfully!")
        
        # Test with sample data
        test_data = [
            {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 30},
            {"last_login": "2024-01-10", "posts": 5, "comments": 8, "likes": 50, "days_active": 20}
        ]
        
        result = func(test_data, "2024-01-01", "2024-01-31")
        print("✅ Function execution successful!")
        print(f"📊 Result: {result}")
        
        # Verify result structure
        expected_keys = {'average_engagement', 'top_performers', 'active_count'}
        if set(result.keys()) == expected_keys:
            print("✅ Result structure correct!")
        else:
            print(f"❌ Result structure incorrect: {set(result.keys())} vs {expected_keys}")
            
    except SecureExecutionError as e:
        print(f"❌ Security error: {e}")
        return False
    except Exception as e:
        print(f"❌ Execution error: {e}")
        return False
    
    print("\n🎉 Expanded imports test completed successfully!")
    return True

def test_still_blocked_imports():
    """Test that dangerous imports are still blocked"""
    print("\n🛡️ Testing that dangerous imports are still blocked...")
    
    dangerous_code = '''
import os
import subprocess
import socket
import urllib

def calculate_user_metrics(users, start_date, end_date):
    # Try to do dangerous things
    os.system("echo 'This should be blocked'")
    return {"average_engagement": 0, "top_performers": [], "active_count": 0}
'''
    
    try:
        validate_and_execute_code(dangerous_code)
        print("❌ SECURITY FAILURE: Dangerous imports were allowed!")
        return False
    except SecureExecutionError as e:
        print(f"✅ Security working: {e}")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success1 = test_expanded_imports()
    success2 = test_still_blocked_imports()
    
    if success1 and success2:
        print("\n🎯 All tests passed! Security system properly updated.")
    else:
        print("\n⚠️ Some tests failed. Check the security configuration.") 