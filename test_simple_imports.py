#!/usr/bin/env python3
"""Simple test for expanded imports"""

from competitive_programming import validate_and_execute_code, SecureExecutionError

def test_basic_imports():
    """Test basic functionality of expanded imports"""
    print("🧪 Testing Basic Expanded Imports\n")
    
    test_code = '''
from typing import List, Dict, Optional
import sys
import math
import heapq

def calculate_user_metrics(users: List[Dict], start_date: str, end_date: str) -> Dict:
    """Simple test with type hints and math"""
    
    if not users:
        return {'average_engagement': 0, 'top_performers': [], 'active_count': 0}
    
    active_users = []
    total_score = 0
    
    for user in users:
        if user.get('last_login', '') >= start_date:
            posts = user.get('posts', 0)
            comments = user.get('comments', 0)
            likes = user.get('likes', 0)
            
            # Use math functions
            score = posts * 2 + comments * 1.5 + math.sqrt(max(1, likes))
            
            days_active = user.get('days_active', 1)
            if days_active > 0:
                user['engagement_score'] = score / days_active
            else:
                user['engagement_score'] = 0
                
            active_users.append(user)
            total_score += score
    
    if not active_users:
        return {'average_engagement': 0, 'top_performers': [], 'active_count': 0}
    
    avg = total_score / len(active_users)
    top = heapq.nlargest(5, active_users, key=lambda x: x.get('engagement_score', 0))
    
    return {
        'average_engagement': avg,
        'top_performers': top,
        'active_count': len(active_users)
    }
'''
    
    try:
        func, _ = validate_and_execute_code(test_code)
        print("✅ Security validation passed!")
        
        # Test execution
        test_data = [
            {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 30},
            {"last_login": "2024-01-10", "posts": 5, "comments": 8, "likes": 50, "days_active": 20}
        ]
        
        result = func(test_data, "2024-01-01", "2024-01-31")
        print("✅ Function execution successful!")
        print(f"📊 Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if test_basic_imports():
        print("\n🎯 Basic imports test passed!")
    else:
        print("\n⚠️ Basic imports test failed!") 