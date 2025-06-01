#!/usr/bin/env python3
"""Test dateutil import functionality"""

from competitive_programming import validate_and_execute_code, SecureExecutionError

def test_dateutil():
    """Test dateutil functionality"""
    print("🧪 Testing dateutil Import\n")
    
    test_code = '''
from dateutil import parser
from datetime import datetime

def calculate_user_metrics(users, start_date, end_date):
    """Test with dateutil for robust date parsing"""
    
    if not users:
        return {'average_engagement': 0, 'top_performers': [], 'active_count': 0}
    
    active_users = []
    total_score = 0
    
    # Parse dates with dateutil for flexibility
    try:
        start_dt = parser.parse(start_date)
        end_dt = parser.parse(end_date)
    except:
        # Fallback to basic string comparison
        start_dt = start_date
        end_dt = end_date
    
    for user in users:
        try:
            # Try parsing with dateutil first
            user_login = parser.parse(user.get('last_login', '1900-01-01'))
            if isinstance(start_dt, datetime) and isinstance(end_dt, datetime):
                date_check = start_dt <= user_login <= end_dt
            else:
                date_check = user.get('last_login', '') >= start_date
        except:
            # Fallback to string comparison
            date_check = user.get('last_login', '') >= start_date
            
        if date_check:
            posts = user.get('posts', 0)
            comments = user.get('comments', 0)
            
            score = posts * 2 + comments * 1.5
            
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
    
    return {
        'average_engagement': avg,
        'top_performers': active_users[:5],
        'active_count': len(active_users)
    }
'''
    
    try:
        func, _ = validate_and_execute_code(test_code)
        print("✅ Security validation passed!")
        print("✅ dateutil import allowed!")
        
        # Test with various date formats
        test_data = [
            {"last_login": "2024-01-15", "posts": 10, "comments": 5, "days_active": 30},
            {"last_login": "01/10/2024", "posts": 5, "comments": 8, "days_active": 20},  # US format
            {"last_login": "2024-01-20T10:30:00", "posts": 8, "comments": 3, "days_active": 25}  # ISO format
        ]
        
        result = func(test_data, "2024-01-01", "2024-01-31")
        print("✅ Function execution successful!")
        print(f"📊 Result: {result}")
        print(f"🎯 Processed {result['active_count']} users with various date formats")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if test_dateutil():
        print("\n🎯 dateutil test passed!")
    else:
        print("\n⚠️ dateutil test failed!") 