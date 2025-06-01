#!/usr/bin/env python3
"""Test the zero days active case specifically"""

from competitive_programming import validate_and_execute_code, SecureExecutionError

def test_zero_days_active():
    """Test that code can handle zero days active without ValueError issues"""
    print("🧪 Testing Zero Days Active Case\n")
    
    # Test code that properly handles zero days active
    test_code = '''
def calculate_user_metrics(users, start_date, end_date):
    """Calculate user engagement metrics with zero days handling"""
    
    if not users:
        return {'average_engagement': 0, 'top_performers': [], 'active_count': 0}
    
    active_users = []
    total_score = 0
    
    for user in users:
        if user.get('last_login', '') >= start_date:
            posts = user.get('posts', 0)
            comments = user.get('comments', 0)
            likes = user.get('likes', 0)
            
            # Calculate score
            score = posts * 2 + comments * 1.5 + likes * 0.1
            
            # Handle zero days active with ValueError (test that ValueError is available)
            days_active = user.get('days_active', 1)
            if days_active <= 0:
                raise ValueError(f"Days active must be positive, got {days_active}")
                
            user['engagement_score'] = score / days_active
            active_users.append(user)
            total_score += score
    
    if not active_users:
        return {'average_engagement': 0, 'top_performers': [], 'active_count': 0}
    
    avg = total_score / len(active_users)
    top = sorted(active_users, key=lambda x: x.get('engagement_score', 0), reverse=True)[:5]
    
    return {
        'average_engagement': avg,
        'top_performers': top, 
        'active_count': len(active_users)
    }
'''
    
    # Test case that should trigger ValueError
    test_input = {
        "users": [
            {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 0}
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    }
    
    try:
        # Validate and execute the code
        func, _ = validate_and_execute_code(test_code)
        print("✅ Code validation and execution successful")
        
        # Try to run the function (should raise ValueError)
        try:
            result = func(**test_input)
            print("❌ Expected ValueError was not raised!")
            return False
        except ValueError as e:
            print(f"✅ ValueError properly raised: {e}")
            return True
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
            
    except SecureExecutionError as e:
        print(f"❌ Security validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected validation error: {e}")
        return False

if __name__ == "__main__":
    if test_zero_days_active():
        print("\n🎉 Zero days active test passed!")
    else:
        print("\n⚠️ Zero days active test failed!") 