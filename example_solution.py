import heapq
from typing import Dict, List, Any, Union
from datetime import datetime

def calculate_user_metrics(users, start_date, end_date):
    """
    Calculate engagement metrics for active users
    
    Fixed version addressing all bugs, edge cases, performance, and security issues.
    
    Args:
        users: List of user dictionaries
        start_date: Start date for filtering (string or datetime)
        end_date: End date for filtering (string or datetime)
    
    Returns:
        Dictionary with engagement metrics
    """
    # Input validation and security
    if not isinstance(users, list):
        raise TypeError("Users must be a list")
    
    if not users:  # Handle empty input
        return {
            'average_engagement': 0.0,
            'top_performers': [],
            'active_count': 0
        }
    
    # Validate and sanitize date inputs
    try:
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date).date()
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date).date()
    except (ValueError, TypeError):
        raise ValueError("Invalid date format. Use YYYY-MM-DD or datetime objects")
    
    # Single loop for efficiency
    total_score = 0
    active_users = []
    
    for user in users:
        # Input validation - check if user is a dictionary
        if not isinstance(user, dict):
            continue  # Skip invalid user entries
        
        try:
            # Validate required keys exist
            last_login = user.get('last_login')
            posts = user.get('posts', 0)
            comments = user.get('comments', 0) 
            likes = user.get('likes', 0)
            days_active = user.get('days_active', 1)  # Default to 1 to avoid division by zero
            
            # Sanitize and validate data types
            posts = max(0, int(posts)) if isinstance(posts, (int, float)) else 0
            comments = max(0, float(comments)) if isinstance(comments, (int, float)) else 0
            likes = max(0, float(likes)) if isinstance(likes, (int, float)) else 0
            days_active = max(1, int(days_active)) if isinstance(days_active, (int, float)) else 1
            
            # Handle date validation
            if isinstance(last_login, str):
                last_login = datetime.fromisoformat(last_login).date()
            elif not hasattr(last_login, '__ge__'):  # Not a date-like object
                continue
            
            # Check if user is active in date range
            if start_date <= last_login <= end_date:
                # Calculate engagement score
                score = posts * 2 + comments * 1.5 + likes * 0.1
                
                # Fixed: Division by zero protection (days_active defaults to 1)
                engagement_score = score / days_active
                
                # Create a copy to avoid modifying original data
                user_copy = user.copy()
                user_copy['engagement_score'] = engagement_score
                
                total_score += score
                active_users.append(user_copy)
                
        except (KeyError, ValueError, TypeError):
            # Skip users with invalid data instead of crashing
            continue
    
    # Handle no active users case
    if not active_users:
        return {
            'average_engagement': 0.0,
            'top_performers': [],
            'active_count': 0
        }
    
    # Fixed: Calculate average using active users count, not total users
    avg_score = total_score / len(active_users)
    
    # Fixed: Performance optimization - use heapq for top-k instead of full sort
    # Fixed: Correct sorting direction (largest engagement scores first)
    top_users = heapq.nlargest(5, active_users, key=lambda x: x['engagement_score'])
    
    return {
        'average_engagement': round(avg_score, 2),
        'top_performers': top_users,
        'active_count': len(active_users)
    }

# Example test function to validate the solution
def test_solution():
    """Test the corrected function with various edge cases"""
    
    # Test case 1: Normal operation
    users1 = [
        {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 30},
        {"last_login": "2024-01-10", "posts": 5, "comments": 8, "likes": 50, "days_active": 20}
    ]
    result1 = calculate_user_metrics(users1, "2024-01-01", "2024-01-31")
    print("Test 1 (Normal):", result1)
    
    # Test case 2: Empty users list
    result2 = calculate_user_metrics([], "2024-01-01", "2024-01-31")
    print("Test 2 (Empty):", result2)
    
    # Test case 3: Zero days active
    users3 = [
        {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 0}
    ]
    result3 = calculate_user_metrics(users3, "2024-01-01", "2024-01-31")
    print("Test 3 (Zero days):", result3)
    
    # Test case 4: Missing keys
    users4 = [
        {"last_login": "2024-01-15", "posts": 10},  # Missing comments, likes, days_active
        {"last_login": "2024-01-12", "posts": 5, "comments": 3, "likes": 25, "days_active": 15}
    ]
    result4 = calculate_user_metrics(users4, "2024-01-01", "2024-01-31")
    print("Test 4 (Missing keys):", result4)
    
    # Test case 5: No active users in date range
    users5 = [
        {"last_login": "2023-12-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 30}
    ]
    result5 = calculate_user_metrics(users5, "2024-01-01", "2024-01-31")
    print("Test 5 (No active users):", result5)

if __name__ == "__main__":
    test_solution() 