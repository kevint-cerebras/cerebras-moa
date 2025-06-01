def calculate_user_metrics(users, start_date, end_date):
    """
    Calculate user engagement metrics with optimized performance and comprehensive error handling.
    
    This function processes user engagement data with:
    - Streamlined validation and minimal logging overhead
    - Optimized date parsing with caching
    - Robust error handling for all operations
    - Correct mathematical calculations for all edge cases
    
    Args:
        users (List[Dict]): List of user dictionaries with engagement data
        start_date: Start date for filtering (supports multiple formats)
        end_date: End date for filtering (supports multiple formats)
        
    Returns:
        Dict: Contains average_engagement, top_performers, and active_count
        
    Raises:
        TypeError: For invalid input types
        ValueError: For invalid input values or date ranges
    """
    import logging
    from datetime import datetime, date
    from typing import List, Dict, Any, Union, Optional
    import sys
    from dateutil import parser as date_parser  # Robust date parsing library
    import heapq  # For efficient sorting of large datasets
    
    # === OPTIMIZED LOGGING SETUP (MINIMAL OVERHEAD) ===
    logging.basicConfig(level=logging.WARNING)  # Reduced logging for performance
    logger = logging.getLogger(__name__)
    
    # Date parsing cache for performance optimization
    _date_cache = {}
    
    def parse_date_optimized(date_obj: Any, field_name: str) -> Optional[datetime]:
        """
        Optimized date parsing with caching and streamlined validation.
        
        Performance optimizations:
        - Caching for repeated date strings
        - Fast-path for common types
        - Minimal error handling overhead
        """
        if date_obj is None:
            raise ValueError(f"Date field '{field_name}' cannot be None")
        
        # Fast path for datetime objects
        if isinstance(date_obj, datetime):
            return date_obj
        elif isinstance(date_obj, date):
            return datetime.combine(date_obj, datetime.min.time())
        
        # Cache lookup for string dates (performance optimization)
        if isinstance(date_obj, str):
            cache_key = date_obj.strip()
            if cache_key in _date_cache:
                return _date_cache[cache_key]
            
            if not cache_key:
                raise ValueError(f"Date field '{field_name}' cannot be empty")
            
            try:
                # Optimized parsing with dateutil
                parsed_date = date_parser.parse(cache_key)
                
                # Quick validation for reasonable range
                if not (datetime(1900, 1, 1) <= parsed_date <= datetime(2100, 12, 31)):
                    raise ValueError(f"Date '{date_obj}' outside reasonable range (1900-2100)")
                
                # Cache the result for performance
                _date_cache[cache_key] = parsed_date
                return parsed_date
                
            except Exception as e:
                raise ValueError(f"Invalid date format for {field_name}: '{date_obj}'")
        
        # Handle numeric timestamps
        elif isinstance(date_obj, (int, float)):
            return datetime.fromtimestamp(date_obj)
        else:
            raise TypeError(f"Unsupported date type for {field_name}: {type(date_obj)}")
    
    def validate_numeric_optimized(value: Any, field_name: str, default_value: float = 0.0) -> float:
        """
        Streamlined numeric validation with minimal overhead.
        
        Performance optimizations:
        - Fast-path for common numeric types
        - Simplified error handling
        - Efficient type conversion
        """
        if value is None:
            return default_value
        
        # Fast path for numeric types
        if isinstance(value, (int, float)):
            if value < 0:
                return default_value
            return float(value)
        
        # String conversion with basic cleaning
        elif isinstance(value, str):
            cleaned = value.strip().replace(',', '')
            if not cleaned:
                return default_value
            try:
                converted = float(cleaned)
                return max(0.0, converted)  # Ensure non-negative
            except ValueError:
                return default_value
        else:
            return default_value
    
    # === STREAMLINED INPUT VALIDATION ===
    
    if not isinstance(users, list):
        raise TypeError(f"'users' must be a list, received {type(users).__name__}")
    
    if not users:
        return {'average_engagement': 0.0, 'top_performers': [], 'active_count': 0}
    
    # Optimized date validation
    try:
        parsed_start_date = parse_date_optimized(start_date, "start_date")
        parsed_end_date = parse_date_optimized(end_date, "end_date")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Date validation failed: {str(e)}")
    
    if parsed_start_date > parsed_end_date:
        raise ValueError(f"start_date ({parsed_start_date}) cannot be after end_date ({parsed_end_date})")
    
    # === OPTIMIZED PROCESSING WITH CORRECT MATHEMATICAL CALCULATIONS ===
    
    active_users = []
    total_engagement_score = 0.0
    
    # Single-pass processing with minimal overhead
    for user_index, user_data in enumerate(users):
        # Fast validation
        if not isinstance(user_data, dict):
            continue
        
        # Handle missing last_login field
        if 'last_login' not in user_data:
            continue
            
        # Optimized date parsing
        try:
            user_last_login = parse_date_optimized(user_data['last_login'], f"user[{user_index}].last_login")
        except (ValueError, TypeError):
            continue
        
        # Date range filtering
        if not (parsed_start_date <= user_last_login <= parsed_end_date):
            continue
        
        # Extract and validate numeric fields with defaults
        posts = validate_numeric_optimized(user_data.get('posts', 0), f"posts")
        comments = validate_numeric_optimized(user_data.get('comments', 0), f"comments") 
        likes = validate_numeric_optimized(user_data.get('likes', 0), f"likes")
        
        # FIXED: Handle division by zero with proper default and edge case validation
        days_active = validate_numeric_optimized(user_data.get('days_active', 1), f"days_active", default_value=1.0)
        # Ensure days_active is never zero to prevent division by zero in ALL cases
        if days_active <= 0:
            days_active = 1.0
        
        # Calculate engagement score
        raw_engagement_score = posts * 2.0 + comments * 1.5 + likes * 0.1
        engagement_score = raw_engagement_score / days_active
        
        # Create user record
        user_record = {
            'posts': int(posts),
            'comments': int(comments),
            'likes': int(likes),
            'days_active': int(days_active),
            'last_login': user_last_login,
            'engagement_score': round(engagement_score, 6)
        }
        
        # FIXED: Correct denominator usage - sum engagement scores for proper average
        total_engagement_score += engagement_score
        active_users.append(user_record)
    
    # Handle no active users edge case
    active_count = len(active_users)
    if active_count == 0:
        return {'average_engagement': 0.0, 'top_performers': [], 'active_count': 0}
    
    # === PERFORMANCE-OPTIMIZED SORTING WITH ERROR HANDLING ===
    
    try:
        # FIXED: Correct average calculation using engagement scores (not raw scores)
        # This ensures the denominator is always correct in all mathematical cases
        average_engagement = total_engagement_score / active_count
        
        # Optimized sorting with comprehensive error handling
        try:
            if active_count <= 5:
                # Simple sort for small datasets
                top_performers = sorted(active_users, key=lambda x: x['engagement_score'], reverse=True)
            elif active_count <= 1000:
                # Standard sort for medium datasets
                top_performers = sorted(active_users, key=lambda x: x['engagement_score'], reverse=True)[:5]
            else:
                # FIXED: Added error handling for heapq.nlargest function
                top_performers = heapq.nlargest(5, active_users, key=lambda x: x['engagement_score'])
        except (TypeError, ValueError, KeyError) as sort_error:
            # Fallback to simple sort if heapq fails
            logger.warning(f"Heap sort failed, using fallback: {sort_error}")
            top_performers = sorted(active_users, key=lambda x: x.get('engagement_score', 0), reverse=True)[:5]
        
        # Ensure exactly 5 results maximum
        top_performers = top_performers[:5]
        
        # Final result with validated calculations
        result = {
            'average_engagement': round(average_engagement, 6),
            'top_performers': top_performers,
            'active_count': active_count
        }
        
        return result
        
    except ZeroDivisionError:
        # Additional safety check for denominator edge cases
        return {'average_engagement': 0.0, 'top_performers': [], 'active_count': 0}
    except Exception as e:
        raise ValueError(f"Calculation failed: {str(e)}")