import json
import time
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Tuple
import streamlit as st
from moa.agent import MOAgent

# Scoring configuration
SCORING = {
    'bug_detection': {
        'division_by_zero': 15,
        'wrong_denominator': 15, 
        'sorting_direction': 10,
        'keyerror_handling': 10
    },
    'edge_cases': {
        'empty_input': 10,
        'missing_keys': 10,
        'date_formats': 8,
        'no_active_users': 7
    },
    'performance': {
        'sorting_optimization': 12,
        'single_loop': 8,
        'unnecessary_operations': 5
    },
    'security': {
        'input_validation': 10,
        'data_sanitization': 8,
        'injection_risks': 7
    },
    'speed_bonus': {
        'first_place': 20,
        'second_place': 10,
        'third_place': 5
    }
}

# Original buggy function for reference
ORIGINAL_FUNCTION = """
def calculate_user_metrics(users, start_date, end_date):
    \"\"\"Calculate engagement metrics for active users\"\"\"
    total_score = 0
    active_users = []
    
    for user in users:
        if user['last_login'] >= start_date and user['last_login'] <= end_date:
            # Calculate engagement score
            score = user['posts'] * 2 + user['comments'] * 1.5 + user['likes'] * 0.1
            user['engagement_score'] = score / user['days_active']
            total_score += score
            active_users.append(user)
    
    # Calculate averages
    avg_score = total_score / len(users)
    top_users = sorted(active_users, key=lambda x: x['engagement_score'])[-5:]
    
    return {
        'average_engagement': avg_score,
        'top_performers': top_users,
        'active_count': len(active_users)
    }
"""

# Scoring schema for structured outputs
SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 50,
            "description": "The score for this analysis category"
        },
        "feedback": {
            "type": "string",
            "description": "Brief feedback about the analysis"
        },
        "issues_found": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific issues found"
        }
    },
    "required": ["score", "feedback", "issues_found"],
    "additionalProperties": False
}

# Updated agent prompts for cleaner analysis
AGENT_PROMPTS = {
    "bug_hunter": """Analyze the provided Python code for bugs and assign a score out of 50 points.

Common bugs to look for:
- Division by zero errors
- KeyError from missing dictionary keys
- Wrong sorting direction
- Incorrect denominator usage
- Logic errors

Provide a score (0-50) based on how well the code handles these potential bugs.""",

    "edge_case_checker": """Analyze the provided Python code for edge case handling and assign a score out of 35 points.

Edge cases to consider:
- Empty input lists
- Missing dictionary keys
- Zero or negative values
- Invalid date ranges
- No active users scenario

Provide a score (0-35) based on how well the code handles edge cases.""",

    "performance_agent": """Analyze the provided Python code for performance and assign a score out of 25 points.

Performance factors:
- Efficient sorting algorithms
- Minimal loops and operations
- Good algorithm complexity
- Unnecessary operations

Provide a score (0-25) based on the code's performance characteristics.""",

    "security_agent": """Analyze the provided Python code for security and assign a score out of 25 points.

Security considerations:
- Input validation
- Safe data handling
- Protection against injection
- Proper error handling

Provide a score (0-25) based on the code's security practices."""
}

class CompetitiveProgrammingSystem:
    def __init__(self):
        self.db_path = "competition.db"
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for submissions and leaderboard"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # Enable WAL mode for better concurrent access
                conn.execute('PRAGMA journal_mode=WAL;')
                conn.execute('PRAGMA synchronous=NORMAL;')
                conn.execute('PRAGMA cache_size=10000;')
                conn.execute('PRAGMA temp_store=memory;')
                
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS submissions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_name TEXT NOT NULL,
                        code TEXT NOT NULL,
                        code_hash TEXT,
                        submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        bug_score INTEGER DEFAULT 0,
                        edge_case_score INTEGER DEFAULT 0,
                        performance_score INTEGER DEFAULT 0,
                        security_score INTEGER DEFAULT 0,
                        speed_bonus INTEGER DEFAULT 0,
                        total_score INTEGER DEFAULT 0,
                        analysis_complete BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS leaderboard (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_name TEXT NOT NULL,
                        best_score INTEGER DEFAULT 0,
                        submissions_count INTEGER DEFAULT 0,
                        last_submission TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        position INTEGER DEFAULT 0
                    )
                ''')
                
                conn.commit()
                print("✅ Database initialized successfully with WAL mode")
        except Exception as e:
            print(f"❌ Error initializing database: {str(e)}")
    
    def create_specialized_agents(self) -> Dict[str, MOAgent]:
        """Create specialized MOA agents for different analysis types"""
        agents = {}
        
        # Define max scores per agent type
        max_scores = {
            "bug_hunter": 50,
            "edge_case_checker": 35, 
            "performance_agent": 25,
            "security_agent": 25
        }
        
        for agent_type, prompt in AGENT_PROMPTS.items():
            # Create schema specific to this agent's max score
            agent_schema = {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "description": f"The score for {agent_type} analysis (0-{max_scores[agent_type]})"
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Brief feedback about the analysis"
                    },
                    "issues_found": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific issues found"
                    }
                },
                "required": ["score", "feedback", "issues_found"],
                "additionalProperties": False
            }
            
            agent_config = {
                "main_model": "llama-3.3-70b",
                "cycles": 1,
                "temperature": 0.1,
                "system_prompt": prompt,
                # Add structured output configuration
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{agent_type}_analysis", 
                        "strict": True,
                        "schema": agent_schema
                    }
                }
            }
            
            layer_config = {
                f"{agent_type}_agent": {
                    "system_prompt": prompt + " {helper_response}",
                    "model_name": "llama-4-scout-17b-16e-instruct",
                    "temperature": 0.1
                }
            }
            
            agents[agent_type] = MOAgent.from_config(
                main_model=agent_config["main_model"],
                system_prompt=agent_config["system_prompt"],
                cycles=agent_config["cycles"],
                temperature=agent_config["temperature"],
                layer_agent_config=layer_config,
                **{"response_format": agent_config["response_format"]}  # Pass structured output config
            )
        
        return agents
    
    def submit_solution(self, student_name: str, code: str) -> Dict[str, Any]:
        """Submit a solution for analysis"""
        # Generate hash for tracking (but allow duplicates)
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Insert new submission (removed duplicate check)
                cursor.execute('''
                    INSERT INTO submissions (student_name, code, code_hash)
                    VALUES (?, ?, ?)
                ''', (student_name, code, code_hash))
                
                submission_id = cursor.lastrowid
                conn.commit()
                
                return {"submission_id": submission_id, "status": "submitted", "code_hash": code_hash}
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}
    
    def analyze_submission(self, submission_id: int, agents: Dict[str, MOAgent]) -> Dict[str, Any]:
        """Analyze a submission using specialized agents"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Get submission
                cursor.execute('SELECT code, student_name FROM submissions WHERE id = ?', (submission_id,))
                result = cursor.fetchone()
                if not result:
                    return {"error": "Submission not found"}
                
                code, student_name = result
                
                # Analyze with each specialized agent
                analysis_results = {}
                total_score = 0
                
                for agent_type, agent in agents.items():
                    try:
                        # Debug: Print the code being analyzed
                        print(f"🔍 Analyzing with {agent_type}...")
                        print(f"📋 Code snippet: {code[:100]}...")
                        
                        # Pass the code as user input instead of formatting into system prompt
                        user_message = f"Please analyze this Python code:\n\n```python\n{code}\n```"
                        
                        print(f"📝 User message length: {len(user_message)}")
                        
                        # Get agent response with structured output
                        response = ""
                        for chunk in agent.chat(user_message):
                            # Handle both string chunks and ResponseChunk objects
                            if isinstance(chunk, str):
                                response += chunk
                            elif hasattr(chunk, 'get') and chunk.get('response_type') == 'output':
                                response += chunk.get('delta', '')
                            elif hasattr(chunk, 'response_type') and chunk.response_type == 'output':
                                response += chunk.delta
                        
                        print(f"📝 {agent_type} response: {response[:200]}...")  # First 200 chars
                        
                        # Parse JSON response - should be valid due to structured outputs
                        try:
                            analysis_data = json.loads(response)
                            analysis_results[agent_type] = analysis_data
                            agent_score = analysis_data.get('score', 0)
                            total_score += agent_score
                            print(f"✅ {agent_type} scored: {agent_score}")
                            print(f"💬 {agent_type} feedback: {analysis_data.get('feedback', 'No feedback')}")
                        except json.JSONDecodeError as json_error:
                            print(f"❌ {agent_type} JSON parse error: {str(json_error)}")
                            print(f"📄 Raw response: {response}")
                            
                            # This should rarely happen with structured outputs, but keep fallback
                            fallback_score = self.analyze_code_quality(code, agent_type)
                            analysis_results[agent_type] = {
                                "error": "Failed to parse agent response", 
                                "score": fallback_score,
                                "feedback": "Fallback scoring used due to parsing error",
                                "issues_found": ["JSON parsing failed"],
                                "fallback": True
                            }
                            total_score += fallback_score
                            print(f"🔄 {agent_type} fallback score: {fallback_score}")
                        
                    except Exception as e:
                        print(f"❌ {agent_type} analysis error: {str(e)}")
                        analysis_results[agent_type] = {
                            "error": str(e), 
                            "score": 0,
                            "feedback": "Analysis failed due to error",
                            "issues_found": [f"Error: {str(e)}"]
                        }
                
                # Calculate speed bonus (based on submission order)
                cursor.execute('SELECT COUNT(*) FROM submissions WHERE analysis_complete = TRUE')
                completed_submissions = cursor.fetchone()[0]
                
                speed_bonus = 0
                if completed_submissions == 0:
                    speed_bonus = SCORING['speed_bonus']['first_place']
                elif completed_submissions == 1:
                    speed_bonus = SCORING['speed_bonus']['second_place']
                elif completed_submissions == 2:
                    speed_bonus = SCORING['speed_bonus']['third_place']
                
                total_score += speed_bonus
                
                # Update submission with results
                cursor.execute('''
                    UPDATE submissions 
                    SET bug_score = ?, edge_case_score = ?, performance_score = ?, security_score = ?,
                        speed_bonus = ?, total_score = ?, analysis_complete = TRUE
                    WHERE id = ?
                ''', (
                    analysis_results.get('bug_hunter', {}).get('score', 0),
                    analysis_results.get('edge_case_checker', {}).get('score', 0),
                    analysis_results.get('performance_agent', {}).get('score', 0),
                    analysis_results.get('security_agent', {}).get('score', 0),
                    speed_bonus,
                    total_score,
                    submission_id
                ))
                
                conn.commit()
                
                # Update leaderboard in a separate transaction to avoid locking
                try:
                    self.update_leaderboard(student_name, total_score)
                except Exception as leaderboard_error:
                    print(f"Warning: Leaderboard update failed: {str(leaderboard_error)}")
                    # Don't fail the analysis if leaderboard update fails
                
                return {
                    "analysis_results": analysis_results,
                    "total_score": total_score,
                    "speed_bonus": speed_bonus,
                    "student_name": student_name
                }
        except Exception as e:
            return {"error": f"Database error during analysis: {str(e)}"}
    
    def update_leaderboard(self, student_name: str, score: int):
        """Update the leaderboard with new score"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Check if student exists in leaderboard
                cursor.execute('SELECT best_score, submissions_count FROM leaderboard WHERE student_name = ?', (student_name,))
                result = cursor.fetchone()
                
                if result:
                    current_best, submissions_count = result
                    new_best = max(current_best, score)
                    
                    cursor.execute('''
                        UPDATE leaderboard 
                        SET best_score = ?, submissions_count = ?, last_submission = CURRENT_TIMESTAMP
                        WHERE student_name = ?
                    ''', (new_best, submissions_count + 1, student_name))
                else:
                    cursor.execute('''
                        INSERT INTO leaderboard (student_name, best_score, submissions_count)
                        VALUES (?, ?, 1)
                    ''', (student_name, score))
                
                # Update positions
                cursor.execute('''
                    UPDATE leaderboard 
                    SET position = (
                        SELECT COUNT(*) + 1 
                        FROM leaderboard l2 
                        WHERE l2.best_score > leaderboard.best_score
                    )
                ''')
                
                conn.commit()
        except Exception as e:
            print(f"Error updating leaderboard: {str(e)}")
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current leaderboard"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT student_name, best_score, submissions_count, last_submission, position
                    FROM leaderboard 
                    ORDER BY best_score DESC, last_submission ASC
                    LIMIT 20
                ''')
                
                results = cursor.fetchall()
                
                leaderboard = []
                for i, (name, score, submissions, last_sub, position) in enumerate(results):
                    leaderboard.append({
                        "position": i + 1,
                        "student_name": name,
                        "best_score": score,
                        "submissions_count": submissions,
                        "last_submission": last_sub,
                        "percentage": round((score / 170) * 100, 1)  # Max score is 170
                    })
                
                return leaderboard
        except Exception as e:
            print(f"Error getting leaderboard: {str(e)}")
            return []
    
    def get_submission_details(self, submission_id: int) -> Dict[str, Any]:
        """Get detailed analysis of a specific submission"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT student_name, code, submission_time, bug_score, edge_case_score,
                           performance_score, security_score, speed_bonus, total_score
                    FROM submissions 
                    WHERE id = ?
                ''', (submission_id,))
                
                result = cursor.fetchone()
                
                if not result:
                    return {"error": "Submission not found"}
                
                return {
                    "student_name": result[0],
                    "code": result[1],
                    "submission_time": result[2],
                    "bug_score": result[3],
                    "edge_case_score": result[4],
                    "performance_score": result[5],
                    "security_score": result[6],
                    "speed_bonus": result[7],
                    "total_score": result[8]
                }
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}
    
    def rebuild_leaderboard(self):
        """Rebuild the entire leaderboard from scratch"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Clear existing leaderboard
                cursor.execute('DELETE FROM leaderboard')
                
                # Get all completed submissions grouped by student
                cursor.execute('''
                    SELECT student_name, MAX(total_score) as best_score, COUNT(*) as submission_count
                    FROM submissions 
                    WHERE analysis_complete = TRUE
                    GROUP BY student_name
                    ORDER BY best_score DESC
                ''')
                
                results = cursor.fetchall()
                
                # Rebuild leaderboard
                for student_name, best_score, submission_count in results:
                    cursor.execute('''
                        INSERT INTO leaderboard (student_name, best_score, submissions_count)
                        VALUES (?, ?, ?)
                    ''', (student_name, best_score, submission_count))
                
                # Update positions
                cursor.execute('''
                    UPDATE leaderboard 
                    SET position = (
                        SELECT COUNT(*) + 1 
                        FROM leaderboard l2 
                        WHERE l2.best_score > leaderboard.best_score
                    )
                ''')
                
                conn.commit()
                print(f"✅ Leaderboard rebuilt with {len(results)} entries")
                return True
        except Exception as e:
            print(f"❌ Error rebuilding leaderboard: {str(e)}")
            return False

    def analyze_code_quality(self, code: str, agent_type: str) -> int:
        """Provide fallback scoring based on static code analysis"""
        try:
            # Basic static analysis to provide reasonable fallback scores
            score = 0
            
            if agent_type == "bug_hunter":
                # Check for common bug fixes
                if 'days_active' in code and '> 0' in code:
                    score += 15  # Division by zero check
                if "'last_login' in user" in code:
                    score += 10  # Key existence check
                if "'posts' in user" in code:
                    score += 10  # Key existence check  
                if "reverse=True" in code:
                    score += 10  # Correct sorting direction
                if "len(active_users)" in code:
                    score += 5   # Correct denominator
                    
            elif agent_type == "edge_case_checker":
                # Check for edge case handling
                if "if active_users" in code or "len(active_users)" in code:
                    score += 10  # Empty list check
                if " in user" in code:
                    score += 10  # Key existence checks
                if "else" in code:
                    score += 10  # Alternative handling
                if "0" in code:
                    score += 5   # Zero value handling
                    
            elif agent_type == "performance_agent":
                # Check for performance patterns
                if "sorted(" in code:
                    score += 10  # Good sorting
                if "lambda" in code:
                    score += 8   # Efficient key function
                if code.count("for") <= 1:
                    score += 7   # Single loop
                    
            elif agent_type == "security_agent":
                # Check for security patterns
                if " in user" in code:
                    score += 10  # Input validation
                if "get(" in code:
                    score += 8   # Safe dictionary access
                if "try:" in code:
                    score += 7   # Error handling
                    
            return min(score, 50)  # Cap at reasonable maximum
            
        except Exception:
            # If static analysis fails, return minimal score
            return 5

# Test cases for validation
TEST_CASES = [
    {
        "name": "Normal case",
        "users": [
            {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 30},
            {"last_login": "2024-01-10", "posts": 5, "comments": 8, "likes": 50, "days_active": 20}
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    },
    {
        "name": "Empty users",
        "users": [],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    },
    {
        "name": "Zero days active",
        "users": [
            {"last_login": "2024-01-15", "posts": 10, "comments": 5, "likes": 100, "days_active": 0}
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    },
    {
        "name": "Missing keys",
        "users": [
            {"last_login": "2024-01-15", "posts": 10}  # Missing comments, likes, days_active
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    }
] 