# 🏆 Code Competition Arena - MoA-Powered Assessment System

This is a competitive programming system built on top of the Cerebras MoA (Mixture of Agents) architecture. It uses specialized AI agents to evaluate code submissions across multiple dimensions: bug detection, edge case handling, performance optimization, and security considerations.

## 🎯 Competition Challenge

**Fix the Buggy Function Challenge**: Students are given a deliberately buggy Python function with multiple issues and must fix all problems to achieve the highest score.

### Original Buggy Function Issues:

1. **Critical Bugs**:
   - Division by zero when `days_active = 0`
   - Wrong denominator in average calculation (uses total users instead of active users)
   - Incorrect sorting direction (ascending instead of descending)
   - Missing KeyError handling for dictionary access

2. **Edge Cases**:
   - Empty input list handling
   - Missing required dictionary keys
   - Invalid date formats
   - No active users in date range

3. **Performance Issues**:
   - Using full sort instead of `heapq.nlargest()` for top-k elements
   - Multiple loops where a single loop would suffice
   - Unnecessary operations and redundant calculations

4. **Security Concerns**:
   - No input validation
   - No data type checking
   - Potential for code injection through eval-like operations

## 🤖 Specialized MoA Agents

The system uses four specialized agents, each with specific expertise:

### 1. Bug Hunter Agent
- **Purpose**: Detect and analyze critical bugs
- **Scoring**: 50 points maximum
- **Focus Areas**:
  - Division by zero errors (15 pts)
  - Wrong denominators (15 pts)
  - Sorting direction issues (10 pts)
  - KeyError handling (10 pts)

### 2. Edge Case Checker Agent
- **Purpose**: Identify edge case handling and robustness
- **Scoring**: 35 points maximum
- **Focus Areas**:
  - Empty input handling (10 pts)
  - Missing dictionary keys (10 pts)
  - Date format issues (8 pts)
  - No active users scenario (7 pts)

### 3. Performance Agent
- **Purpose**: Analyze algorithmic efficiency and optimizations
- **Scoring**: 25 points maximum
- **Focus Areas**:
  - Sorting optimization (12 pts)
  - Single loop efficiency (8 pts)
  - Unnecessary operations (5 pts)

### 4. Security Agent
- **Purpose**: Assess security vulnerabilities and risks
- **Scoring**: 25 points maximum
- **Focus Areas**:
  - Input validation (10 pts)
  - Data sanitization (8 pts)
  - Injection risks (7 pts)

## 🚀 Speed Bonuses

Additional points for quick, correct submissions:
- **1st Place**: +20 points
- **2nd Place**: +10 points
- **3rd Place**: +5 points

**Maximum Total Score**: 135 points

## 🔧 System Architecture

### Database Schema

The system uses SQLite for persistence:

```sql
-- Submissions table
CREATE TABLE submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name TEXT NOT NULL,
    code TEXT NOT NULL,
    code_hash TEXT UNIQUE,
    submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bug_score INTEGER DEFAULT 0,
    edge_case_score INTEGER DEFAULT 0,
    performance_score INTEGER DEFAULT 0,
    security_score INTEGER DEFAULT 0,
    speed_bonus INTEGER DEFAULT 0,
    total_score INTEGER DEFAULT 0,
    analysis_complete BOOLEAN DEFAULT FALSE
);

-- Leaderboard table
CREATE TABLE leaderboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name TEXT NOT NULL,
    best_score INTEGER DEFAULT 0,
    submissions_count INTEGER DEFAULT 0,
    last_submission TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    position INTEGER DEFAULT 0
);
```

### Agent Workflow

1. **Submission Receipt**: Code is hashed to prevent duplicates
2. **Parallel Analysis**: All four specialized agents analyze the code simultaneously
3. **JSON Response Parsing**: Each agent returns structured analysis data
4. **Score Calculation**: Individual scores are aggregated with speed bonuses
5. **Leaderboard Update**: Real-time leaderboard positioning
6. **Results Display**: Detailed feedback for the student

## 🎮 User Interface Features

### 1. Challenge Tab
- Problem description
- Original buggy code
- Test cases
- Hints and tips

### 2. Submit Solution Tab
- Code editor with syntax highlighting
- Local testing capability
- Real-time submission feedback
- Detailed analysis results

### 3. Live Leaderboard Tab
- Auto-refreshing leaderboard
- Medal icons for top 3
- Progress bars showing score percentages
- Submission count tracking

### 4. Analysis Dashboard Tab
- Score distribution charts
- Top performer visualizations
- Competition statistics
- Performance analytics

### 5. Scoring Guide Tab
- Detailed scoring breakdown
- Category explanations
- Point values for each issue
- Example fixes and optimizations

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Competition

1. **Start the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Navigate to Competition**:
   - Select "🏆 Code Competition" from the sidebar

3. **Submit Solutions**:
   - Enter your name
   - Fix the buggy function
   - Submit for analysis

### Environment Setup

Create a `.env` file with your Cerebras API key:
```
CEREBRAS_API_KEY=your_api_key_here
```

## 🏅 Example Perfect Solution

Here's what a perfect scoring solution might look like:

```python
import heapq
from typing import Dict, List, Any
from datetime import datetime

def calculate_user_metrics(users, start_date, end_date):
    # Input validation
    if not isinstance(users, list) or not users:
        return {'average_engagement': 0.0, 'top_performers': [], 'active_count': 0}
    
    # Date processing
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date).date()
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date).date()
    
    total_score = 0
    active_users = []
    
    for user in users:
        if not isinstance(user, dict):
            continue
            
        try:
            last_login = datetime.fromisoformat(user['last_login']).date()
            if start_date <= last_login <= end_date:
                posts = user.get('posts', 0)
                comments = user.get('comments', 0)
                likes = user.get('likes', 0)
                days_active = max(1, user.get('days_active', 1))  # Avoid division by zero
                
                score = posts * 2 + comments * 1.5 + likes * 0.1
                user_copy = user.copy()
                user_copy['engagement_score'] = score / days_active
                
                total_score += score
                active_users.append(user_copy)
        except (KeyError, ValueError, TypeError):
            continue
    
    if not active_users:
        return {'average_engagement': 0.0, 'top_performers': [], 'active_count': 0}
    
    # Fixed: Use active users for average, not total users
    avg_score = total_score / len(active_users)
    
    # Performance: Use heapq instead of full sort
    top_users = heapq.nlargest(5, active_users, key=lambda x: x['engagement_score'])
    
    return {
        'average_engagement': avg_score,
        'top_performers': top_users,
        'active_count': len(active_users)
    }
```

## 🔍 Testing and Validation

The system includes comprehensive test cases:

1. **Normal Operation**: Standard user data
2. **Empty Input**: Empty user list
3. **Zero Days Active**: Division by zero scenario
4. **Missing Keys**: Incomplete user data
5. **No Active Users**: Date range with no matches

## 📊 Analytics and Insights

The system provides rich analytics:
- Real-time submission tracking
- Score distribution analysis
- Performance comparisons
- Time-based submission patterns
- Category-wise scoring breakdowns

## 🎓 Educational Value

This system teaches students:
- **Bug Detection**: How to identify and fix common programming errors
- **Edge Case Handling**: Robust code that handles unexpected inputs
- **Performance Optimization**: Algorithmic efficiency considerations
- **Security Awareness**: Input validation and secure coding practices
- **Testing**: Comprehensive test case development

## 🔮 Future Enhancements

Potential improvements:
- Team-based competitions
- Multiple programming languages
- Custom challenge creation
- Integration with LMS systems
- Automated test case generation
- Code similarity detection
- Detailed execution profiling

## 📝 License

This project is part of the Cerebras MoA system and follows the same licensing terms.

---

*Built with ❤️ using Cerebras MoA, Streamlit, and specialized AI agents* 