# 🚀 Quick Start Guide - Code Competition Arena

## For Students

### 1. Launch the Competition
```bash
streamlit run app.py
```

### 2. Navigate to Competition
- Open your browser (usually opens automatically)
- In the sidebar, select **"🏆 Code Competition"**

### 3. Start Competing!
1. **Read the Challenge** - Check out the buggy function and test cases
2. **Submit Your Solution** - Enter your name and fix the code
3. **Watch the Leaderboard** - See your score and ranking in real-time!

### 4. Scoring Breakdown
- **Bug Detection**: 50 points (division by zero, wrong denominators, etc.)
- **Edge Cases**: 35 points (empty inputs, missing keys, etc.) 
- **Performance**: 25 points (efficient algorithms, single loops, etc.)
- **Security**: 25 points (input validation, sanitization, etc.)
- **Speed Bonus**: Up to 20 points for first 3 submissions!

**Maximum Score: 135 points** 🏆

## For Instructors

### Setting Up a Competition

1. **Environment Setup**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up Cerebras API key
   echo "CEREBRAS_API_KEY=your_api_key_here" > .env
   ```

2. **Launch the System**
   ```bash
   # Option 1: Use the launch script
   python launch_competition.py
   
   # Option 2: Direct Streamlit launch
   streamlit run app.py
   ```

3. **Competition Flow**
   - Students submit code solutions
   - 4 specialized AI agents analyze each submission:
     - 🐛 **Bug Hunter**: Finds critical bugs
     - 🎯 **Edge Case Checker**: Tests robustness  
     - ⚡ **Performance Agent**: Optimizes algorithms
     - 🔒 **Security Agent**: Validates safety
   - Real-time scoring and leaderboard updates
   - Detailed feedback for learning

### Features for Education

- **Live Leaderboard** with auto-refresh
- **Detailed Analysis** from specialized AI agents
- **Local Testing** before submission
- **Progress Tracking** with analytics dashboard
- **Duplicate Prevention** via code hashing
- **Export Results** for grading integration

### Customizing the Challenge

Want to create your own challenge? Modify these files:
- `ORIGINAL_FUNCTION` in `competitive_programming.py`
- `TEST_CASES` for validation scenarios
- `SCORING` configuration for point values
- `AGENT_PROMPTS` for specialized analysis criteria

## Troubleshooting

### Common Issues

1. **"Competition system not available"**
   - Check that `competition_ui.py` exists and imports work
   - Install missing dependencies: `pip install -r requirements.txt`

2. **Agent analysis fails**
   - Verify Cerebras API key is set correctly
   - Check internet connection
   - Ensure API quota is available

3. **Database errors**
   - Delete `competition.db` to reset the competition
   - Check write permissions in the directory

### Getting Help

- Check the full documentation in `COMPETITION_README.md`
- Review example solutions in `example_solution.py`
- Test basic functionality with `python example_solution.py`

## Competition Tips for Students

### 🎯 How to Score Maximum Points

1. **Fix All Bugs (50 pts)**
   - Handle `days_active = 0` (division by zero)
   - Use active users count for average calculation
   - Sort engagement scores in descending order
   - Add try/except for missing dictionary keys

2. **Handle Edge Cases (35 pts)**
   - Check for empty user lists
   - Validate required dictionary keys exist
   - Handle different date formats
   - Return appropriate values when no active users

3. **Optimize Performance (25 pts)**
   - Use `heapq.nlargest()` instead of full sort
   - Combine operations in a single loop
   - Eliminate redundant calculations

4. **Secure Your Code (25 pts)**
   - Validate input types and ranges
   - Sanitize user data
   - Prevent code injection vulnerabilities

5. **Submit Fast (20 pts bonus)**
   - Be among the first 3 correct submissions!

### 📚 Learning Resources

- **Python heapq module**: For efficient top-k selection
- **Error handling**: try/except patterns
- **Input validation**: isinstance() and type checking
- **Algorithm optimization**: Big O complexity analysis

---

**Ready to compete? Launch the system and show your coding skills!** 🏆 