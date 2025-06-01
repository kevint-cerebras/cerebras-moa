import streamlit as st
import time
import asyncio
import threading
from streamlit_ace import st_ace
from competitive_programming import CompetitiveProgrammingSystem, ORIGINAL_FUNCTION, TEST_CASES, SCORING, validate_and_execute_code, SecureExecutionError
import pandas as pd
import sqlite3

def init_competition_session():
    """Initialize competition system in session state"""
    if 'competition_system' not in st.session_state:
        st.session_state.competition_system = CompetitiveProgrammingSystem()
    
    if 'specialized_agents' not in st.session_state:
        st.session_state.specialized_agents = st.session_state.competition_system.create_specialized_agents()
    
    if 'analysis_queue' not in st.session_state:
        st.session_state.analysis_queue = []
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True

def render_competition_page():
    """Main competition page with tabs for different sections"""
    init_competition_session()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .leaderboard-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        .score-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        .submission-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .stAlert > div {
            padding: 1rem;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1>🏆 Code Competition Arena</h1>
        <h3>Fix the Buggy Function Challenge</h3>
        <p>Maximum Score: 135 points | AI-powered grading with comprehensive analysis! 🚀</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Challenge", "📝 Submit Solution", "🏆 Live Leaderboard", "ℹ️ Scoring Guide"
    ])
    
    with tab1:
        render_challenge_tab()
    
    with tab2:
        render_submission_tab()
    
    with tab3:
        render_leaderboard_tab()
    
    with tab4:
        render_scoring_guide()

def render_challenge_tab():
    """Render the challenge description and original code"""
    st.markdown("### 🎯 Your Mission: Fix the Buggy Function")
    st.markdown("""
    **Challenge:** The function below has multiple bugs, edge cases, performance issues, and security vulnerabilities. 
    Your goal is to fix them all and submit your solution for AI-powered grading.
    
    **Focus on:** Code functionality, bug fixes, and performance - comments and documentation are optional!
    """)
    
    # Show the buggy function prominently
    st.markdown("#### 🐛 The Buggy Function (DO NOT EDIT HERE)")
    st.code(ORIGINAL_FUNCTION, language='python')
    
    st.info("👆 This is the buggy function. Go to the **'Submit Solution'** tab to edit and fix it!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Known Issues to Fix")
        st.markdown("""
        **Bugs & Edge Cases:**
        - ❌ Division by zero errors
        - ❌ Wrong calculation denominators  
        - ❌ Incorrect sorting directions
        - ❌ Missing key handling
        - ❌ Empty input scenarios
        
        **Performance Issues:**
        - ⚠️ Inefficient sorting algorithms
        - ⚠️ Unnecessary operations
        - ⚠️ Multiple loops where one would suffice
        
        **Security Concerns:**
        - 🔒 Input validation missing
        - 🔒 No data sanitization
        - 🔒 Potential injection risks
        """)
    
    with col2:
        st.markdown("### 🏅 Scoring Breakdown")
        st.markdown("""
        **Maximum: 135 points**
        
        📊 **Category Scores:**
        - 🐛 Bug Detection: **50 pts**
        - 🎯 Edge Cases: **35 pts**  
        - ⚡ Performance: **25 pts**
        - 🔒 Security: **25 pts**
        
        **Note:** Comments and documentation are automatically removed and don't affect scoring!
        """)
        
        st.markdown("### 📋 Test Cases")
        with st.expander("📝 View Test Cases"):
            for i, test_case in enumerate(TEST_CASES):
                st.markdown(f"**Test Case {i+1}: {test_case['name']}**")
                st.json(test_case)
                st.markdown("---")

def render_submission_tab():
    """Render the code submission interface"""
    st.markdown("### 📝 Edit and Submit Your Solution")
    st.markdown("Fix the bugs in the function below and submit for AI-powered grading!")
    
    # Add info about fair grading (without mentioning injection)
    st.info("""
    🤖 **AI-Powered Fair Grading**: 
    - Comments are automatically removed from your code before analysis
    - Focus on functionality, not documentation - comments won't affect your score
    - Your code logic is preserved while ensuring fair and secure grading
    """)
    
    # Add info about allowed imports
    st.success("""
    📦 **Expanded Import Support**: You can now use these modules in your solutions:
    - `typing` (List, Dict, Optional, etc.) for type hints
    - `logging` for debugging and monitoring  
    - `sys` (safe attributes only) for system info
    - `dateutil` for robust date parsing
    - `math` (expanded functions) for calculations
    - `heapq`, `datetime`, `json` (as before)
    """)
    
    # Student name input - more prominent
    student_name = st.text_input(
        "🎓 Your Name", 
        placeholder="Enter your name for the leaderboard",
        help="This will appear on the leaderboard"
    )
    
    # Code editor - larger and more prominent
    st.markdown("#### ✏️ Code Editor")
    st.markdown("Edit the function below to fix all the bugs:")
    
    code = st_ace(
        value=ORIGINAL_FUNCTION,
        language='python',
        theme='monokai',
        key="code_editor",
        height=500,  # Increased height
        auto_update=True,  # Changed to True for automatic updates
        font_size=16,  # Larger font
        tab_size=4,
        show_gutter=True,
        show_print_margin=True,
        wrap=False,
        annotations=None,
        markers=None
    )
    
    # Action buttons - more prominent
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🧪 Test Your Code", use_container_width=True, type="secondary"):
            if code.strip():
                test_code_locally(code)
            else:
                st.error("Please write some code first!")
    
    with col2:
        if st.button("🚀 Submit for Grading", type="primary", use_container_width=True):
            if not student_name.strip():
                st.error("⚠️ Please enter your name before submitting!")
            elif not code.strip():
                st.error("⚠️ Please write your solution before submitting!")
            else:
                st.success("🔄 Submitting your solution for AI analysis...")
                submit_solution(student_name.strip(), code)
    
    # Help section - compact
    with st.expander("💡 Need Help? Click Here"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **🔧 Common Fixes:**
            - Add `try/except` blocks for error handling
            - Check for empty inputs first
            - Validate dictionary keys before accessing
            - Use `heapq.nlargest()` for top-k elements
            - Add input type validation
            """)
        
        with col2:
            st.markdown("""
            **⚡ Performance & Tools:**
            - Use `from typing import List, Dict` for type hints
            - Use `dateutil.parser.parse()` for flexible date parsing
            - Use `logging` for debugging (won't affect scoring)
            - Choose efficient data structures
            - Focus on code logic, not comments
            """)
    
    # Quick actions
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Reset to Original", use_container_width=True):
            st.session_state.code_editor = ORIGINAL_FUNCTION
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Editor", use_container_width=True):
            st.session_state.code_editor = ""
            st.rerun()
            
    with col3:
        if st.button("📋 Copy Template", use_container_width=True):
            st.code(ORIGINAL_FUNCTION, language='python')
            st.info("👆 Copy this code into the editor above")
    
    # Show last analysis results if available
    if 'last_analysis' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Your Latest Analysis Results")
        display_analysis_results(st.session_state.last_analysis)
        
        # Option to clear results
        if st.button("🗑️ Clear Results", key="clear_analysis"):
            del st.session_state.last_analysis
            st.rerun()

def submit_solution(student_name: str, code: str):
    """Handle solution submission"""
    system = st.session_state.competition_system
    
    # Submit solution
    result = system.submit_solution(student_name, code)
    
    if "error" in result:
        st.error(f"❌ {result['error']}")
    else:
        submission_id = result["submission_id"]
        
        # Store submission info in session state
        st.session_state.last_submission = {
            'id': submission_id,
            'student_name': student_name,
            'timestamp': time.time()
        }
        
        # Show submission confirmation
        st.success(f"✅ Solution submitted successfully! Submission ID: {submission_id}")
        
        # Start analysis in background
        with st.spinner("🔍 Analyzing your solution with specialized agents..."):
            try:
                analysis_result = system.analyze_submission(
                    submission_id, 
                    st.session_state.specialized_agents
                )
                
                if "error" not in analysis_result:
                    # Store analysis results in session state
                    st.session_state.last_analysis = analysis_result
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display results immediately
                    display_analysis_results(analysis_result)
                else:
                    st.error(f"Analysis failed: {analysis_result['error']}")
                    
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

def display_analysis_results(analysis_result):
    """Display detailed analysis results"""
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    
    # Check if validation failed
    if analysis_result.get('validation_failed', False):
        st.error("❌ Code Validation Failed - AI Analysis Skipped")
        
        # Display validation details
        validation_results = analysis_result.get('validation_results', {})
        
        with st.expander("🧪 Validation Details", expanded=True):
            st.markdown("#### 📝 Code Validation Results")
            
            col1, col2 = st.columns(2)
            with col1:
                syntax_status = "✅ Pass" if validation_results.get('syntax_valid', False) else "❌ Fail"
                st.metric("Syntax Check", syntax_status)
                
                tests_passed = validation_results.get('passed_tests', 0)
                total_tests = validation_results.get('total_tests', 0)
                st.metric("Unit Tests", f"{tests_passed}/{total_tests}")
            
            with col2:
                overall_status = "✅ Pass" if validation_results.get('passes_tests', False) else "❌ Fail"
                st.metric("Overall Status", overall_status)
            
            # Show specific test results
            if validation_results.get('test_results'):
                st.markdown("#### 🔍 Test Case Results")
                for test in validation_results['test_results']:
                    status_icon = "✅" if test['passed'] else "❌"
                    st.write(f"{status_icon} **{test['test_name']}**")
                    if test['error']:
                        st.error(f"Error: {test['error']}")
                    elif test['output']:
                        with st.expander(f"Output for {test['test_name']}", expanded=False):
                            st.json(test['output'])
            
            # Show runtime errors if any
            if validation_results.get('runtime_errors'):
                st.markdown("#### ❌ Runtime Errors")
                for error in validation_results['runtime_errors']:
                    st.error(error)
        
        st.info("💡 **Fix the validation errors above and resubmit to get AI analysis and scoring!**")
        return
    
    # Display validation results if analysis succeeded
    validation_results = analysis_result.get('validation_results', {})
    if validation_results:
        with st.expander("🧪 Code Validation Results", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Syntax Check", "Pass")
            with col2:
                tests_passed = validation_results.get('passed_tests', 0)
                total_tests = validation_results.get('total_tests', 0)
                st.metric("🧪 Unit Tests", f"{tests_passed}/{total_tests}")
            with col3:
                st.metric("🎯 Validation", "✅ Pass")
            
            # Show AI processing info
            st.success("🤖 **AI Processing**: Comments automatically removed, code analyzed by specialized AI agents")
    
    # Score summary - make it full width with larger metrics
    st.markdown("#### 🎯 Score Breakdown")
    
    # Create wider columns for better visibility
    col1, col2 = st.columns(2)
    
    with col1:
        bug_score = analysis_result['analysis_results'].get('bug_hunter', {}).get('score', 0)
        st.metric("🐛 Bug Detection", f"{bug_score}/50", help="Identified and fixed code bugs")
        
        perf_score = analysis_result['analysis_results'].get('performance_agent', {}).get('score', 0)
        st.metric("⚡ Performance", f"{perf_score}/25", help="Code optimization and efficiency")
    
    with col2:
        edge_score = analysis_result['analysis_results'].get('edge_case_checker', {}).get('score', 0)
        st.metric("🎯 Edge Cases", f"{edge_score}/35", help="Handled edge cases and boundary conditions")
        
        sec_score = analysis_result['analysis_results'].get('security_agent', {}).get('score', 0)
        st.metric("🔒 Security", f"{sec_score}/25", help="Input validation and security measures")
    
    # Total score - full width and prominent
    total_score = analysis_result['total_score']
    percentage = round((total_score / 135) * 100, 1)  # Updated to 135 max score
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;'>
        <h1>🏆 Total Score: {total_score}/135 ({percentage}%)</h1>
        <p style='font-size: 1.2rem; margin: 0;'>Student: {analysis_result.get('student_name', 'Unknown')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed analysis per agent - full width expandable sections
    st.markdown("#### 🔍 Detailed Agent Analysis")
    
    agent_names = {
        'bug_hunter': '🐛 Bug Hunter Agent',
        'edge_case_checker': '🎯 Edge Case Checker', 
        'performance_agent': '⚡ Performance Agent',
        'security_agent': '🔒 Security Agent'
    }
    
    for agent_type, analysis in analysis_result['analysis_results'].items():
        if "error" not in analysis:
            agent_display_name = agent_names.get(agent_type, agent_type.replace('_', ' ').title())
            with st.expander(f"📋 {agent_display_name} - Scored {analysis.get('score', 0)} points", expanded=False):
                # Display the analysis in a more readable format
                st.json(analysis)
        else:
            agent_display_name = agent_names.get(agent_type, agent_type.replace('_', ' ').title())
            st.error(f"❌ {agent_display_name}: {analysis['error']}")

def test_code_locally(code: str):
    """Test the code with predefined test cases using secure execution"""
    st.markdown("### 🧪 Local Test Results")
    
    try:
        # Import the secure execution function
        from competitive_programming import validate_and_execute_code, SecureExecutionError
        
        # Validate and extract function securely
        try:
            func, _ = validate_and_execute_code(code)
            st.success("✅ Security validation passed!")
        except SecureExecutionError as e:
            st.error(f"❌ Security Error: {str(e)}")
            st.warning("🚨 **Your code contains dangerous operations!** Please remove them and try again.")
            return
        
        # Test each case with secure execution
        for i, test_case in enumerate(TEST_CASES):
            st.markdown(f"**Test Case {i+1}: {test_case['name']}**")
            
            try:
                result = func(
                    test_case['users'], 
                    test_case['start_date'], 
                    test_case['end_date']
                )
                st.success("✅ Test passed!")
                st.json(result)
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
            
            st.markdown("---")
                
    except Exception as e:
        st.error(f"❌ Code validation failed: {str(e)}")

def render_leaderboard_tab():
    """Render live leaderboard with auto-refresh"""
    st.markdown("### 🏆 Live Leaderboard")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        auto_refresh = st.checkbox("Auto Refresh (every 10 seconds)", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
    
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    with col3:
        if st.button("🔧 Rebuild Leaderboard"):
            with st.spinner("Rebuilding leaderboard..."):
                success = st.session_state.competition_system.rebuild_leaderboard()
                if success:
                    st.success("Leaderboard rebuilt successfully!")
                    st.rerun()
                else:
                    st.error("Failed to rebuild leaderboard")
    
    # Auto-refresh mechanism
    if auto_refresh:
        # Display countdown
        placeholder = st.empty()
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        time_since_refresh = time.time() - st.session_state.last_refresh
        
        if time_since_refresh >= 10:  # 10 seconds
            st.session_state.last_refresh = time.time()
            st.rerun()
        else:
            countdown = 10 - int(time_since_refresh)
            placeholder.info(f"Auto-refresh in {countdown} seconds...")
    
    # Get leaderboard data
    try:
        leaderboard = st.session_state.competition_system.get_leaderboard()
        
        # Debug information
        st.write(f"Debug: Found {len(leaderboard)} leaderboard entries")
        
        # Also check submissions count for debugging
        try:
            # Let's try to get submission count directly
            db_path = st.session_state.competition_system.db_path
            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM submissions')
                submission_count = cursor.fetchone()[0]
                cursor.execute('SELECT COUNT(*) FROM leaderboard')
                leaderboard_count = cursor.fetchone()[0]
                st.write(f"Debug: {submission_count} submissions, {leaderboard_count} leaderboard entries in database")
        except Exception as debug_error:
            st.write(f"Debug error: {str(debug_error)}")
            
    except Exception as e:
        st.error(f"Error loading leaderboard: {str(e)}")
        return
    
    if not leaderboard:
        st.info("No submissions yet. Be the first to submit!")
        return
    
    # Display top 10 leaderboard in simple format
    st.markdown("#### 🥇 Top 10 Rankings")
    
    # Take only top 10
    top_10 = leaderboard[:10]
    
    for entry in top_10:
        position = entry['position']
        name = entry['student_name']
        score = entry['best_score']
        percentage = round((score / 135) * 100, 1)
        
        # Medal icons for top 3
        if position == 1:
            medal = "🥇"
        elif position == 2:
            medal = "🥈"
        elif position == 3:
            medal = "🥉"
        else:
            medal = f"#{position}"
        
        # Simple leaderboard entry
        st.markdown(f"""
        <div style='padding: 1rem; margin: 0.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <h3>{medal} {name}</h3>
                <p>{percentage}% • {score}/135 points</p>
            </div>
            <div style='text-align: right; font-size: 2rem; font-weight: bold;'>
                {score}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_scoring_guide():
    """Render detailed scoring guide"""
    st.markdown("### ℹ️ Detailed Scoring Guide")
    
    st.markdown("""
    This competition uses a comprehensive scoring system with specialized AI agents 
    analyzing different aspects of your code.
    """)
    
    # Bug Detection
    st.markdown("#### 🐛 Bug Detection (50 points)")
    bug_df = pd.DataFrame([
        {"Issue": "Division by Zero", "Points": 15, "Description": "Handling days_active = 0"},
        {"Issue": "Wrong Denominator", "Points": 15, "Description": "Using active users vs all users"},
        {"Issue": "Sorting Direction", "Points": 10, "Description": "Correct descending sort"},
        {"Issue": "KeyError Handling", "Points": 10, "Description": "Missing dictionary keys"}
    ])
    st.dataframe(bug_df, use_container_width=True)
    
    # Edge Cases
    st.markdown("#### 🎯 Edge Cases (35 points)")
    edge_df = pd.DataFrame([
        {"Case": "Empty Input", "Points": 10, "Description": "users = []"},
        {"Case": "Missing Keys", "Points": 10, "Description": "Required keys not present"},
        {"Case": "Date Formats", "Points": 8, "Description": "Different date string formats"},
        {"Case": "No Active Users", "Points": 7, "Description": "No users in date range"}
    ])
    st.dataframe(edge_df, use_container_width=True)
    
    # Performance
    st.markdown("#### ⚡ Performance (25 points)")
    perf_df = pd.DataFrame([
        {"Optimization": "Sorting Algorithm", "Points": 12, "Description": "Use heapq.nlargest()"},
        {"Optimization": "Single Loop", "Points": 8, "Description": "Combine operations in one loop"},
        {"Optimization": "Remove Unnecessary Operations", "Points": 5, "Description": "Eliminate redundant calculations"}
    ])
    st.dataframe(perf_df, use_container_width=True)
    
    # Security
    st.markdown("#### 🔒 Security (25 points)")
    sec_df = pd.DataFrame([
        {"Aspect": "Input Validation", "Points": 10, "Description": "Type and range checking"},
        {"Aspect": "Data Sanitization", "Points": 8, "Description": "Clean input data"},
        {"Aspect": "Injection Risks", "Points": 7, "Description": "Prevent code injection"}
    ])
    st.dataframe(sec_df, use_container_width=True)
    
    st.markdown("**Maximum Possible Score: 135 points**") 