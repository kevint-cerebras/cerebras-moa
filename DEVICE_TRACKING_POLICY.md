# Device Tracking Policy for Fair Resubmissions

## Overview

The competitive programming system now implements **device tracking** to ensure fair competition while allowing students to improve their submissions through iteration and learning.

## Policy: Same Device Rewrites

### ✅ **ALLOWED**
- **Multiple submissions from the same device**
- **Iterative improvement** - students can refine their code
- **Learning through trial and error**
- **Best score tracking** - only your highest score counts

### ❌ **BLOCKED**  
- **Submissions from different devices**
- **Device sharing between users**
- **Circumventing submission limits**

## How It Works

### 1. Device Fingerprinting
```
🔍 Device fingerprint: a1b2c3d4... (based on 5 characteristics)
```
The system creates a unique identifier based on:
- Operating system
- Machine type  
- Processor info
- Computer name
- Username

### 2. User-Device Binding
- **First submission**: User is bound to their device
- **Subsequent submissions**: Must come from the same device
- **Violation detection**: Different device attempts are blocked

### 3. Database Tracking
```sql
-- Enhanced submissions table
CREATE TABLE submissions (
    student_name TEXT,
    device_fingerprint TEXT,  -- New: tracks device
    code TEXT,
    total_score INTEGER
);

-- Device tracking table  
CREATE TABLE user_devices (
    student_name TEXT,
    device_fingerprint TEXT,
    submission_count INTEGER,
    first_seen TIMESTAMP
);
```

## Benefits

### 🎯 **Fairness**
- Prevents device sharing abuse
- Ensures individual work
- Maintains competition integrity

### 📈 **Learning**
- Encourages iterative improvement
- Rewards learning and refinement
- Shows progress over time

### 🏆 **Competition**
- Best score per user displayed
- Multiple attempts allowed
- Fair ranking system

## Example Scenarios

### ✅ **Allowed: Alice Improves Her Code**
```
1. Alice submits basic solution from her laptop → Score: 45
2. Alice improves code on same laptop → Score: 78  
3. Alice optimizes further on same laptop → Score: 95
Result: Leaderboard shows Alice with 95 points
```

### ❌ **Blocked: Device Sharing**
```
1. Alice submits from her laptop → Accepted
2. Alice tries to submit from Bob's computer → BLOCKED
Error: "Device validation failed: Submissions must come from the same device"
```

### ❌ **Blocked: Account Sharing**
```
1. Alice submits from Device A → Accepted  
2. Bob tries to submit as "Alice" from Device B → BLOCKED
Error: "Expected device: A, got device: B"
```

## Technical Implementation

### Device Validation Process
```python
def validate_user_device(student_name, device_fingerprint):
    existing_devices = get_user_devices(student_name)
    
    if not existing_devices:
        return {"valid": True, "status": "new_user"}
    
    if device_fingerprint in existing_devices:
        return {"valid": True, "status": "known_device"}
    
    return {
        "valid": False, 
        "status": "device_mismatch",
        "reason": "Submissions must come from the same device"
    }
```

### Leaderboard Display
```
Rank Name           Score %     Submissions Device
---- -------------- ----- ----- ----------- ----------
1    Alice          95    70.4% 3           a1b2c3d4...
2    Bob            78    57.8% 1           x9y8z7w6...
3    Charlie        65    48.1% 2           m5n4o3p2...
```

## Error Messages

### Device Mismatch
```json
{
    "error": "Device validation failed: Submissions must come from the same device",
    "device_fingerprint": "new_device_id",
    "allowed_device": "original_device_id"
}
```

### Success Messages
```json
{
    "submission_id": 123,
    "status": "submitted",
    "device_status": "known_device",
    "device_fingerprint": "a1b2c3d4..."
}
```

## Administrative Features

### User History Tracking
```python
history = system.get_user_submission_history("Alice")
# Shows all submissions with scores and devices
```

### Device Management
```python
# Check user's registered devices
devices = system.get_user_devices("Alice")

# View submission patterns
stats = system.get_device_statistics()
```

## Security Considerations

### Robust Fingerprinting
- Multiple device characteristics
- Resistant to minor system changes
- Stable across sessions

### Privacy Protection  
- Device IDs are hashed
- Only partial IDs displayed
- No personal information stored

### Bypass Prevention
- Server-side validation
- Database integrity checks
- Audit logging

## Migration Strategy

### Existing Users
- First submission after update binds device
- Previous submissions grandfathered in
- Gradual rollout possible

### Database Updates
```sql
-- Automatic schema migration
ALTER TABLE submissions ADD COLUMN device_fingerprint TEXT;
ALTER TABLE leaderboard ADD COLUMN device_fingerprint TEXT;
```

## Monitoring & Analytics

### Metrics Tracked
- Submission attempts per device
- Device mismatch violations
- User improvement patterns
- Competition engagement

### Admin Dashboard
- Device usage statistics
- Violation reports  
- User progress tracking
- System health metrics

## Support & Troubleshooting

### Common Issues

**Q: What if I get a new computer?**
A: Contact administrators to reset your device binding.

**Q: What if I use multiple devices legitimately?**  
A: Current policy allows one device per user. Contact support for exceptions.

**Q: Can I share code with classmates?**
A: Code sharing is separate from device policy. Check academic integrity rules.

### Technical Support
- Device fingerprint issues
- False positive blocks
- Migration assistance
- Account recovery

## Future Enhancements

### Planned Features
- Multiple device support with approval
- Device verification via email/SMS
- Temporary device exceptions
- Enhanced analytics dashboard

### Possible Improvements
- Browser fingerprinting for web interface
- Hardware-based identification
- Machine learning fraud detection
- Real-time collaboration detection

---

## Summary

The device tracking system ensures **fair competition** by:
- ✅ Allowing rewrites from the same device
- ❌ Blocking attempts from different devices  
- 📊 Tracking best scores per user
- 🔒 Maintaining competition integrity

This policy encourages learning while preventing abuse, creating a fair and educational competitive programming environment. 