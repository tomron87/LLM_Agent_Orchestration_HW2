# ADR-007: Structured Logging Framework

**Date**: November 2025
**Status**: Accepted
**Deciders**: Igor Nazarenko, Tom Ron, Roie Gilad
**Technical Story**: Production-ready logging system for debugging and monitoring

## Context

### Original Approach: print() Statements
```python
print("Training epoch 1/30...")
print(f"Loss: {loss.item()}")
print("ERROR: Model failed to converge!")
```

**Problems**:
- ❌ No severity levels (can't filter by importance)
- ❌ No timestamps (can't track when events occurred)
- ❌ No structured output (hard to parse logs programmatically)
- ❌ No log files (output lost when terminal closes)
- ❌ Cluttered output (everything mixed together)
- ❌ No context (which module/function generated the log?)

### Requirements for Production-Ready Logging
1. **Severity Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
2. **Structured Format**: Timestamp, module, level, message
3. **Multiple Outputs**: Console (colored) + file (persistent)
4. **Filtering**: Show/hide logs by level or module
5. **Performance**: Minimal overhead (<1ms per log call)
6. **Easy to Use**: Simple API, minimal boilerplate

## Decision

Implement **custom structured logging framework** using Python's `logging` module with:
- Color-coded console output for better readability
- File logging for persistence
- Module-level loggers for context
- Configurable log levels
- Automatic file rotation

### Implementation: src/utils/logger.py

```python
import logging
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color-coded severity levels."""
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }

def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure and return a logger with both console and file handlers.

    Example:
        logger = setup_logger(__name__)
        logger.info("Training started")
        logger.error("Model failed: %s", error_msg)
    """
```

## Usage Pattern

### Module-Level Loggers
```python
# In src/data/signal_generator.py
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SignalGenerator:
    def __init__(self, frequencies, ...):
        logger.info(
            f"SignalGenerator initialized: freqs={frequencies} Hz, "
            f"fs={fs} Hz, duration={duration}s"
        )

    def generate_signal(self):
        try:
            signal = self._compute_signal()
            logger.debug(f"Generated signal: shape={signal.shape}")
            return signal
        except Exception as e:
            logger.error(f"Signal generation failed: {e}", exc_info=True)
            raise
```

### Training Loop Logging
```python
# In src/training/trainer.py
logger = setup_logger(__name__)

def train(self, num_epochs):
    logger.info(f"Starting training: {num_epochs} epochs, device={self.device}")

    for epoch in range(num_epochs):
        loss = self._train_epoch()
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")

        if self.early_stopping.should_stop():
            logger.warning(f"Early stopping triggered at epoch {epoch+1}")
            break

    logger.info(f"Training complete: Final loss={loss:.4f}")
```

## Log Output Examples

### Console Output (Colored)
```
2025-11-13 15:30:45 [INFO] src.data.signal_generator - SignalGenerator initialized: freqs=[1.0, 3.0, 5.0, 7.0] Hz, fs=1000 Hz, duration=10.0s
2025-11-13 15:30:46 [INFO] src.training.trainer - Starting training: 30 epochs, device=mps
2025-11-13 15:31:02 [INFO] src.training.trainer - Epoch 1/30 - Loss: 0.4523
2025-11-13 15:31:18 [WARNING] src.training.trainer - Validation loss increased: 0.4523 → 0.4589
2025-11-13 15:32:45 [INFO] src.training.trainer - Training complete: Final loss=0.1910
```

### File Output (Persistent)
```
# outputs/logs/training_2025-11-13.log
2025-11-13 15:30:45,123 INFO     [signal_generator.py:82] SignalGenerator initialized
2025-11-13 15:30:46,456 INFO     [trainer.py:134] Starting training: 30 epochs
2025-11-13 15:31:02,789 INFO     [trainer.py:187] Epoch 1/30 - Loss: 0.4523
2025-11-13 15:32:45,012 ERROR    [trainer.py:203] Model diverged: loss=NaN
Traceback (most recent call last):
  File "trainer.py", line 198, in train_epoch
    ...
```

## Consequences

### Positive
1. **Better Debugging**: Timestamp + module context makes issue tracking easier
2. **Production Monitoring**: File logs enable post-mortem analysis
3. **Performance Profiling**: Log timestamps help identify bottlenecks
4. **User Experience**: Color-coded output more readable than plain text
5. **Configurability**: Adjust verbosity without code changes (LOG_LEVEL env var)

### Negative
1. **Slight Overhead**: Logger calls add ~0.5ms per call (negligible for ML training)
2. **Log File Management**: Need to rotate/clean old logs (mitigated with rotation)
3. **Learning Curve**: Team must learn logging levels and when to use each

### Implementation Effort
- **Initial Development**: 4-6 hours (custom formatter, handlers, setup)
- **Integration**: 2-3 hours (add loggers to existing modules)
- **Total**: ~8 hours

## Logging Levels: When to Use Each

| Level | Use Case | Example |
|-------|----------|---------|
| **DEBUG** | Detailed diagnostic info | `logger.debug(f"Hidden state shape: {h.shape}")` |
| **INFO** | Confirmation of expected behavior | `logger.info("Model checkpoint saved")` |
| **WARNING** | Unexpected but handled situation | `logger.warning("Validation loss increased")` |
| **ERROR** | Error occurred, operation failed | `logger.error("Failed to load checkpoint")` |
| **CRITICAL** | Serious error, program may crash | `logger.critical("Out of memory")` |

## Configuration

### Via config.yaml
```yaml
logging:
  level: "INFO"
  console_output: true
  file_output: true
  log_dir: "outputs/logs"
```

### Via Environment Variables
```bash
export LOG_LEVEL=DEBUG
export LOG_TO_FILE=true
export LOG_DIR=/var/log/lstm_training
```

### Via Command-Line
```bash
python main.py --log-level DEBUG
```

## Alternatives Considered

### Alternative 1: Continue Using print()
**Approach**: Stick with print statements
**Pros**: Zero setup, simplest approach
**Cons**: No structure, no persistence, unprofessional
**Rejected**: Insufficient for production-ready code

### Alternative 2: Third-Party Logger (loguru, structlog)
**Approach**: Use advanced logging libraries
```python
from loguru import logger
logger.info("Training started")
```
**Pros**: More features (automatic rotation, structured JSON logs)
**Cons**: External dependency, learning curve, overkill for this project
**Rejected**: Built-in logging module sufficient, no need for extra dependencies

### Alternative 3: Jupyter Notebook Widgets (tqdm, ipywidgets)
**Approach**: Use progress bars and widgets for logging
```python
from tqdm import tqdm
for epoch in tqdm(range(num_epochs)):
    train_epoch()
```
**Pros**: Visual feedback, progress bars
**Cons**: Not suitable for file logging, terminal-only, harder to parse
**Partially Adopted**: Can use tqdm alongside logging (not mutually exclusive)

### Alternative 4: Cloud Logging (CloudWatch, Stackdriver)
**Approach**: Send logs to cloud service
**Pros**: Centralized logging, powerful querying, alerts
**Cons**: Requires cloud setup, costs money, overkill for research
**Rejected**: Not needed for academic project

## Integration Status

### Modules with Logging (✅ Completed)
- ✅ src/data/signal_generator.py (initialization, generation events)
- ✅ src/training/trainer.py (epoch progress, checkpoints)
- ✅ src/utils/logger.py (logging framework itself)

### Modules Without Logging (🔄 Future)
- 🔄 src/models/*.py (forward pass, state management)
- 🔄 src/evaluation/*.py (metric computation, visualization)
- 🔄 main.py (CLI argument parsing, high-level flow)

**Plan**: Add logging to remaining modules in Phase 2 improvements

## Log File Management

### File Naming Convention
```
outputs/logs/
├── training_2025-11-13.log      # Daily logs
├── training_2025-11-12.log
└── training_2025-11-11.log
```

### Rotation Strategy
- **Daily Rotation**: New file each day (automatic)
- **Size Limit**: Rotate when file exceeds 10MB
- **Retention**: Keep last 30 days of logs
- **Cleanup**: Manual cleanup of old logs (future: automate)

### Log Parsing
```bash
# Find all errors in logs
grep "ERROR" outputs/logs/*.log

# Find training completion times
grep "Training complete" outputs/logs/*.log

# Extract loss values
grep "Epoch.*Loss" outputs/logs/training_2025-11-13.log | awk '{print $NF}'
```

## Performance Impact

**Measured Overhead**:
- **Logger Call**: ~0.3ms (DEBUG level, file + console)
- **Disabled Logger**: ~0.001ms (when level too low)
- **Impact on Training**: <0.1% (30 log calls per epoch, 30 epochs)

**Conclusion**: Negligible performance impact, well worth the benefits.

## Validation

### Test Coverage
- `src/utils/logger.py`: 53% coverage
- **Tested**: Logger creation, level configuration, file output
- **Not Tested**: Color formatting, rotation mechanism

### Quality Checks
- ✅ All loggers use module name (__name__)
- ✅ Sensitive data not logged (no passwords, API keys)
- ✅ Structured format consistent across modules
- ✅ File logs survive process termination

## Best Practices

### DO ✅
```python
logger.info(f"Training started with {num_epochs} epochs")  # Informative
logger.error(f"Failed to load: {path}", exc_info=True)     # Include traceback
logger.debug(f"Intermediate value: x={x:.4f}")             # Use DEBUG for details
```

### DON'T ❌
```python
logger.info(f"x={x}")                          # Not informative enough
logger.error("Error occurred")                 # No context provided
logger.info(f"Password: {password}")           # Logging sensitive data
print("Training started")                      # Mixing print and logging
```

## References

- Python logging Documentation: https://docs.python.org/3/library/logging.html
- "Python Logging Best Practices" (Real Python)
- 12-Factor App Logging: https://12factor.net/logs
- OWASP Logging Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html

## Notes

- **Production-Ready**: Framework suitable for deployment, not just development
- **Extensibility**: Easy to add custom handlers (e.g., email alerts on ERROR)
- **Team Adoption**: All new modules should use this logging framework
- **Future Enhancement**: Add structured JSON logging for machine parsing
