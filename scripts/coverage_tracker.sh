#!/bin/bash

# Coverage Tracking Dashboard
# Monitors test coverage progress toward 85% goal
# Usage: ./scripts/coverage_tracker.sh [--html] [--detailed]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
TARGET_COVERAGE=85
CURRENT_BASELINE=42
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COVERAGE_DIR="${PROJECT_ROOT}/.coverage_history"
HTML_FLAG=""
DETAILED_FLAG=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --html)
            HTML_FLAG="--html"
            ;;
        --detailed)
            DETAILED_FLAG="--detailed"
            ;;
    esac
done

# Create coverage history directory
mkdir -p "${COVERAGE_DIR}"

echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║           Test Coverage Tracking Dashboard                 ║${NC}"
echo -e "${BOLD}${BLUE}║     Goal: Increase coverage from 42% to 85%                ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Run pytest with coverage
echo -e "${YELLOW}Running pytest with coverage...${NC}"
cd "${PROJECT_ROOT}"

if [ -n "$HTML_FLAG" ]; then
    pytest --cov=src --cov-report=html --cov-report=term-missing --cov-report=json -q
else
    pytest --cov=src --cov-report=term-missing --cov-report=json -q
fi

# Parse JSON coverage report
if [ -f "coverage.json" ]; then
    TOTAL_COVERAGE=$(python3 -c "import json; data=json.load(open('coverage.json')); print(f\"{data['totals']['percent_covered']:.1f}\")")
    TOTAL_STATEMENTS=$(python3 -c "import json; data=json.load(open('coverage.json')); print(data['totals']['num_statements'])")
    COVERED_STATEMENTS=$(python3 -c "import json; data=json.load(open('coverage.json')); print(data['totals']['covered_lines'])")
    MISSING_STATEMENTS=$(python3 -c "import json; data=json.load(open('coverage.json')); print(data['totals']['missing_lines'])")

    # Save to history
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${TIMESTAMP},${TOTAL_COVERAGE}" >> "${COVERAGE_DIR}/history.csv"

    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}                    COVERAGE SUMMARY                         ${NC}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Calculate progress
    COVERAGE_GAIN=$(python3 -c "print(f\"{${TOTAL_COVERAGE} - ${CURRENT_BASELINE}:.1f}\")")
    REMAINING_GAIN=$(python3 -c "print(f\"{${TARGET_COVERAGE} - ${TOTAL_COVERAGE}:.1f}\")")
    PROGRESS_PERCENT=$(python3 -c "print(f\"{(${TOTAL_COVERAGE} - ${CURRENT_BASELINE}) / (${TARGET_COVERAGE} - ${CURRENT_BASELINE}) * 100:.1f}\")")

    # Display current coverage
    if (( $(echo "$TOTAL_COVERAGE >= $TARGET_COVERAGE" | bc -l) )); then
        echo -e "Current Coverage: ${GREEN}${BOLD}${TOTAL_COVERAGE}%${NC} ✓ ${GREEN}TARGET REACHED!${NC}"
    elif (( $(echo "$TOTAL_COVERAGE >= 75" | bc -l) )); then
        echo -e "Current Coverage: ${GREEN}${TOTAL_COVERAGE}%${NC} (Good progress!)"
    elif (( $(echo "$TOTAL_COVERAGE >= 60" | bc -l) )); then
        echo -e "Current Coverage: ${YELLOW}${TOTAL_COVERAGE}%${NC} (Making progress)"
    else
        echo -e "Current Coverage: ${RED}${TOTAL_COVERAGE}%${NC} (Needs work)"
    fi

    echo -e "Target Coverage:  ${BOLD}${TARGET_COVERAGE}%${NC}"
    echo -e "Baseline:         ${CURRENT_BASELINE}%"
    echo ""
    echo -e "Progress:         ${COVERAGE_GAIN}% gained (${REMAINING_GAIN}% remaining)"
    echo -e "Completion:       ${PROGRESS_PERCENT}% of goal"
    echo ""
    echo -e "Statements:       ${COVERED_STATEMENTS}/${TOTAL_STATEMENTS} covered (${MISSING_STATEMENTS} missing)"
    echo ""

    # Progress bar
    PROGRESS_BARS=$(python3 -c "print(int(${PROGRESS_PERCENT} / 2))")
    EMPTY_BARS=$((50 - PROGRESS_BARS))
    echo -n "Progress: ["
    for i in $(seq 1 $PROGRESS_BARS); do echo -n "█"; done
    for i in $(seq 1 $EMPTY_BARS); do echo -n "░"; done
    echo -e "] ${PROGRESS_PERCENT}%"
    echo ""

    # Module breakdown
    if [ -n "$DETAILED_FLAG" ]; then
        echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
        echo -e "${BOLD}              MODULE-BY-MODULE BREAKDOWN                     ${NC}"
        echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
        echo ""

        python3 << 'EOF'
import json
with open('coverage.json') as f:
    data = json.load(f)

# Define module priorities and targets
priorities = {
    'src/training/trainer.py': {'target': 75, 'priority': 1},
    'src/evaluation/visualization.py': {'target': 75, 'priority': 2},
    'src/evaluation/metrics.py': {'target': 80, 'priority': 2},
    'src/models/lstm_conditioned.py': {'target': 70, 'priority': 3},
    'src/data/signal_generator.py': {'target': 80, 'priority': 3},
}

print(f"{'Module':<45} {'Coverage':<12} {'Target':<10} {'Status':<10} {'Priority'}")
print("─" * 95)

for file_path, file_data in sorted(data['files'].items(),
                                   key=lambda x: x[1]['summary']['percent_covered']):
    if file_path.startswith('src/') and not file_path.endswith('__init__.py'):
        coverage = file_data['summary']['percent_covered']

        # Get target and priority
        target = priorities.get(file_path, {}).get('target', 85)
        priority = priorities.get(file_path, {}).get('priority', 4)

        # Determine status
        if coverage >= target:
            status = "✓ DONE"
            color_code = "\033[0;32m"  # Green
        elif coverage >= target - 10:
            status = "CLOSE"
            color_code = "\033[1;33m"  # Yellow
        else:
            status = "NEEDS WORK"
            color_code = "\033[0;31m"  # Red

        priority_str = "★" * priority

        print(f"{file_path:<45} {color_code}{coverage:>6.1f}%\033[0m      {target:>3}%       {status:<10} {priority_str}")

print("─" * 95)
print("\nPriority Legend: ★ = High (Phase 1), ★★ = Medium (Phase 2), ★★★ = Lower (Phase 3)")
EOF
        echo ""
    fi

    # Show recent history
    if [ -f "${COVERAGE_DIR}/history.csv" ]; then
        echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
        echo -e "${BOLD}                  COVERAGE HISTORY                           ${NC}"
        echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
        echo ""
        tail -10 "${COVERAGE_DIR}/history.csv" | while IFS=',' read -r timestamp coverage; do
            echo -e "${timestamp}: ${coverage}%"
        done
        echo ""
    fi

    # Estimate remaining work
    echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}              REMAINING WORK ESTIMATE                        ${NC}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
    echo ""

    REMAINING_STATEMENTS=$(python3 -c "import math; total=${TOTAL_STATEMENTS}; target=${TARGET_COVERAGE}/100; covered=${COVERED_STATEMENTS}; needed = math.ceil(total * target - covered); print(max(0, needed))")
    ESTIMATED_TESTS=$(python3 -c "print(int(${REMAINING_STATEMENTS} / 5))")  # Assume 1 test covers ~5 statements
    ESTIMATED_HOURS=$(python3 -c "print(f\"{${ESTIMATED_TESTS} * 0.5:.1f}\")")  # Assume 30 min per test

    echo -e "Statements to cover: ${REMAINING_STATEMENTS}"
    echo -e "Estimated tests needed: ~${ESTIMATED_TESTS}"
    echo -e "Estimated time: ~${ESTIMATED_HOURS} hours"
    echo ""

    # Recommendations
    echo -e "${BOLD}Recommendations:${NC}"
    if (( $(echo "$TOTAL_COVERAGE < 60" | bc -l) )); then
        echo -e "  1. Start with ${BOLD}trainer.py${NC} tests (highest impact)"
        echo -e "  2. Use test templates in ${BOLD}tests/test_trainer_extended.py${NC}"
        echo -e "  3. Focus on checkpoint and training loop tests first"
    elif (( $(echo "$TOTAL_COVERAGE < 75" | bc -l) )); then
        echo -e "  1. Continue with ${BOLD}metrics.py${NC} and ${BOLD}visualization.py${NC}"
        echo -e "  2. Use templates in ${BOLD}tests/test_metrics_extended.py${NC}"
        echo -e "  3. Test plotting functions with 'Agg' backend"
    else
        echo -e "  1. Focus on remaining models and edge cases"
        echo -e "  2. Target ${BOLD}lstm_conditioned.py${NC} and ${BOLD}data modules${NC}"
        echo -e "  3. Add integration tests for full pipeline"
    fi
    echo ""

    # HTML report link
    if [ -n "$HTML_FLAG" ]; then
        echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}HTML report generated: file://${PROJECT_ROOT}/htmlcov/index.html${NC}"
        echo -e "Open in browser to see detailed line-by-line coverage"
        echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
        echo ""
    fi

else
    echo -e "${RED}Error: coverage.json not found${NC}"
    exit 1
fi

# Cleanup
rm -f coverage.json .coverage
