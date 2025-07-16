#!/bin/bash

# HLS Synthesis Script for ViT Implementation
# This script sets up and runs Vivado HLS synthesis

PROJECT_NAME="vit_hls_project"
SOLUTION_NAME="solution1"
TOP_FUNCTION="vit_inference"
PART="xcu280-fsvh2892-2L-e"  # Xilinx Alveo U280 (can be changed)
CLOCK_PERIOD=10  # 100MHz clock

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting ViT HLS Synthesis${NC}"
echo "Project: $PROJECT_NAME"
echo "Solution: $SOLUTION_NAME"
echo "Top Function: $TOP_FUNCTION"
echo "Target Part: $PART"
echo "Clock Period: ${CLOCK_PERIOD}ns"
echo ""

# Check if Vivado HLS is available
if ! command -v vitis_hls &> /dev/null; then
    echo -e "${RED}Error: vitis_hls not found. Please source Vivado/Vitis environment.${NC}"
    exit 1
fi

# Create HLS TCL script
cat > run_hls.tcl << EOF
# HLS Project Setup
open_project $PROJECT_NAME

# Set top function
set_top $TOP_FUNCTION

# Add source files
add_files src/vit.cpp
add_files src/vit_layers.cpp
add_files -tb test/test_vit.cpp

# Open solution
open_solution "$SOLUTION_NAME"

# Set target part
set_part {$PART}

# Create clock constraint
create_clock -period $CLOCK_PERIOD -name default

# Configuration flags
config_compile -name_max_length 256
config_schedule -relax_ii_for_timing
config_bind -effort high

# Run C simulation
puts "Running C simulation..."
csim_design

# Run synthesis
puts "Running C synthesis..."
csynth_design

# Run C/RTL co-simulation
puts "Running C/RTL co-simulation..."
cosim_design -trace_level all

# Export design
puts "Exporting design..."
export_design -format ip_catalog

# Generate reports
puts "Generating reports..."
puts "Synthesis report:"
puts [exec cat $PROJECT_NAME/$SOLUTION_NAME/syn/report/${TOP_FUNCTION}_csynth.rpt]

puts "Timing report:"
puts [exec cat $PROJECT_NAME/$SOLUTION_NAME/syn/report/${TOP_FUNCTION}_csynth_timing.rpt]

puts "Resource utilization:"
puts [exec cat $PROJECT_NAME/$SOLUTION_NAME/syn/report/${TOP_FUNCTION}_csynth_utilization.rpt]

exit
EOF

# Run HLS synthesis
echo -e "${YELLOW}Running HLS synthesis...${NC}"
vitis_hls -f run_hls.tcl

# Check if synthesis completed successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}HLS synthesis completed successfully!${NC}"
    echo ""
    echo "Results:"
    echo "- Project directory: $PROJECT_NAME"
    echo "- Solution directory: $PROJECT_NAME/$SOLUTION_NAME"
    echo "- IP catalog: $PROJECT_NAME/$SOLUTION_NAME/impl/ip"
    echo "- Reports: $PROJECT_NAME/$SOLUTION_NAME/syn/report"
    echo ""
    echo "Key files:"
    echo "- Synthesis report: $PROJECT_NAME/$SOLUTION_NAME/syn/report/${TOP_FUNCTION}_csynth.rpt"
    echo "- Timing report: $PROJECT_NAME/$SOLUTION_NAME/syn/report/${TOP_FUNCTION}_csynth_timing.rpt"
    echo "- Utilization report: $PROJECT_NAME/$SOLUTION_NAME/syn/report/${TOP_FUNCTION}_csynth_utilization.rpt"
else
    echo -e "${RED}HLS synthesis failed!${NC}"
    echo "Check the log files for errors."
    exit 1
fi

# Optional: Extract key metrics
echo -e "${YELLOW}Extracting key metrics...${NC}"
if [ -f "$PROJECT_NAME/$SOLUTION_NAME/syn/report/${TOP_FUNCTION}_csynth.rpt" ]; then
    echo "Performance metrics:"
    grep -A 10 "Performance Estimates" "$PROJECT_NAME/$SOLUTION_NAME/syn/report/${TOP_FUNCTION}_csynth.rpt" || echo "Performance section not found"
    
    echo ""
    echo "Resource utilization:"
    grep -A 15 "Resource Utilization" "$PROJECT_NAME/$SOLUTION_NAME/syn/report/${TOP_FUNCTION}_csynth_utilization.rpt" || echo "Utilization section not found"
fi

echo -e "${GREEN}Synthesis complete!${NC}"