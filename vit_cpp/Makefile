# Makefile for ViT HLS Implementation

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -O3 -I./include

# Xilinx HLS settings (uncomment when using Vivado HLS)
# HLS_CXX = vitis_hls
# HLS_FLAGS = -std=c++14 -I./include

# Directories
SRC_DIR = src
INC_DIR = include
TEST_DIR = test
BUILD_DIR = build

# Source files for simulation
SIM_SOURCES = $(SRC_DIR)/vit_layers_sim.cpp $(SRC_DIR)/vit_sim.cpp
HLS_SOURCES = $(SRC_DIR)/vit_layers.cpp $(SRC_DIR)/vit.cpp
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.cpp)

# Object files
SIM_OBJECTS = $(SIM_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
HLS_OBJECTS = $(HLS_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Targets
SIM_TARGET = $(BUILD_DIR)/vit_test_sim
HLS_TARGET = vit_hls_project

# Default target (simulation)
all: $(SIM_TARGET)

# Simulation target
sim: $(SIM_TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile simulation source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link simulation executable
$(SIM_TARGET): $(SIM_OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Run simulation tests
test: $(SIM_TARGET)
	./$(SIM_TARGET)

# HLS synthesis (requires Vivado HLS)
hls: $(HLS_SOURCES)
	@echo "Creating HLS project..."
	@echo "open_project $(HLS_TARGET)" > run_hls.tcl
	@echo "set_top vit_inference" >> run_hls.tcl
	@echo "add_files $(SRC_DIR)/vit.cpp" >> run_hls.tcl
	@echo "add_files $(SRC_DIR)/vit_layers.cpp" >> run_hls.tcl
	@echo "add_files -tb $(TEST_DIR)/test_vit.cpp" >> run_hls.tcl
	@echo "open_solution \"solution1\"" >> run_hls.tcl
	@echo "set_part {xcu280-fsvh2892-2L-e}" >> run_hls.tcl
	@echo "create_clock -period 10 -name default" >> run_hls.tcl
	@echo "csim_design" >> run_hls.tcl
	@echo "csynth_design" >> run_hls.tcl
	@echo "cosim_design" >> run_hls.tcl
	@echo "export_design -format ip_catalog" >> run_hls.tcl
	@echo "exit" >> run_hls.tcl
	vitis_hls -f run_hls.tcl

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	rm -f run_hls.tcl

# Clean HLS project
clean-hls:
	rm -rf $(HLS_TARGET)

# Install dependencies (placeholder)
install-deps:
	@echo "Installing dependencies..."
	@echo "For HLS synthesis, please install Xilinx Vivado HLS or Vitis HLS"
	@echo "For simulation, standard C++ compiler is sufficient"

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build the simulation test executable"
	@echo "  sim        - Build the simulation test executable"
	@echo "  test       - Build and run simulation tests"
	@echo "  hls        - Run HLS synthesis (requires Vivado HLS)"
	@echo "  clean      - Clean build files"
	@echo "  clean-hls  - Clean HLS project files"
	@echo "  help       - Show this help"

.PHONY: all sim test hls clean clean-hls install-deps help