#!/bin/bash

# BWR-DNC Mini Setup Script
# Automated setup for the modular BWR-DNC implementation

set -e  # Exit on any error

echo "======================================"
echo "BWR-DNC Mini Modular Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version: $python_version"
}

# Setup virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        print_info "Virtual environment created."
    else
        print_warning "Virtual environment already exists."
    fi
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source venv/bin/activate
    print_info "Virtual environment activated."
}

# Install dependencies
install_dependencies() {
    print_info "Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install backend dependencies
    if [ -f "backend/requirements.txt" ]; then
        print_info "Installing backend dependencies..."
        pip install -r backend/requirements.txt
    fi
    
    # Install API dependencies
    if [ -f "api/requirements.txt" ]; then
        print_info "Installing API dependencies..."
        pip install -r api/requirements.txt
    fi
    
    # Install development dependencies
    print_info "Installing development dependencies..."
    pip install pytest black flake8 isort
    
    print_info "Dependencies installed successfully."
}

# Verify directory structure
verify_structure() {
    print_info "Verifying modular directory structure..."
    
    required_dirs=("tests" "debug" "docs" "backend" "api")
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_info "✓ Directory '$dir' exists"
        else
            print_error "✗ Directory '$dir' missing"
            exit 1
        fi
    done
    
    required_files=(".gitignore" "Makefile" "README_MODULAR.md")
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            print_info "✓ File '$file' exists"
        else
            print_error "✗ File '$file' missing"
            exit 1
        fi
    done
    
    print_info "Directory structure verified."
}

# Run basic tests
run_basic_tests() {
    print_info "Running basic test verification..."
    
    if [ -f "tests/test_runner.py" ]; then
        python tests/test_runner.py --unit || print_warning "Some unit tests failed (this may be expected for initial setup)"
    else
        print_warning "Test runner not found, skipping test verification."
    fi
}

# Main setup flow
main() {
    print_info "Starting BWR-DNC Mini setup..."
    
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Run setup steps
    check_python
    setup_venv
    activate_venv
    install_dependencies
    verify_structure
    run_basic_tests
    
    print_info "Setup completed successfully!"
    print_info ""
    print_info "Next steps:"
    print_info "1. Activate the virtual environment: source venv/bin/activate"
    print_info "2. Run tests: make test"
    print_info "3. Start training: make train-basic"
    print_info "4. Access debug tools: make debug"
    print_info ""
    print_info "For more commands, run: make help"
}

# Run main function
main "$@"
