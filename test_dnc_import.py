#!/usr/bin/env python3

"""
Test script to verify DNC imports work correctly
"""

def test_dnc_imports():
    try:
        # Test importing the main DNC class
        from bwr.model import DNC
        print("✓ Successfully imported DNC class")
        
        # Test importing from the main package
        from bwr import DNC, AdvancedDNC
        print("✓ Successfully imported from bwr package")
        
        # Test importing integration components
        from bwr.integration import create_advanced_dnc, AdvancedDNCTrainer
        print("✓ Successfully imported integration components")
        
        # Test importing unlimited context
        from bwr.unlimited_context import UnlimitedContextDNC
        print("✓ Successfully imported UnlimitedContextDNC")
        
        print("\nAll DNC imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_dnc_imports()
    exit(0 if success else 1)