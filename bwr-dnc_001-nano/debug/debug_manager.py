"""
Debug utilities manager for BWR-DNC development.
Provides centralized access to all debug tools and utilities.
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Any

class DebugManager:
    """Centralized debug manager for BWR-DNC."""
    
    def __init__(self):
        self.debug_dir = Path(__file__).parent
        self.available_tools = self._discover_debug_tools()
    
    def _discover_debug_tools(self) -> Dict[str, str]:
        """Discover all available debug tools."""
        tools = {}
        for file in self.debug_dir.glob("debug_*.py"):
            tool_name = file.stem.replace("debug_", "")
            tools[tool_name] = str(file)
        return tools
    
    def list_tools(self) -> List[str]:
        """List all available debug tools."""
        return list(self.available_tools.keys())
    
    def run_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """Run a specific debug tool."""
        if tool_name not in self.available_tools:
            raise ValueError(f"Debug tool '{tool_name}' not found. Available tools: {self.list_tools()}")
        
        # Import and run the debug tool
        module_name = f"debug_{tool_name}"
        spec = importlib.util.spec_from_file_location(module_name, self.available_tools[tool_name])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for a main function or run the module
        if hasattr(module, 'main'):
            return module.main(*args, **kwargs)
        elif hasattr(module, 'run'):
            return module.run(*args, **kwargs)
        else:
            print(f"Debug tool '{tool_name}' executed")
    
    def get_tool_info(self, tool_name: str) -> str:
        """Get information about a debug tool."""
        if tool_name not in self.available_tools:
            return f"Debug tool '{tool_name}' not found."
        
        try:
            with open(self.available_tools[tool_name], 'r') as f:
                content = f.read()
                # Extract docstring if available
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '"""' in line or "'''" in line:
                        # Find the end of docstring
                        for j in range(i+1, len(lines)):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                return '\n'.join(lines[i:j+1])
                return f"Debug tool: {tool_name}"
        except Exception as e:
            return f"Error reading tool info: {e}"

def main():
    """Interactive debug tool selector."""
    manager = DebugManager()
    
    print("BWR-DNC Debug Manager")
    print("=" * 30)
    print(f"Available debug tools: {len(manager.list_tools())}")
    
    for i, tool in enumerate(manager.list_tools(), 1):
        print(f"{i}. {tool}")
    
    print("\nEnter tool number or name to run, 'info <tool>' for details, or 'quit' to exit:")
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.startswith('info '):
                tool_name = user_input[5:].strip()
                print(manager.get_tool_info(tool_name))
                continue
            
            # Try to parse as number
            try:
                tool_index = int(user_input) - 1
                if 0 <= tool_index < len(manager.list_tools()):
                    tool_name = manager.list_tools()[tool_index]
                else:
                    print("Invalid tool number")
                    continue
            except ValueError:
                tool_name = user_input
            
            print(f"Running debug tool: {tool_name}")
            manager.run_tool(tool_name)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
