#!/usr/bin/env python3
"""
BWR-DNC Mini Project Status Checker
Provides overview of project health and structure.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

class ProjectStatusChecker:
    """Check the status of BWR-DNC Mini project."""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.status = {
            'structure': {},
            'files': {},
            'dependencies': {},
            'tests': {},
            'git': {}
        }
    
    def check_directory_structure(self):
        """Check if all required directories exist."""
        required_dirs = [
            'backend',
            'api', 
            'tests',
            'tests/unit',
            'tests/integration',
            'tests/performance',
            'debug',
            'docs',
            'checkpoints'
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            exists = full_path.exists() and full_path.is_dir()
            self.status['structure'][dir_path] = exists
            
        return all(self.status['structure'].values())
    
    def check_required_files(self):
        """Check if all required files exist."""
        required_files = [
            '.gitignore',
            'Makefile',
            'README_MODULAR.md',
            'project.conf',
            'setup_modular.sh',
            'tests/test_runner.py',
            'debug/debug_manager.py',
            'backend/requirements.txt',
            'api/requirements.txt'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            exists = full_path.exists() and full_path.is_file()
            self.status['files'][file_path] = exists
            
        return all(self.status['files'].values())
    
    def count_test_files(self):
        """Count test files in each category."""
        test_dirs = ['unit', 'integration', 'performance']
        
        for test_dir in test_dirs:
            test_path = self.project_root / 'tests' / test_dir
            if test_path.exists():
                test_files = list(test_path.glob('test_*.py'))
                self.status['tests'][test_dir] = len(test_files)
            else:
                self.status['tests'][test_dir] = 0
    
    def count_debug_files(self):
        """Count debug files."""
        debug_path = self.project_root / 'debug'
        if debug_path.exists():
            debug_files = list(debug_path.glob('debug_*.py'))
            self.status['debug'] = len(debug_files)
        else:
            self.status['debug'] = 0
    
    def count_doc_files(self):
        """Count documentation files."""
        docs_path = self.project_root / 'docs'
        if docs_path.exists():
            doc_files = list(docs_path.glob('*.md'))
            self.status['docs'] = len(doc_files)
        else:
            self.status['docs'] = 0
    
    def check_git_status(self):
        """Check git repository status."""
        git_path = self.project_root / '.git'
        self.status['git']['is_repo'] = git_path.exists()
        
        gitignore_path = self.project_root / '.gitignore'
        self.status['git']['has_gitignore'] = gitignore_path.exists()
    
    def run_full_check(self):
        """Run all status checks."""
        print("BWR-DNC Mini Project Status Check")
        print("=" * 40)
        print(f"Project Root: {self.project_root}")
        print(f"Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Directory structure
        structure_ok = self.check_directory_structure()
        print(f"📁 Directory Structure: {'✅ OK' if structure_ok else '❌ ISSUES'}")
        for dir_path, exists in self.status['structure'].items():
            status_icon = "✅" if exists else "❌"
            print(f"   {status_icon} {dir_path}")
        print()
        
        # Required files
        files_ok = self.check_required_files()
        print(f"📄 Required Files: {'✅ OK' if files_ok else '❌ MISSING'}")
        for file_path, exists in self.status['files'].items():
            status_icon = "✅" if exists else "❌"
            print(f"   {status_icon} {file_path}")
        print()
        
        # Test files
        self.count_test_files()
        print("🧪 Test Files:")
        for test_type, count in self.status['tests'].items():
            print(f"   {test_type}: {count} files")
        print()
        
        # Debug files
        self.count_debug_files()
        print(f"🐛 Debug Files: {self.status['debug']} files")
        print()
        
        # Documentation
        self.count_doc_files()
        print(f"📚 Documentation: {self.status['docs']} files")
        print()
        
        # Git status
        self.check_git_status()
        print("🔧 Git Status:")
        print(f"   Repository: {'✅' if self.status['git']['is_repo'] else '❌'}")
        print(f"   .gitignore: {'✅' if self.status['git']['has_gitignore'] else '❌'}")
        print()
        
        # Overall status
        overall_ok = structure_ok and files_ok
        print("=" * 40)
        print(f"Overall Status: {'✅ GOOD' if overall_ok else '❌ NEEDS ATTENTION'}")
        
        if not overall_ok:
            print("\nTo fix issues, run: ./setup_modular.sh")
        
        return overall_ok

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BWR-DNC Mini Project Status Checker")
    parser.add_argument("--path", help="Project root path (default: current directory)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    checker = ProjectStatusChecker(args.path)
    
    if args.quiet:
        # Just return exit code
        ok = checker.run_full_check()
        sys.exit(0 if ok else 1)
    else:
        checker.run_full_check()

if __name__ == "__main__":
    main()
