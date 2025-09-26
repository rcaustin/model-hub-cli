#!/usr/bin/env python3
"""
Test script for GitHub token validation
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import validate_github_token

def test_github_token_validation():
    """Test GitHub token validation scenarios."""
    
    print("🧪 Testing GitHub Token Validation")
    print("=" * 50)
    
    # Test scenarios
    scenarios = [
        {
            "name": "No GitHub Token",
            "token": None,
            "expected": False
        },
        {
            "name": "Empty GitHub Token",
            "token": "",
            "expected": False
        },
        {
            "name": "Invalid GitHub Token",
            "token": "invalid_token_12345",
            "expected": False
        },
        {
            "name": "Current GitHub Token",
            "token": os.getenv("GITHUB_TOKEN"),
            "expected": True if os.getenv("GITHUB_TOKEN") else False
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🔍 Testing: {scenario['name']}")
        print("-" * 30)
        
        # Set the environment variable
        if scenario['token'] is None:
            os.environ.pop("GITHUB_TOKEN", None)
        else:
            os.environ["GITHUB_TOKEN"] = scenario['token']
        
        # Test validation
        try:
            result = validate_github_token()
            expected = scenario['expected']
            
            if result == expected:
                print(f"✅ PASS: Expected {expected}, got {result}")
            else:
                print(f"❌ FAIL: Expected {expected}, got {result}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n✅ GitHub token validation tests completed!")

if __name__ == "__main__":
    test_github_token_validation()
