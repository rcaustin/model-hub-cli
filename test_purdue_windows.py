#!/usr/bin/env python3
"""
Windows-compatible test script for Purdue GenAI integration
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set the API key directly in the script
os.environ["PURDUE_GENAI_API_KEY"] = "sk-3b7fb8303ac440278d6c2c4c7fa12a5f"

from src.Model import Model
from src.ModelCatalogue import ModelCatalogue

def main():
    """Test the Purdue GenAI integration."""
    
    print("🧪 Testing Purdue GenAI Studio API Integration")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv("PURDUE_GENAI_API_KEY")
    if api_key:
        print(f"✅ API Key found: {api_key[:10]}...")
        print("🤖 LLM analysis will be enabled")
    else:
        print("⚠️  No API key found")
        print("📝 Rule-based analysis will be used")
    
    # Create a test model
    test_urls = [
        "https://huggingface.co/facebook/bart-large",
        "https://github.com/huggingface/transformers", 
        "https://huggingface.co/datasets/squad"
    ]
    
    print(f"\n�� Testing with model: {test_urls[0]}")
    
    try:
        # Create model and catalogue
        print("�� Creating model and catalogue...")
        model = Model(test_urls)
        catalogue = ModelCatalogue()
        catalogue.addModel(model)
        
        # Evaluate the model
        print("📊 Evaluating model metrics...")
        catalogue.evaluateModels()
        
        # Get performance claims results
        performance_score = model.getScore("PerformanceClaimsMetric")
        performance_latency = model.getLatency("PerformanceClaimsMetric")
        
        print(f"\n🎯 Performance Claims Score: {performance_score}")
        print(f"⏱️  Analysis Latency: {performance_latency}ms")
        
        # Determine if LLM was used
        if api_key and performance_latency > 1000:
            print("🤖 LLM analysis was used (high latency indicates API call)")
        else:
            print("📝 Rule-based analysis was used")
        
        # Show other metrics
        print(f"\n📊 Other Metrics:")
        print(f"   License: {model.getScore('LicenseMetric')}")
        print(f"   Size: {model.getScore('SizeMetric')}")
        print(f"   Ramp Up: {model.getScore('RampUpMetric')}")
        print(f"   Net Score: {model.getScore('NetScore')}")
        
        print(f"\n✅ Test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
