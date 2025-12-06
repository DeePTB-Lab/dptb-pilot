import os
import sys
import json
from pathlib import Path

# Add backend to path
sys.path.append(os.path.abspath("backend"))

from dptb_agent_tools.modules.config_tool import generate_deeptb_e3_training_config
from dptb_agent_tools.modules.sk_baseline_model import generate_sk_baseline_model

def test_config_gen():
    print("Testing generate_deeptb_e3_training_config...")
    try:
        result = generate_deeptb_e3_training_config(
            material="Si",
            work_path=".",
            output_file_name="test_config.json"
        )
        print(f"✅ Config generated at: {result['config_path']}")
        
        with open("test_config.json", "r") as f:
            data = json.load(f)
            if data["common_options"]["basis"]["Si"] == "1s":
                print("✅ Content verification passed")
            else:
                print("❌ Content verification failed")
                
    except Exception as e:
        print(f"❌ Config generation failed: {e}")

def test_sk_model_gen():
    print("\nTesting generate_sk_baseline_model...")
    try:
        result = generate_sk_baseline_model(
            basemodel="poly4",
            basis='{"Si": ["s", "p"]}',
            work_path="."
        )
        print(f"✅ Model generated at: {result['model_path']}")
        
        if os.path.exists("Si_poly4.json"): # Assuming default name
             print("✅ Model file exists")
        else:
             print("⚠️ Model file not found (check naming convention)")
             
    except Exception as e:
        print(f"❌ SK model generation failed: {e}")

if __name__ == "__main__":
    test_config_gen()
    test_sk_model_gen()
