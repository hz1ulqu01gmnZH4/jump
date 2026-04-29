#!/usr/bin/env python3
"""
Standalone script to generate turn-1 and turn-2 prompts for world_ca_001
WITHOUT actually calling Claude. Used for isolation testing.

Simulates:
- turn-1 prompt (fresh start)
- turn-2 prompt (after get_train_obs returns train_obs)

Writes prompts to arm_b/debug_prompt_world_ca_001_t{n}.txt
"""
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arm_b.run_arm_b_v2 import build_prompt, load_skill_body
from worlds.gen import generate_all

# Get world_ca_001
all_instances = generate_all()
instance = next(i for i in all_instances if i["id"] == "world_ca_001")

skill_body = load_skill_body()

# Turn-1: no history
prompt_t1 = build_prompt(skill_body, instance, [], 0)
out_t1 = Path("arm_b/debug_prompt_world_ca_001_t1.txt")
out_t1.write_text(prompt_t1)
print(f"Turn-1 prompt: {len(prompt_t1)} chars -> {out_t1}")

# Turn-2: simulate get_train_obs return value
from harness import WorldHarness
harness = WorldHarness(instance)
train_obs_result = harness.get_train_obs()

history_after_t1 = [
    {"tool": "get_train_obs", "args": {}, "result": train_obs_result}
]

prompt_t2 = build_prompt(skill_body, instance, history_after_t1, 1)
out_t2 = Path("arm_b/debug_prompt_world_ca_001_t2.txt")
out_t2.write_text(prompt_t2)
print(f"Turn-2 prompt: {len(prompt_t2)} chars -> {out_t2}")

print("\nFirst 200 chars of turn-2 prompt:")
print(prompt_t2[:200])
print("\nLast 200 chars of turn-2 prompt:")
print(prompt_t2[-200:])
