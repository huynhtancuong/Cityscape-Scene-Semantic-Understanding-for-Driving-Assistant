# Updated LLM Setup for Movement Decision Making with DeepSeek Models
import json
import re
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import ollama

client = ollama.Client()

def create_llm_prompt_for_movement(scene_description, movement_goal="safe navigation"):
    """
    Create a structured prompt for LLM to make movement decisions based on scene analysis.
    
    Args:
        scene_description (str): The scene description from vision-to-text conversion
        movement_goal (str): The desired movement objective
    
    Returns:
        str: Formatted prompt for LLM
    """
    
    prompt_template = f"""
AUTONOMOUS NAVIGATION TASK

OBJECTIVE: {movement_goal}

SCENE ANALYSIS:
{scene_description}

INSTRUCTIONS FOR DECISION MAKING:
Based on the scene analysis above, please provide:

1. SAFETY ASSESSMENT:
   - Rate overall safety level (1-10, where 10 is safest)
   - Identify potential hazards and their risk levels
   - Note any immediate dangers requiring attention

2. RECOMMENDED ACTIONS:
   - Primary movement direction (forward/backward/left/right/stop)
   - Speed recommendation (stop/slow/normal/fast)
   - Specific maneuvers if needed (lane change, yield, etc.)

3. REASONING:
   - Explain why this action is recommended
   - What factors influenced the decision
   - Alternative actions considered

4. MONITORING PRIORITIES:
   - What objects/areas to monitor continuously
   - Expected changes in the next few seconds
   - Conditions that would require action change

Please provide your response in a structured format for autonomous vehicle control.
"""
    
    return prompt_template


def get_response_from_llm(llm_prompt) -> str:
    model = "qwen3:0.6b"
    response = client.generate(model=model, prompt=llm_prompt)
    # parsed_response = parse_llm_response(response.response)

    return response.response

def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response with improved extraction for DeepSeek format and structured bullet points."""
    
    # print("LLM Raw Response:")
    # print(response)
    # print("-" * 40)

    decision = {
        'safety_score': 5,
        'hazards': [],
        'primary_action': 'stop',
        'speed': 'stop',
        'reasoning': 'Default safe action - unable to parse response',
        'monitor_priorities': []
    }

    try:
        # Normalize response for easier parsing
        norm = response.replace('**', '').replace('__', '').replace('–', '-')

        # Safety score extraction (handle various phrasings)
        safety_score_match = re.search(r'(?:Safety rating|Overall Safety Level)[:\s]*([0-9]{1,2})', norm, re.IGNORECASE)
        if safety_score_match:
            score = int(safety_score_match.group(1))
            if 1 <= score <= 10:
                decision['safety_score'] = score

        # Hazards and risk extraction (nested bullets)
        hazards = []
        # Find the Potential Hazards and Risk Levels section
        hazards_section = re.search(r'Potential Hazards and Risk Levels:\s*((?:-\s*[^\n]+\n?)+)', norm, re.IGNORECASE)
        if hazards_section:
            block = hazards_section.group(1)
            # Find all sub-bullets (e.g., Car (15.9% risk))
            for line in block.split('\n'):
                m = re.match(r'\s*-\s*([A-Za-z ]+)\s*\(([^)]+risk)\)', line)
                if m:
                    hazards.append(f"{m.group(1).strip()} ({m.group(2).strip()})")
        # Also check for Moving Obstacles sub-bullets
        moving_obs_section = re.search(r'Moving Obstacles:\s*((?:-\s*[^\n]+\n?)+)', norm, re.IGNORECASE)
        if moving_obs_section:
            block = moving_obs_section.group(1)
            for line in block.split('\n'):
                m = re.match(r'\s*-\s*([A-Za-z ]+)\s*\(([^)]+risk)\)', line)
                if m:
                    hazards.append(f"{m.group(1).strip()} ({m.group(2).strip()})")

        # Immediate Danger extraction
        immediate_danger = []
        immediate_section = re.search(r'Immediate Danger:\s*((?:-\s*[^\n]+\n?)+)', norm, re.IGNORECASE)
        if immediate_section:
            block = immediate_section.group(1)
            for line in block.split('\n'):
                line = line.strip('-').strip()
                if line:
                    immediate_danger.append(line)
        # Or single line
        else:
            m = re.search(r'Immediate Danger:\s*([^\n]+)', norm, re.IGNORECASE)
            if m:
                immediate_danger.append(m.group(1).strip())
        # Add immediate dangers to hazards if not already present
        for d in immediate_danger:
            if d and d not in hazards:
                hazards.append(d)
        # Remove 'None' and deduplicate
        hazards = [h for h in hazards if h and h.lower() != 'none']
        decision['hazards'] = list(dict.fromkeys(hazards))

        # --- Improved RECOMMENDED ACTIONS parsing ---
        # Find the Recommended Actions section
        actions_section = re.search(r'Recommended Actions:?\s*((?:-\s*[^\n]+\n?)+)', norm, re.IGNORECASE)
        action_found = False
        speed_found = False
        if actions_section:
            block = actions_section.group(1)
            for line in block.split('\n'):
                # Primary Movement Direction
                m = re.match(r'-\s*Primary Movement Direction:?\s*([A-Za-z]+)', line, re.IGNORECASE)
                if m:
                    action = m.group(1).strip().lower()
                    if action in ['forward', 'backward', 'left', 'right', 'stop']:
                        decision['primary_action'] = action
                        action_found = True
                # Speed Recommendation
                m = re.match(r'-\s*Speed Recommendation:?\s*([A-Za-z]+)', line, re.IGNORECASE)
                if m:
                    speed = m.group(1).strip().lower()
                    if speed in ['stop', 'slow', 'normal', 'fast']:
                        decision['speed'] = speed
                        speed_found = True
                    else:
                        # Try to extract speed from text (e.g., 'slow (to avoid collisions)')
                        if 'slow' in speed:
                            decision['speed'] = 'slow'
                            speed_found = True
                        elif 'stop' in speed:
                            decision['speed'] = 'stop'
                            speed_found = True
                        elif 'fast' in speed:
                            decision['speed'] = 'fast'
                            speed_found = True
                        elif 'normal' in speed:
                            decision['speed'] = 'normal'
                            speed_found = True
        # Fallback to previous regex if not found in block
        if not action_found:
            primary_dir_match = re.search(r'Primary Movement Direction[:\s]*([^\n\*\-]+)', norm, re.IGNORECASE)
            if primary_dir_match:
                action = primary_dir_match.group(1).strip().lower()
                if action in ['forward', 'backward', 'left', 'right', 'stop']:
                    decision['primary_action'] = action
        if not speed_found:
            speed_match = re.search(r'Speed Recommendation[:\s]*([^\n\*\-]+)', norm, re.IGNORECASE)
            if speed_match:
                speed = speed_match.group(1).strip().lower()
                if speed in ['stop', 'slow', 'normal', 'fast']:
                    decision['speed'] = speed
                else:
                    # Try to extract speed from text (e.g., 'slow (1-2 mph)')
                    if 'slow' in speed:
                        decision['speed'] = 'slow'
                    elif 'stop' in speed:
                        decision['speed'] = 'stop'
                    elif 'fast' in speed:
                        decision['speed'] = 'fast'
                    elif 'normal' in speed:
                        decision['speed'] = 'normal'

        # Reasoning extraction (handle nested bullets)
        reasoning = []
        reasoning_section = re.search(r'Reasoning:\s*((?:-\s*[^\n]+\n?)+)', norm, re.IGNORECASE)
        if reasoning_section:
            block = reasoning_section.group(1)
            for line in block.split('\n'):
                line = line.strip('-').strip()
                if line:
                    reasoning.append(line)
            decision['reasoning'] = ' '.join(reasoning)
        else:
            # fallback: single line
            reason_line = re.search(r'Reasoning[:\s]*([^\n]+)', norm, re.IGNORECASE)
            if reason_line:
                decision['reasoning'] = reason_line.group(1).strip()
        # --- Improved REASONING parsing ---
        reasoning = []
        # Find the Reasoning section (including sub-bullets)
        reasoning_section = re.search(r'Reasoning:?\s*((?:-\s*[^\n]+\n?)+)', norm, re.IGNORECASE)
        if reasoning_section:
            block = reasoning_section.group(1)
            current_subheader = None
            subreasoning = {}
            for line in block.split('\n'):
                line = line.strip()
                # Subheader (e.g., - Factors Influencing Decision:)
                m = re.match(r'-\s*([A-Za-z ]+):', line)
                if m:
                    current_subheader = m.group(1).strip()
                    subreasoning[current_subheader] = []
                    continue
                # Sub-bullet (e.g., - Obstacle Presence: ...)
                m2 = re.match(r'-\s*([A-Za-z ]+):\s*(.+)', line)
                if m2 and current_subheader:
                    subreasoning[current_subheader].append(f"{m2.group(1).strip()}: {m2.group(2).strip()}")
                    continue
                # Simple bullet
                if line.startswith('-'):
                    content = line.lstrip('-').strip()
                    if current_subheader:
                        subreasoning[current_subheader].append(content)
                    else:
                        reasoning.append(content)
            # Flatten subreasoning
            for k, v in subreasoning.items():
                if v:
                    reasoning.append(f"{k}: " + ' '.join(v))
            if reasoning:
                decision['reasoning'] = ' '.join(reasoning)
        else:
            # fallback: single line
            reason_line = re.search(r'Reasoning[:\s]*([^\n]+)', norm, re.IGNORECASE)
            if reason_line:
                decision['reasoning'] = reason_line.group(1).strip()

        # Monitoring priorities (handle nested bullets and flexible headers)
        monitor = []
        # Objects/Areas to Monitor
        obj_area_section = re.search(r'Objects/Areas to Monitor:\s*((?:-\s*[^\n]+\n?)+)', norm, re.IGNORECASE)
        if obj_area_section:
            block = obj_area_section.group(1)
            for line in block.split('\n'):
                line = line.strip('-').strip()
                if line:
                    monitor.append(f"Monitor: {line}")
        # Expected Changes
        expected_section = re.search(r'Expected Changes[^:]*:\s*((?:-\s*[^\n]+\n?)+)', norm, re.IGNORECASE)
        if expected_section:
            block = expected_section.group(1)
            for line in block.split('\n'):
                line = line.strip('-').strip()
                if line:
                    monitor.append(f"Expected change: {line}")
        # Conditions Requiring Action
        cond_section = re.search(r'Conditions Requiring Action:\s*((?:-\s*[^\n]+\n?)+)', norm, re.IGNORECASE)
        if cond_section:
            block = cond_section.group(1)
            for line in block.split('\n'):
                line = line.strip('-').strip()
                if line:
                    monitor.append(f"Condition: {line}")
        # Remove empty/None and deduplicate
        monitor = [m for m in monitor if m and m.lower() != 'none' and m.strip() != ':']
        decision['monitor_priorities'] = list(dict.fromkeys(monitor))

    except Exception as e:
        print(f"Warning: Error parsing LLM response: {e}")

    return decision

if __name__ == "__main__":
    # For quick testing
    sample_response = """
**Autonomous Navigation Task**

**Safety Assessment:**

* Safety rating: 7/10
* Potential hazards:
    * Moving cars and trucks
    * Buildings and fences
* Immediate dangers:
    * Car at center

**Recommended Actions:**

* Primary movement direction: Forward
* Speed recommendation: Slow
* Specific maneuvers:
    * Yield to the car at the center of the road

**Reasoning:**

* The car at the center is a potential hazard with a high risk level. Yielding to it is the safest course of action.
* The speed should be slowed to allow for safe navigation around the obstacle.

**Monitoring Priorities:**

* Object: Car at center
* Expected changes: Vehicle may move or change lane
* Conditions for action change:
    * Car crosses into the lane
    * Car speeds up significantly

**Additional Considerations:**

* Vegetation and sky coverage provide limited visibility.
* The rider is present but has a low coverage area."""

    parsed = parse_llm_response(sample_response)
    print("Parsed Decision:")
    print(json.dumps(parsed, indent=2))