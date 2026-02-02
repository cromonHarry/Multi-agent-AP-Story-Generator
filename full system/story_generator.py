import json
from config import SYSTEM_PROMPT
from utils import parse_json_response

# 仅供写作 Agent 使用的创意 Prompt
CREATIVE_SYSTEM_PROMPT = "You are an award-winning Science Fiction author. Your goal is to write compelling, logical, and creative narratives based on given data."

class StoryGenerator:
    def __init__(self, openai_client):
        self.client = openai_client

    # ==========================================
    # 0. Global Overseer: Briefing Director
    # ==========================================
    def _overseer_prepare_brief(self, ap_context_data, target_type):
        print(f"  [Global Overseer] Preparing brief for {target_type}...")
        ap_context_str = json.dumps(ap_context_data, indent=2, ensure_ascii=False)

        if target_type == "setting":
            focus_instruction = "Extract ONLY the static elements relevant to World Building."
        else:
            focus_instruction = "Extract ONLY the dynamic elements relevant to Plot."

        prompt = f"""
You are the **Global Overseer**. You hold the full Sociological Model (AP Model) for the **FUTURE WORLD**. 
Your task is to give necessary information to agents, to help them complete a creative science fiction set in this unique future.

## The Future World Model (Master File)
{ap_context_str}

## Task
Create a **Concept Brief** for the {target_type} Agent.
{focus_instruction}

## Output Format (JSON)
{{
    "briefing_theme": "A short theme title",
    "relevant_data_points": "A summary of the specific AP model elements (Nodes/Arrows) that this Agent should focus on."
}}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )
        return parse_json_response(response.choices[0].message.content)


    # ==========================================
    # 1. Global Overseer: The Critic
    # ==========================================
    def _global_check(self, content_type, content_data, context_data, ap_master_data, specific_criteria):
        print(f"  [Global Overseer] Reviewing {content_type}...")
        ap_master_str = json.dumps(ap_master_data, indent=2, ensure_ascii=False)

        prompt = f"""
You are an award-winning Science Fiction author and editor. Now your role is a strict **Global Overseer**. 
Your job is to ensure the content follows the logic of the **AP Model** and the specific instructions provided.

## Reference Material 1: The Future Sociological Model (Ground Truth)
{ap_master_str}

## Reference Material 2: Instructions provided to the Agent (The Brief)
{context_data}

## Review Criteria
{specific_criteria}

## The Content to Review ({content_type})
{json.dumps(content_data, indent=2, ensure_ascii=False)}

## Output Format (JSON)
{{
    "approved": true/false,
    "feedback": "If approved, keep empty. If rejected, provide specific advice on how to fix the contradiction with the AP Model or the Brief."
}}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )
        return parse_json_response(response.choices[0].message.content)

    # ==========================================
    # 2. Setting Agent (World & Characters)
    # ==========================================
    def _agent_build_settings(self, setting_brief, feedback=""):
        print(f"  [Setting Agent] Drafting World & Characters... {(f'(Fixing: {feedback})' if feedback else '')}")
        brief_str = json.dumps(setting_brief, indent=2, ensure_ascii=False)

        prompt = f"""
You are the **Setting Agent**. Your task is to design amazing and creative World & Character settings for a sci-fi story.

## Director's Brief
{brief_str}

## Instructions
1. **World View**: Describe the year, the background, the state of the product or concept.
2. **Characters**: Create EXACTLY 4 key characters.

## Previous Feedback (If any, you MUST fix this)
{feedback}

## Output Format (JSON)
{{
    "world_view": "Description",
    "characters": [ {{ "name": "...", "role": "...", "motivation": "..." }} ]
}}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CREATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return parse_json_response(response.choices[0].message.content)

    # ==========================================
    # 3. Outline Agent (Plot Architect)
    # ==========================================
    def _agent_build_outline_step(self, step_name, step_goal, settings, plot_brief, current_outline_history, feedback=""):
        print(f"  [Outline Agent] Drafting {step_name}... {(f'(Fixing: {feedback})' if feedback else '')}")

        history_text = "\n".join([f"{k}: {v['summary']}" for k, v in current_outline_history.items()])
        settings_str = json.dumps(settings, indent=2, ensure_ascii=False)
        brief_str = json.dumps(plot_brief, indent=2, ensure_ascii=False)
        
        prompt = f"""
You are the **Outline Agent**. Write the **{step_name}** of the story.

## The Story Settings
{settings_str}

## Director's Plot Instructions
{brief_str}

## Current Plot History
{history_text if history_text else "This is the beginning of the story."}

## Step Goal
{step_goal}

## Previous Feedback (If any, fix this)
{feedback}

## Output Format (JSON)
{{
    "summary": "Detailed narrative paragraph of what happens. Focus on character actions and plot progression. (Approx 100 words)."
}}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CREATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return parse_json_response(response.choices[0].message.content)

    # ==========================================
    # 4. Main Workflow Orchestrator
    # ==========================================
    def generate_outline(self, ap_data_dict: dict) -> str:
        print("\n=== Starting Multi-Agent Story Architecture (Mode: PURE FUTURE) ===")
        
        # 直接使用传入的 AP Data（现在只有 Stage 3）
        future_context_data = ap_data_dict.get("Stage 3", ap_data_dict)

        # --- PHASE 0: Director prepares Briefs ---
        setting_brief = self._overseer_prepare_brief(future_context_data, "setting")
        print(f"  > Director's Setting Brief: {setting_brief.get('briefing_theme', 'Unknown Theme')}")

        # --- PHASE 1: Build & Verify Settings ---
        settings = None
        feedback = ""
        max_retries = 3
        
        for i in range(max_retries):
            settings = self._agent_build_settings(setting_brief, feedback)
            criteria = "Check if the 'World View' and 'Characters' logically reflect the Director's Brief provided AND do not contradict the Future AP Model."
            context_data_str = json.dumps(setting_brief, ensure_ascii=False)
            review = self._global_check("Story Settings", settings, context_data_str, future_context_data, criteria)
            
            if review['approved']:
                print("  [Global Overseer] Settings Approved. ✅")
                break
            else:
                feedback = review['feedback']
                print(f"  [Global Overseer] Rejected Settings. Feedback: {feedback} ❌")
        
        if not settings:
             return "Error: Settings generation failed."

        # --- PHASE 0.5: Director prepares Plot Brief ---
        plot_brief = self._overseer_prepare_brief(future_context_data, "outline")
        print(f"  > Director's Plot Brief: {plot_brief.get('briefing_theme', 'Unknown Theme')}")

        # --- PHASE 2: Build Outline Step-by-Step ---
        steps_config = [
            {"name": "1. Exposition", "goal": "The story begins in the setting, introducing the characters and the setting of the story."},
            {"name": "2. Rising Action", "goal": "An event or conflict is introduced in the story and characters begins to face a series of challenges or conflicts."},
            {"name": "3. Climax", "goal": "This is the most exciting moment or a turning point in the story."},
            {"name": "4. Falling Action", "goal": "After the climax, the story begins to transition to the ending."},
            {"name": "5. Resolution", "goal": "The ending of the story."}
        ]

        final_outline_steps = {}

        for step in steps_config:
            print(f"\n-- Processing {step['name']} --")
            step_content = None
            feedback = ""
            
            for i in range(max_retries):
                step_content = self._agent_build_outline_step(
                    step['name'], 
                    step['goal'], 
                    settings, 
                    plot_brief,
                    final_outline_steps, 
                    feedback
                )
                
                context_for_review = f"PLOT BRIEF: {json.dumps(plot_brief)}\nPREVIOUS PLOT: {json.dumps(final_outline_steps)}"
                criteria = "Consistency Check: Does this outline step follow the Director's Plot Brief AND remains consistent with the Future AP Model?"
                review = self._global_check(step['name'], step_content, context_for_review, future_context_data, criteria)
                
                if review['approved']:
                    print(f"  [Global Overseer] {step['name']} Approved. ✅")
                    final_outline_steps[step['name']] = step_content
                    break
                else:
                    feedback = review['feedback']
                    print(f"  [Global Overseer] Rejected {step['name']}. Feedback: {feedback} ❌")
            
            if step['name'] not in final_outline_steps:
                 final_outline_steps[step['name']] = step_content

        # --- PHASE 3: Compile Final Output (PURE TEXT ONLY) ---
        print("\n=== Compiling Final Story Outline (5 Paragraphs) ===")
        
        paragraphs = []
        for step in steps_config:
            step_name = step['name']
            step_data = final_outline_steps.get(step_name, {})
            # 只提取 summary 内容，去除标题和备注
            text = step_data.get('summary', '').strip()
            if text:
                paragraphs.append(text)

        # 使用双换行符连接 5 个段落
        final_pure_text = "\n\n".join(paragraphs)
        return final_pure_text