import json
from config import AP_MODEL_STRUCTURE
from agent_manager import AgentManager

class APBuilder:
    def __init__(self, openai_client):
        self.client = openai_client
        self.agent_manager = AgentManager(openai_client)

    def generate_future_stage_multi_agent(self, tech_topic: str) -> dict:
        """
        Stage 3 Only: Build using Multi-Agent System.
        No dependency on Stage 2 data. Pure imagination based on Theme.
        """
        print(f"\n--- Generating Stage 3 (Future) with Multi-Agents ---")
        
        # 1. Initialize Agents
        self.agent_manager.generate_agents(tech_topic)
        
        stage_3_model = {
            "stage": "Stage 3",
            "era": "Future (Maturity Period)",
            "nodes": {},
            "arrows": []
        }

        # Prepare Context String (Pure Theme)
        base_context = f"## Theme: {tech_topic}\n## Era: Future (Maturity/Transformation Period)\n"

        # 2. Iterate Objects Sequentially
        print("\n[Phase 1] Generating Objects...")
        for obj_name in AP_MODEL_STRUCTURE["objects"]:
            current_stage3_context = f"\n## Stage 3 (Generated so far):\n{json.dumps(stage_3_model, indent=2, ensure_ascii=False)}"
            full_context = base_context + current_stage3_context
            
            content = self.agent_manager.run_multi_agent_generation(
                element_type=f"Object: {obj_name}",
                element_desc="", 
                topic=tech_topic,
                full_context_str=full_context
            )
            stage_3_model["nodes"][obj_name] = content

        # 3. Iterate Arrows Sequentially
        print("\n[Phase 2] Generating Arrows...")
        for arrow_name, info in AP_MODEL_STRUCTURE["arrows"].items():
            current_stage3_context = f"\n## Stage 3 (Generated so far):\n{json.dumps(stage_3_model, indent=2, ensure_ascii=False)}"
            full_context = base_context + current_stage3_context
            
            arrow_desc = f"From '{info['from']}' to '{info['to']}'"
            content = self.agent_manager.run_multi_agent_generation(
                element_type=f"Arrow: {arrow_name}",
                element_desc=arrow_desc,
                topic=tech_topic,
                full_context_str=full_context
            )
            
            stage_3_model["arrows"].append({
                "source": info["from"],
                "target": info["to"],
                "type": arrow_name,
                "definition": content,
                "example": "Future Concept" 
            })

        return stage_3_model