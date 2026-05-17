import json
from config import AP_MODEL_STRUCTURE
from agent_manager import AgentManager

class APBuilder:
    def __init__(self, openai_client):
        self.client = openai_client
        self.agent_manager = AgentManager(openai_client)

    def generate_future_stage_multi_agent(self, tech_topic: str) -> dict:
        """Build the 18-element AP model (6 objects + 12 arrows) for the given topic set in the future."""
        print(f"\n--- Generating Stage 3 (Future) with Multi-Agents ---")

        self.agent_manager.generate_agents(tech_topic)

        model = {
            "stage": "Stage 3",
            "era": "Future (Maturity Period)",
            "nodes": {},
            "arrows": []
        }
        base_context = f"## Theme: {tech_topic}\n## Era: Future (Maturity/Transformation Period)\n"

        self._generate_objects(model, tech_topic, base_context)
        self._generate_arrows(model, tech_topic, base_context)

        return model

    def _snapshot_context(self, base_context: str, model: dict) -> str:
        """Append the model built so far to the base context string."""
        return base_context + f"\n## Stage 3 (Generated so far):\n{json.dumps(model, indent=2, ensure_ascii=False)}"

    def _generate_objects(self, model: dict, topic: str, base_context: str):
        print("\n[Phase 1] Generating Objects...")
        for obj_name in AP_MODEL_STRUCTURE["objects"]:
            context = self._snapshot_context(base_context, model)
            content = self.agent_manager.run_multi_agent_generation(
                element_type=f"Object: {obj_name}",
                element_desc="",
                topic=topic,
                full_context_str=context
            )
            model["nodes"][obj_name] = content

    def _generate_arrows(self, model: dict, topic: str, base_context: str):
        print("\n[Phase 2] Generating Arrows...")
        for arrow_name, info in AP_MODEL_STRUCTURE["arrows"].items():
            context = self._snapshot_context(base_context, model)
            content = self.agent_manager.run_multi_agent_generation(
                element_type=f"Arrow: {arrow_name}",
                element_desc=f"From '{info['from']}' to '{info['to']}'",
                topic=topic,
                full_context_str=context
            )
            model["arrows"].append({
                "source": info["from"],
                "target": info["to"],
                "type": arrow_name,
                "definition": content,
                "example": "Future Concept"
            })
