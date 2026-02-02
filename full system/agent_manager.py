import concurrent.futures
import json
from openai import OpenAI
from config import SYSTEM_PROMPT, NUM_AGENTS, NUM_ITERATIONS
from utils import parse_json_response

class AgentManager:
    def __init__(self, openai_client):
        self.client = openai_client
        self.agents = []

    def generate_agents(self, topic: str) -> list:
        # 使用 Config 中的 NUM_AGENTS 参数
        print(f"\n[Agent Manager] Hiring {NUM_AGENTS} agents with DIVERSE perspectives for: {topic}...")
        
        prompt = f"""
You are the architect of a Sci-Fi Think Tank. Your goal is to predict the wild, mature future (Stage 3) of "{topic}".

Task: Create {NUM_AGENTS} distinct expert agents.
**CRITICAL REQUIREMENT**: To ensure a rich prediction, these {NUM_AGENTS} agents must hold **completely different views** or come from **completely different disciplines**.

Output in JSON format:
{{ "agents": [ {{ "name": "Creative Name", "expertise": "Field of expertise", "personality": "Personality/Tone", "perspective": "Their core belief about the future of {topic}" }} ] }}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=1.0, 
            response_format={"type": "json_object"}
        )
        result = parse_json_response(response.choices[0].message.content)
        self.agents = result.get("agents", [])
        
        # 截断或填充以确保符合 NUM_AGENTS 数量 (虽然 Prompt 已经要求，但做个双保险)
        if len(self.agents) > NUM_AGENTS:
            self.agents = self.agents[:NUM_AGENTS]
            
        for agent in self.agents:
            print(f"  - Agent Hired: {agent['name']} ({agent['expertise']})")
        return self.agents

    def _agent_think(self, agent, element_type, context_str, history):
        history_text = "\n".join([f"- {h}" for h in history]) if history else "None"
        
        # 优化思考 Prompt：不再提及 Past/Present 数据，完全基于想象
        prompt = f"""
You are **{agent['name']}**.
Expertise: {agent['expertise']}
Perspective: {agent['perspective']}

Task: Brainstorm the AP Model element "{element_type}" for the **Future (Stage 3)**.

## Context (Previous Future Generations):
{context_str}

## INSTRUCTION:
As a visionary, imagine how this technology has **mutated, evolved, or merged** into society in the future.
Your idea should reflect your specific PERSEPCTIVE: "{agent['perspective']}".
Be bold. Be weird.

Output a unique, bold idea (max 50 words). TEXT ONLY.
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=1.2
        )
        return response.choices[0].message.content.strip()

    def _judge_proposals(self, proposals, element_type, topic):
        """Judge selects the best proposal from the agents."""
        proposals_text = "\n".join([f"Proposal {i+1} ({p['agent']}): {p['content']}" for i, p in enumerate(proposals)])
        prompt = f"""
Topic: {topic}
Element: {element_type} (Stage 3: Future)

You are a Sci-Fi Editor selecting the most interesting concept for a story setting.
Here are proposals from different experts.

Selection Criteria:
1. **Creativity & Novelty**: Which idea offers the most interesting "What If"?
2. **Depth**: Which idea implies a deep change in society?
3. **Consistency**: Does it make sense within the context? (But prioritize 'interesting' over 'safe').

{proposals_text}

Output JSON:
{{ "selected_agent": "Name", "selected_content": "Content", "reason": "Reason for selection" }}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return parse_json_response(response.choices[0].message.content)

    def _final_judge(self, iteration_results, element_type, topic):
        """Selects the best result from the iterations."""
        iter_text = ""
        for res in iteration_results:
            iter_text += f"Iteration {res['iteration']}: {res['judgment']['selected_content']} (Reason: {res['judgment']['reason']})\n"
        
        prompt = f"""
Final Decision for "{element_type}" in "{topic}" (Stage 3).
Here are the winners of separate brainstorming iterations:
{iter_text}

Choose the absolute best final content for this element.
Output JSON:
{{ "final_content": "The final refined content text", "reason": "Final justification" }}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return parse_json_response(response.choices[0].message.content)

    def run_multi_agent_generation(self, element_type, element_desc, topic, full_context_str):
        """
        Runs the multi-agent loop based on NUM_ITERATIONS.
        """
        print(f"  > Generating '{element_type}'...")
        iteration_results = []
        agent_history = {agent['name']: [] for agent in self.agents}

        # Run Iterations based on Config
        for i in range(1, NUM_ITERATIONS + 1):
            # Parallel Agent Generation
            proposals = []
            # ThreadPool max_workers set to NUM_AGENTS to ensure full parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_AGENTS) as executor:
                future_to_agent = {
                    executor.submit(self._agent_think, agent, f"{element_type} ({element_desc})", full_context_str, agent_history[agent['name']]): agent 
                    for agent in self.agents
                }
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        content = future.result()
                        proposals.append({"agent": agent['name'], "content": content})
                        agent_history[agent['name']].append(content)
                    except Exception as e:
                        print(f"    Agent {agent['name']} failed: {e}")

            if not proposals:
                continue

            # Judge this iteration
            judgment = self._judge_proposals(proposals, element_type, topic)
            iteration_results.append({"iteration": i, "judgment": judgment})

        # Final Decision
        final_result = self._final_judge(iteration_results, element_type, topic)
        print(f"    -> Final Decision: {final_result.get('final_content')[:50]}...")
        return final_result.get("final_content")