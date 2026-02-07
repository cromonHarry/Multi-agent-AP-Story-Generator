# AP-Based LLM Story Generation: Envisioning Technology Development through Science Fiction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)

This repository contains the official implementation of the paper: **"AP-Based LLM Story Generation: Envisioning Technology Development through Science Fiction"**.

Our system creates diverse and culturally grounded science fiction stories by combining the **Archaeological Prototyping (AP)** social-cultural modeling framework with a **Multi-Agent** competitive ideation process.

## 📖 Abstract

Science fiction serves as a powerful medium for exploring future technologies and their societal impact. However, existing AI-based story generation approaches often lack systematic grounding in social-cultural analysis. 

We present a novel multi-agent system that:
1.  **Constructs a Future World Model:** Uses the Archaeological Prototyping (AP) framework to model technology-society relationships (Stage 1 & 2) and employs a multi-agent competitive debate to predict mature future scenarios (Stage 3).
2.  **Generates Narratives:** Uses a hierarchical Overseer-Executor architecture where an Overseer Agent coordinates Setting and Outline Agents to ensure the story aligns with the sociological logic of the generated world.

Experiments show that this approach significantly improves story diversity (Keyword Diversity +38.5%, Story Diversity +22.4%) compared to baseline methods while maintaining high narrative quality.

## 🏗️ System Architecture

The framework consists of two interconnected multi-agent parts:

1.  **AP Multi-Agent Part:** Constructs the socio-technological model.
    * *Stage 1 & 2:* Evidence-based modeling using Tavily Search API.
    * *Stage 3:* Competitive ideation with 3 specialized agents and a Judge Agent.
2.  **Story Multi-Agent Part:** Hierarchical generation.
    * *Overseer Agent:* Manages consistency and briefs executors.
    * *Executor Agents:* Setting Agent, Outline Agent, and Writing Agent.

> *Note: Please refer to Figure 1 and Figure 4 in the paper for detailed architectural diagrams.*

## 🚀 Features

* **Sociological Grounding:** Uses the 18-element AP model (6 objects, 12 arrows) to map complex tech-society relationships.
* **Multi-Agent Debate:** Simulates diverse perspectives (e.g., Techno-optimist vs. Critical Theorist) to break the "diversity-quality trade-off."
* **Hierarchical Control:** "Overseer-Executor" pattern ensures that creative outputs adhere to the generated sociological constraints.
* **Automated Evaluation:** Includes evaluation scripts using a GPT-4o model fine-tuned on the HANNA dataset (correlation with human judgment: 0.72).

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/cromonHarry/Multi-agent-AP-Story-Generator.git](https://github.com/cromonHarry/Multi-agent-AP-Story-Generator.git)
    cd Multi-agent-AP-Story-Generator
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    TAVILY_API_KEY=your_tavily_api_key_here
    ```

## 💻 Usage

### 1. Generate a Story
Run the main script with your desired topic (theme):

```bash
python main.py --topic "Umbrella" --rounds 2
