#!/usr/bin/env python3
"""
Test script for LLM-powered NPC dialogue generation.
"""

import asyncio
import logging
from pathlib import Path

from rich.console import Console

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_loop.llm.ollama.client import OllamaClient
from game_loop.config.manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_npc_dialogue():
    """Test NPC dialogue generation with LLM."""
    console = Console()
    ollama_client = OllamaClient()
    config_manager = ConfigManager()
    
    # Set up prompt template directory
    project_root = Path(__file__).parent.parent
    prompt_dir = project_root / "src" / "game_loop" / "llm" / "prompts"
    config_manager.config.prompts.template_dir = str(prompt_dir)
    
    console.print("[bold cyan]Testing LLM-Powered NPC Dialogue Generation[/bold cyan]")
    console.print()
    
    # Test context for a security guard in an abandoned office
    context = {
        "npc_type": "security_guard",
        "location_name": "Abandoned Reception Area",
        "location_type": "office_space",
        "location_description": "Fluorescent lights flicker weakly overhead, casting eerie shadows across the reception area.",
        "player_context": {
            "exploration_style": "deep_explorer",
            "experience_level": "intermediate",
            "inventory_items": 2
        }
    }
    
    try:
        # Load the dialogue prompt template
        template_path = prompt_dir / "world_generation" / "npc_dialogue.txt"
        if not template_path.exists():
            console.print(f"[red]Template not found: {template_path}[/red]")
            return
            
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Simple template rendering (replace placeholders)
        prompt = template
        for key, value in context.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    prompt = prompt.replace(f"{{{{{key}.{sub_key}}}}}", str(sub_value))
            else:
                prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
        
        console.print("[bold]Sending prompt to LLM...[/bold]")
        console.print()
        
        # Generate dialogue
        from game_loop.llm.ollama.client import OllamaModelParameters
        
        params = OllamaModelParameters(
            model="qwen2.5:3b",
            temperature=0.85,
            top_p=0.9,
            max_tokens=600,
            format="json"
        )
        
        response = await ollama_client.generate_completion(
            prompt=prompt,
            params=params,
            raw_response=True
        )
        
        if response and "response" in response:
            console.print("[bold green]Generated Dialogue:[/bold green]")
            console.print(response["response"])
            console.print()
            
            # Try to parse as JSON
            try:
                import json
                dialogue_data = json.loads(response["response"])
                
                console.print("[bold]Parsed Dialogue Structure:[/bold]")
                console.print(f"Greeting: {dialogue_data.get('greeting', 'N/A')}")
                console.print(f"Personality: {', '.join(dialogue_data.get('personality_traits', []))}")
                
                if 'conversation_topics' in dialogue_data:
                    console.print("\n[bold]Conversation Topics:[/bold]")
                    for topic in dialogue_data['conversation_topics']:
                        console.print(f"- {topic['topic']}: {topic['dialogue']}")
                
                if 'local_knowledge' in dialogue_data:
                    console.print("\n[bold]Local Knowledge:[/bold]")
                    for knowledge in dialogue_data['local_knowledge']:
                        console.print(f"- {knowledge}")
                        
            except json.JSONDecodeError:
                console.print("[yellow]Response is not valid JSON, using as plain text[/yellow]")
        else:
            console.print("[red]No response from LLM[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_npc_dialogue())