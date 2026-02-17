"""
Persona Manager — Generates detailed farmer personas to contextualize question generation.
Acts as a separate agent (Persona Creator) in the multi-agent system.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .config import Config
from .agent import _build_fallback_model

logger = logging.getLogger(__name__)


class Persona(BaseModel):
    """Structured representation of a farmer persona."""
    name: str = Field(description="Name of the farmer")
    age: int = Field(description="Age of the farmer")
    region: str = Field(description="Region or state in India (e.g., Punjab, Bihar, Maharashtra)")
    farming_type: str = Field(description="Type of farming (e.g., subsistence, commercial, organic, dairy)")
    challenges: List[str] = Field(description="Key challenges faced (e.g., drought, pest, debt)")
    personality_traits: List[str] = Field(description="Personality traits (e.g., anxious, skeptical, optimistic, angry)")
    speaking_style: str = Field(description="Description of how they speak (e.g., 'Broken English', 'Formal', 'Urgent')")
    background_story: str = Field(description="A brief backstory (2-3 sentences)")

    def to_xml(self) -> str:
        """Convert persona to XML string for system prompt injection."""
        return f"""
<persona>
    <name>{self.name}</name>
    <age>{self.age}</age>
    <region>{self.region}</region>
    <farming_type>{self.farming_type}</farming_type>
    <challenges>{', '.join(self.challenges)}</challenges>
    <personality>{', '.join(self.personality_traits)}</personality>
    <speaking_style>{self.speaking_style}</speaking_style>
    <background>{self.background_story}</background>
</persona>
"""


class PersonaManager:
    """
    Manages the creation and retrieval of personas.
    Uses a dedicated pydantic-ai Agent to generate personas.
    """

    def __init__(self, config: Config):
        self.config = config
        self.agent = self._create_persona_agent()

    def _create_persona_agent(self) -> Agent:
        """Create the Persona Creator Agent."""
        model = _build_fallback_model(self.config)
        
        return Agent(
            model,
            output_type=Persona,
            system_prompt=(
                "You are an expert creative writer and agricultural sociologist. "
                "Your task is to create detailed, realistic personas of Indian farmers. "
                "These personas will be used to test an agricultural chatbot. "
                "Include diverse demographics, regions, and challenges. "
                "The persona should feel authentic and specific."
            ),
            retries=self.config.MAX_RETRIES,
        )

    def generate_persona(self, context: Optional[str] = None) -> Persona:
        """
        Generate a new persona using the agent.
        
        Args:
            context: Optional string to guide generation (e.g., "An angry farmer from Punjab").
        """
        prompt = "Generate a detailed Indian farmer persona."
        if context:
            prompt += f"\n\nContext/Requirements: {context}"
        else:
            prompt += "\n\nEnsure diversity in region and farming type."

        logger.info("Generating new persona with context: %s", context or "None")
        
        # synchronous run
        result = self.agent.run_sync(prompt)
        persona = result.output
        logger.info("Generated persona: %s from %s", persona.name, persona.region)
        return persona
