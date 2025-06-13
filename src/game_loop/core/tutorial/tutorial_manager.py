"""Adaptive tutorial system that guides new players."""

import uuid
from datetime import datetime
from typing import Any

from ...database.session_factory import DatabaseSessionFactory
from ...llm.ollama.client import OllamaClient
from ...state.manager import GameStateManager
from ..models.system_models import (
    PlayerSkillLevel,
    TutorialHint,
    TutorialPrompt,
    TutorialSession,
    TutorialType,
)


class TutorialManager:
    """Adaptive tutorial system that guides new players."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        game_state_manager: GameStateManager,
        llm_client: OllamaClient,
    ):
        self.session_factory = session_factory
        self.game_state = game_state_manager
        self.llm_client = llm_client
        self.tutorial_steps = self._load_tutorial_content()
        self.player_progress: dict[uuid.UUID, dict[str, Any]] = {}

    async def check_tutorial_triggers(
        self, context: dict[str, Any]
    ) -> list[TutorialPrompt]:
        """Check if current situation should trigger tutorial hints."""
        try:
            player_id = context.get("player_id")
            if not player_id:
                return []

            player_id = uuid.UUID(str(player_id))
            triggers = []

            # Check player skill level
            skill_level = await self._assess_player_skill_level(player_id, context)

            # Only trigger tutorials for beginners and intermediates
            if skill_level == PlayerSkillLevel.ADVANCED:
                return []

            # Check various trigger conditions
            triggers.extend(await self._check_command_triggers(player_id, context))
            triggers.extend(await self._check_context_triggers(player_id, context))
            triggers.extend(await self._check_progress_triggers(player_id, context))

            return triggers

        except Exception:
            return []

    async def start_tutorial(
        self, player_id: uuid.UUID, tutorial_type: TutorialType
    ) -> TutorialSession:
        """Start a specific tutorial sequence."""
        try:
            # Check if tutorial already exists
            existing_session = await self._get_tutorial_session(
                player_id, tutorial_type
            )
            if existing_session and not existing_session.get("is_completed", False):
                return TutorialSession(
                    tutorial_type=tutorial_type,
                    player_id=player_id,
                    current_step=existing_session.get("current_step", 0),
                    completed_steps=existing_session.get("completed_steps", []),
                    started_at=datetime.fromisoformat(existing_session["started_at"]),
                    last_activity=datetime.now(),
                )

            # Create new tutorial session
            session = TutorialSession(
                tutorial_type=tutorial_type,
                player_id=player_id,
                current_step=0,
                completed_steps=[],
                started_at=datetime.now(),
                last_activity=datetime.now(),
            )

            # Store in database
            await self._store_tutorial_session(session)

            return session

        except Exception:
            # Return default session on error
            return TutorialSession(
                tutorial_type=tutorial_type,
                player_id=player_id,
            )

    async def get_next_hint(
        self, player_id: uuid.UUID, tutorial_type: TutorialType | None = None
    ) -> TutorialHint | None:
        """Get the next tutorial hint for the player."""
        try:
            # If no specific tutorial type, find active tutorials
            if not tutorial_type:
                active_tutorials = await self._get_active_tutorials(player_id)
                if not active_tutorials:
                    return None
                tutorial_type = TutorialType(active_tutorials[0]["tutorial_type"])

            # Get tutorial session
            session_data = await self._get_tutorial_session(player_id, tutorial_type)
            if not session_data:
                return None

            current_step = session_data.get("current_step", 0)
            completed_steps = session_data.get("completed_steps", [])

            # Get tutorial steps for this type
            tutorial_steps = self.tutorial_steps.get(tutorial_type, [])
            if current_step >= len(tutorial_steps):
                return None

            # Get the current step
            step_data = tutorial_steps[current_step]

            return TutorialHint(
                hint_type=step_data["type"],
                message=step_data["message"],
                suggested_action=step_data["suggested_action"],
                priority=step_data.get("priority", 1),
                tutorial_type=tutorial_type,
                step_number=current_step,
            )

        except Exception:
            return None

    async def advance_tutorial_step(
        self, player_id: uuid.UUID, tutorial_type: TutorialType, completed_step: int
    ) -> bool:
        """Mark a tutorial step as completed and advance."""
        try:
            session_data = await self._get_tutorial_session(player_id, tutorial_type)
            if not session_data:
                return False

            completed_steps = session_data.get("completed_steps", [])
            current_step = session_data.get("current_step", 0)

            # Mark step as completed
            if completed_step not in completed_steps:
                completed_steps.append(completed_step)

            # Advance to next step
            next_step = max(current_step, completed_step + 1)

            # Check if tutorial is complete
            tutorial_steps = self.tutorial_steps.get(tutorial_type, [])
            is_completed = next_step >= len(tutorial_steps)

            # Update in database
            await self._update_tutorial_progress(
                player_id, tutorial_type, next_step, completed_steps, is_completed
            )

            return True

        except Exception:
            return False

    def track_player_action(
        self, player_id: uuid.UUID, action: str, context: dict[str, Any]
    ) -> None:
        """Track player actions to assess tutorial needs."""
        try:
            if player_id not in self.player_progress:
                self.player_progress[player_id] = {
                    "total_actions": 0,
                    "action_types": {},
                    "last_actions": [],
                    "locations_visited": set(),
                    "npcs_talked_to": set(),
                    "objects_interacted": set(),
                }

            progress = self.player_progress[player_id]
            progress["total_actions"] += 1

            # Track action types
            action_type = self._classify_action_type(action)
            progress["action_types"][action_type] = (
                progress["action_types"].get(action_type, 0) + 1
            )

            # Track recent actions
            progress["last_actions"].append(action)
            if len(progress["last_actions"]) > 10:
                progress["last_actions"] = progress["last_actions"][-10:]

            # Track context-specific progress
            if context.get("current_location"):
                progress["locations_visited"].add(context["current_location"])

            if context.get("npc_interacted"):
                progress["npcs_talked_to"].add(context["npc_interacted"])

            if context.get("object_used"):
                progress["objects_interacted"].add(context["object_used"])

        except Exception:
            pass  # Non-critical operation

    async def _assess_player_skill_level(
        self, player_id: uuid.UUID, context: dict[str, Any]
    ) -> PlayerSkillLevel:
        """Assess player skill level based on their actions and progress."""
        try:
            progress = self.player_progress.get(player_id, {})
            total_actions = progress.get("total_actions", 0)
            action_types = progress.get("action_types", {})
            locations_visited = len(progress.get("locations_visited", set()))

            player_level = context.get("player_level", 1)

            # Calculate skill score
            skill_score = 0

            # Basic activity scoring
            if total_actions > 50:
                skill_score += 1
            if total_actions > 200:
                skill_score += 1

            # Diversity scoring
            if len(action_types) > 5:
                skill_score += 1
            if locations_visited > 3:
                skill_score += 1

            # Player level scoring
            if player_level > 3:
                skill_score += 1
            if player_level > 10:
                skill_score += 1

            # Determine skill level
            if skill_score <= 2:
                return PlayerSkillLevel.BEGINNER
            elif skill_score <= 4:
                return PlayerSkillLevel.INTERMEDIATE
            else:
                return PlayerSkillLevel.ADVANCED

        except Exception:
            return PlayerSkillLevel.BEGINNER

    def _classify_action_type(self, action: str) -> str:
        """Classify action into broad categories."""
        action_lower = action.lower()

        if any(
            word in action_lower
            for word in [
                "go",
                "move",
                "north",
                "south",
                "east",
                "west",
                "enter",
                "exit",
            ]
        ):
            return "movement"
        elif any(
            word in action_lower for word in ["take", "get", "pick", "drop", "put"]
        ):
            return "inventory"
        elif any(
            word in action_lower for word in ["look", "examine", "inspect", "search"]
        ):
            return "observation"
        elif any(
            word in action_lower for word in ["talk", "say", "ask", "tell", "greet"]
        ):
            return "conversation"
        elif any(
            word in action_lower
            for word in ["use", "open", "close", "unlock", "activate"]
        ):
            return "interaction"
        elif any(
            word in action_lower
            for word in ["save", "load", "help", "quit", "settings"]
        ):
            return "system"
        else:
            return "other"

    async def _check_command_triggers(
        self, player_id: uuid.UUID, context: dict[str, Any]
    ) -> list[TutorialPrompt]:
        """Check for command-based tutorial triggers."""
        triggers = []
        progress = self.player_progress.get(player_id, {})
        total_actions = progress.get("total_actions", 0)
        action_types = progress.get("action_types", {})

        # Basic commands tutorial
        if total_actions < 5:
            triggers.append(
                TutorialPrompt(
                    tutorial_type=TutorialType.BASIC_COMMANDS,
                    trigger_reason="New player detected",
                    suggested_message="Let me help you get started with basic commands!",
                    priority=3,
                )
            )

        # Movement tutorial
        if action_types.get("movement", 0) < 3 and total_actions > 5:
            triggers.append(
                TutorialPrompt(
                    tutorial_type=TutorialType.MOVEMENT,
                    trigger_reason="Player hasn't moved much",
                    suggested_message="Would you like to learn about movement and navigation?",
                    priority=2,
                )
            )

        # Inventory tutorial
        if action_types.get("inventory", 0) == 0 and total_actions > 10:
            triggers.append(
                TutorialPrompt(
                    tutorial_type=TutorialType.INVENTORY,
                    trigger_reason="Player hasn't used inventory",
                    suggested_message="Let me show you how to manage your inventory!",
                    priority=2,
                )
            )

        return triggers

    async def _check_context_triggers(
        self, player_id: uuid.UUID, context: dict[str, Any]
    ) -> list[TutorialPrompt]:
        """Check for context-based tutorial triggers."""
        triggers = []

        # Object interaction tutorial
        nearby_objects = context.get("nearby_objects", [])
        if nearby_objects and not await self._has_completed_tutorial(
            player_id, TutorialType.OBJECT_INTERACTION
        ):
            triggers.append(
                TutorialPrompt(
                    tutorial_type=TutorialType.OBJECT_INTERACTION,
                    trigger_reason="Objects available for interaction",
                    suggested_message="I notice there are objects here you can interact with!",
                    priority=2,
                )
            )

        # Conversation tutorial
        nearby_npcs = context.get("nearby_npcs", [])
        if nearby_npcs and not await self._has_completed_tutorial(
            player_id, TutorialType.CONVERSATION
        ):
            triggers.append(
                TutorialPrompt(
                    tutorial_type=TutorialType.CONVERSATION,
                    trigger_reason="NPCs available for conversation",
                    suggested_message="There are people here you can talk to!",
                    priority=2,
                )
            )

        return triggers

    async def _check_progress_triggers(
        self, player_id: uuid.UUID, context: dict[str, Any]
    ) -> list[TutorialPrompt]:
        """Check for progress-based tutorial triggers."""
        triggers = []

        # Quest tutorial when player gets their first quest
        current_quest = context.get("current_quest")
        if current_quest and not await self._has_completed_tutorial(
            player_id, TutorialType.QUESTS
        ):
            triggers.append(
                TutorialPrompt(
                    tutorial_type=TutorialType.QUESTS,
                    trigger_reason="Player has a quest",
                    suggested_message="You have a quest! Let me explain how quests work.",
                    priority=2,
                )
            )

        return triggers

    def _load_tutorial_content(self) -> dict[TutorialType, list[dict[str, Any]]]:
        """Load tutorial step definitions."""
        return {
            TutorialType.BASIC_COMMANDS: [
                {
                    "type": "welcome",
                    "message": "Welcome to the game! Let's start with basic commands.",
                    "suggested_action": "Try typing 'look around' to see your surroundings.",
                    "priority": 3,
                },
                {
                    "type": "observation",
                    "message": "Great! You can examine things more closely too.",
                    "suggested_action": "Try 'examine <object>' to learn more about something.",
                    "priority": 2,
                },
                {
                    "type": "help",
                    "message": "If you ever need help, just ask!",
                    "suggested_action": "Type 'help' anytime for assistance.",
                    "priority": 1,
                },
            ],
            TutorialType.MOVEMENT: [
                {
                    "type": "directions",
                    "message": "You can move in different directions.",
                    "suggested_action": "Try 'go north' or just 'north' to move.",
                    "priority": 2,
                },
                {
                    "type": "exploration",
                    "message": "Explore different areas to discover new things!",
                    "suggested_action": "Try moving to different locations and looking around.",
                    "priority": 1,
                },
            ],
            TutorialType.INVENTORY: [
                {
                    "type": "checking",
                    "message": "You can check what you're carrying.",
                    "suggested_action": "Type 'inventory' to see your items.",
                    "priority": 2,
                },
                {
                    "type": "taking",
                    "message": "You can pick up objects you find.",
                    "suggested_action": "Try 'take <object>' to pick something up.",
                    "priority": 2,
                },
                {
                    "type": "using",
                    "message": "Items in your inventory can be used.",
                    "suggested_action": "Try 'use <item>' to use something you're carrying.",
                    "priority": 1,
                },
            ],
            TutorialType.OBJECT_INTERACTION: [
                {
                    "type": "examining",
                    "message": "You can examine objects to learn about them.",
                    "suggested_action": "Try 'examine <object>' for detailed information.",
                    "priority": 2,
                },
                {
                    "type": "interacting",
                    "message": "Many objects can be interacted with in different ways.",
                    "suggested_action": "Try 'use <object>' or 'open <object>' to interact.",
                    "priority": 1,
                },
            ],
            TutorialType.CONVERSATION: [
                {
                    "type": "greeting",
                    "message": "You can talk to people you meet.",
                    "suggested_action": "Try 'talk to <person>' to start a conversation.",
                    "priority": 2,
                },
                {
                    "type": "asking",
                    "message": "You can ask about specific topics.",
                    "suggested_action": "Try 'ask <person> about <topic>' for information.",
                    "priority": 1,
                },
            ],
            TutorialType.QUESTS: [
                {
                    "type": "status",
                    "message": "You can check your quest progress.",
                    "suggested_action": "Type 'quest status' to see your current quests.",
                    "priority": 2,
                },
                {
                    "type": "completion",
                    "message": "Complete objectives to advance your quests.",
                    "suggested_action": "Follow the quest objectives to make progress.",
                    "priority": 1,
                },
            ],
        }

    async def _get_tutorial_session(
        self, player_id: uuid.UUID, tutorial_type: TutorialType
    ) -> dict[str, Any] | None:
        """Get tutorial session from database."""
        try:
            async with self.session_factory.get_session() as session:
                result = await session.execute(
                    """
                    SELECT current_step, completed_steps, skill_level, started_at, 
                           last_activity, is_completed
                    FROM tutorial_progress 
                    WHERE player_id = $1 AND tutorial_type = $2
                    """,
                    (player_id, tutorial_type.value),
                )
                row = result.fetchone()

                if row:
                    return {
                        "current_step": row[0],
                        "completed_steps": row[1] or [],
                        "skill_level": row[2],
                        "started_at": row[3].isoformat(),
                        "last_activity": row[4].isoformat(),
                        "is_completed": row[5],
                    }

                return None

        except Exception:
            return None

    async def _store_tutorial_session(self, session: TutorialSession) -> None:
        """Store tutorial session in database."""
        try:
            async with self.session_factory.get_session() as db_session:
                await db_session.execute(
                    """
                    INSERT INTO tutorial_progress 
                    (player_id, tutorial_type, current_step, completed_steps, 
                     skill_level, started_at, last_activity)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (player_id, tutorial_type) 
                    DO UPDATE SET
                        current_step = EXCLUDED.current_step,
                        completed_steps = EXCLUDED.completed_steps,
                        last_activity = EXCLUDED.last_activity
                    """,
                    (
                        session.player_id,
                        session.tutorial_type.value,
                        session.current_step,
                        session.completed_steps,
                        "beginner",  # Default skill level
                        session.started_at,
                        session.last_activity,
                    ),
                )
                await db_session.commit()

        except Exception:
            pass  # Non-critical operation

    async def _update_tutorial_progress(
        self,
        player_id: uuid.UUID,
        tutorial_type: TutorialType,
        current_step: int,
        completed_steps: list[int],
        is_completed: bool,
    ) -> None:
        """Update tutorial progress in database."""
        try:
            async with self.session_factory.get_session() as session:
                await session.execute(
                    """
                    UPDATE tutorial_progress 
                    SET current_step = $3, completed_steps = $4, 
                        is_completed = $5, last_activity = CURRENT_TIMESTAMP
                    WHERE player_id = $1 AND tutorial_type = $2
                    """,
                    (
                        player_id,
                        tutorial_type.value,
                        current_step,
                        completed_steps,
                        is_completed,
                    ),
                )
                await session.commit()

        except Exception:
            pass  # Non-critical operation

    async def _get_active_tutorials(self, player_id: uuid.UUID) -> list[dict[str, Any]]:
        """Get active tutorials for a player."""
        try:
            async with self.session_factory.get_session() as session:
                result = await session.execute(
                    """
                    SELECT tutorial_type, current_step, completed_steps
                    FROM tutorial_progress 
                    WHERE player_id = $1 AND is_completed = FALSE
                    ORDER BY last_activity DESC
                    """,
                    (player_id,),
                )

                return [
                    {
                        "tutorial_type": row[0],
                        "current_step": row[1],
                        "completed_steps": row[2] or [],
                    }
                    for row in result
                ]

        except Exception:
            return []

    async def _has_completed_tutorial(
        self, player_id: uuid.UUID, tutorial_type: TutorialType
    ) -> bool:
        """Check if player has completed a tutorial."""
        try:
            async with self.session_factory.get_session() as session:
                result = await session.execute(
                    """
                    SELECT is_completed 
                    FROM tutorial_progress 
                    WHERE player_id = $1 AND tutorial_type = $2
                    """,
                    (player_id, tutorial_type.value),
                )
                row = result.fetchone()

                return row[0] if row else False

        except Exception:
            return False
