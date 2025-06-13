"""Handles meta-game system commands like save, load, help, settings."""

import uuid
from typing import Any

from ...config.manager import ConfigManager
from ...database.session_factory import DatabaseSessionFactory
from ...llm.ollama.client import OllamaClient
from ...search.semantic_search import SemanticSearchService
from ...state.manager import GameStateManager
from ...state.models import ActionResult
from ..help.help_system import HelpSystem
from ..models.system_models import SystemCommandType
from ..save_system.save_manager import SaveManager
from ..settings.settings_manager import SettingsManager
from ..tutorial.tutorial_manager import TutorialManager


class SystemCommandProcessor:
    """Handles meta-game system commands like save, load, help, settings."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        game_state_manager: GameStateManager,
        config_manager: ConfigManager,
        llm_client: OllamaClient,
        semantic_search: SemanticSearchService | None = None,
    ):
        self.session_factory = session_factory
        self.game_state = game_state_manager
        self.config_manager = config_manager
        self.llm_client = llm_client
        self.semantic_search = semantic_search

        # Initialize subsystems
        self.save_manager = SaveManager(session_factory, game_state_manager)

        # Only initialize help system if semantic_search is available
        self.help_system = None
        if semantic_search:
            self.help_system = HelpSystem(session_factory, llm_client, semantic_search)

        self.tutorial_manager = TutorialManager(
            session_factory, game_state_manager, llm_client
        )
        self.settings_manager = SettingsManager(session_factory, config_manager)

        # Track initialization
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the system command processor."""
        if not self._initialized:
            if self.help_system:
                await self.help_system.initialize()
            await self.settings_manager.initialize()
            self._initialized = True

    async def process_command(
        self,
        command_type: SystemCommandType,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> ActionResult:
        """Route system commands to appropriate handlers."""
        try:
            if not self._initialized:
                await self.initialize()

            player_id = context.get("player_id")
            if player_id:
                player_id = uuid.UUID(str(player_id))

            # Route to appropriate handler
            if command_type == SystemCommandType.SAVE_GAME:
                return await self.handle_save_game(
                    player_id, args.get("save_name"), context
                )
            elif command_type == SystemCommandType.LOAD_GAME:
                return await self.handle_load_game(
                    player_id, args.get("save_name"), context
                )
            elif command_type == SystemCommandType.HELP:
                return await self.handle_help_request(args.get("topic"), context)
            elif command_type == SystemCommandType.TUTORIAL:
                return await self.handle_tutorial_request(
                    args.get("tutorial_type"), context
                )
            elif command_type == SystemCommandType.SETTINGS:
                return await self.handle_settings_command(
                    args.get("setting"), args.get("value"), player_id, context
                )
            elif command_type == SystemCommandType.LIST_SAVES:
                return await self.handle_list_saves(player_id, context)
            elif command_type == SystemCommandType.QUIT_GAME:
                return await self.handle_game_exit(args.get("force", False), context)
            else:
                return ActionResult(
                    success=False,
                    feedback_message=f"Unknown system command: {command_type.value}",
                )

        except Exception as e:
            return ActionResult(
                success=False,
                feedback_message=f"Error processing system command: {str(e)}",
            )

    async def handle_save_game(
        self,
        player_id: uuid.UUID | None,
        save_name: str | None,
        context: dict[str, Any],
    ) -> ActionResult:
        """Save current game state with optional custom name."""
        try:
            if not player_id:
                return ActionResult(
                    success=False,
                    feedback_message="Cannot save game: No player context available",
                )

            result = await self.save_manager.create_save(
                save_name=save_name,
                description=f"Manual save at {context.get('current_location', 'unknown location')}",
            )

            if result.success:
                return ActionResult(
                    success=True,
                    feedback_message=result.message,
                )
            else:
                return ActionResult(
                    success=False,
                    feedback_message=result.message,
                )

        except Exception as e:
            return ActionResult(
                success=False,
                feedback_message=f"Failed to save game: {str(e)}",
            )

    async def handle_load_game(
        self,
        player_id: uuid.UUID | None,
        save_name: str | None,
        context: dict[str, Any],
    ) -> ActionResult:
        """Load saved game state."""
        try:
            if not player_id:
                return ActionResult(
                    success=False,
                    feedback_message="Cannot load game: No player context available",
                )

            # If no save name provided, try to load the most recent save
            if not save_name:
                saves = await self.save_manager.list_saves()
                if not saves:
                    return ActionResult(
                        success=False,
                        feedback_message="No saved games found",
                    )
                save_name = saves[0].save_name

            result = await self.save_manager.load_save(save_name)

            if result.success:
                return ActionResult(
                    success=True,
                    feedback_message=result.message,
                )
            else:
                return ActionResult(
                    success=False,
                    feedback_message=result.message,
                )

        except Exception as e:
            return ActionResult(
                success=False,
                feedback_message=f"Failed to load game: {str(e)}",
            )

    async def handle_help_request(
        self, topic: str | None, context: dict[str, Any]
    ) -> ActionResult:
        """Provide contextual help information."""
        try:
            if not self.help_system:
                # Fallback help when semantic search is not available
                return ActionResult(
                    success=True,
                    feedback_message=self._get_basic_help(topic),
                )

            help_response = await self.help_system.get_help(topic, context)

            return ActionResult(
                success=True,
                feedback_message=help_response.content,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                feedback_message=f"Failed to get help: {str(e)}",
            )

    async def handle_tutorial_request(
        self, tutorial_type: str | None, context: dict[str, Any]
    ) -> ActionResult:
        """Start or continue tutorial guidance."""
        try:
            player_id = context.get("player_id")
            if not player_id:
                return ActionResult(
                    success=False,
                    feedback_message="Cannot start tutorial: No player context available",
                )

            player_id = uuid.UUID(str(player_id))

            # Check for tutorial triggers if no specific type requested
            if not tutorial_type:
                triggers = await self.tutorial_manager.check_tutorial_triggers(context)
                if triggers:
                    # Use the highest priority trigger
                    best_trigger = max(triggers, key=lambda t: t.priority)
                    return ActionResult(
                        success=True,
                        feedback_message=best_trigger.suggested_message,
                    )
                else:
                    return ActionResult(
                        success=True,
                        feedback_message="No specific tutorial needed right now. You're doing great!",
                    )

            # Get next hint for specified tutorial
            from ..models.system_models import TutorialType

            try:
                tutorial_enum = TutorialType(tutorial_type)
                hint = await self.tutorial_manager.get_next_hint(
                    player_id, tutorial_enum
                )

                if hint:
                    return ActionResult(
                        success=True,
                        feedback_message=hint.message,
                    )
                else:
                    return ActionResult(
                        success=True,
                        feedback_message=f"You've completed the {tutorial_type} tutorial!",
                    )

            except ValueError:
                return ActionResult(
                    success=False,
                    feedback_message=f"Unknown tutorial type: {tutorial_type}",
                )

        except Exception as e:
            return ActionResult(
                success=False,
                feedback_message=f"Failed to handle tutorial request: {str(e)}",
            )

    async def handle_settings_command(
        self,
        setting: str | None,
        value: str | None,
        player_id: uuid.UUID | None,
        context: dict[str, Any],
    ) -> ActionResult:
        """View or modify game settings."""
        try:
            # If no setting specified, list all settings
            if not setting:
                settings_list = await self.settings_manager.list_settings(
                    player_id=player_id
                )
                if not settings_list:
                    return ActionResult(
                        success=True,
                        feedback_message="No settings available",
                    )

                # Format settings for display
                settings_text = "**Current Settings:**\n\n"
                current_category = None

                for setting_info in settings_list:
                    if setting_info.category != current_category:
                        current_category = setting_info.category
                        settings_text += f"\n**{current_category.title()}:**\n"

                    settings_text += (
                        f"• {setting_info.name}: {setting_info.current_value}\n"
                    )
                    settings_text += f"  {setting_info.description}\n"

                return ActionResult(
                    success=True,
                    feedback_message=settings_text,
                )

            # If no value specified, show current setting
            if value is None:
                current_value = await self.settings_manager.get_setting(
                    setting, player_id
                )
                if current_value is None:
                    return ActionResult(
                        success=False,
                        feedback_message=f"Unknown setting: {setting}",
                    )

                return ActionResult(
                    success=True,
                    feedback_message=f"Setting '{setting}' is currently: {current_value}",
                )

            # Set the new value
            result = await self.settings_manager.set_setting(setting, value, player_id)

            return ActionResult(
                success=result.success,
                feedback_message=result.message,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                feedback_message=f"Failed to handle settings command: {str(e)}",
            )

    async def handle_list_saves(
        self, player_id: uuid.UUID | None, context: dict[str, Any]
    ) -> ActionResult:
        """List all available save files."""
        try:
            if not player_id:
                return ActionResult(
                    success=False,
                    feedback_message="Cannot list saves: No player context available",
                )

            saves = await self.save_manager.list_saves()

            if not saves:
                return ActionResult(
                    success=True,
                    feedback_message="No saved games found. Use 'save game' to create your first save.",
                )

            # Format saves for display
            saves_text = "**Your Saved Games:**\n\n"
            for i, save in enumerate(saves[:10], 1):  # Limit to 10 most recent
                saves_text += f"{i}. **{save.save_name}**\n"
                saves_text += f"   {save.description}\n"
                saves_text += (
                    f"   Created: {save.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                )
                saves_text += f"   Location: {save.location}\n\n"

            return ActionResult(
                success=True,
                feedback_message=saves_text,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                feedback_message=f"Failed to list saves: {str(e)}",
            )

    async def handle_game_exit(
        self, force: bool, context: dict[str, Any]
    ) -> ActionResult:
        """Handle game exit with optional auto-save."""
        try:
            player_id = context.get("player_id")

            # Offer to save if not forcing exit
            if not force and player_id:
                player_id = uuid.UUID(str(player_id))

                # Check if auto-save is enabled
                auto_save_enabled = await self.settings_manager.get_setting(
                    "auto_save_on_exit", player_id
                )

                if auto_save_enabled:
                    # Perform auto-save
                    save_result = await self.save_manager.auto_save()
                    if save_result.success:
                        return ActionResult(
                            success=True,
                            feedback_message="Game saved automatically. Thanks for playing!",
                        )
                    else:
                        return ActionResult(
                            success=True,
                            feedback_message="Warning: Could not auto-save. Exiting anyway. Thanks for playing!",
                        )
                else:
                    return ActionResult(
                        success=True,
                        feedback_message="Don't forget to save your progress! Thanks for playing!",
                    )
            else:
                return ActionResult(
                    success=True,
                    feedback_message="Thanks for playing!",
                )

        except Exception as e:
            return ActionResult(
                success=True,  # Always succeed exit
                feedback_message=f"Exiting game (with errors: {str(e)})",
            )

    def _get_basic_help(self, topic: str | None) -> str:
        """Provide basic help when semantic search is not available."""
        if not topic:
            return """Game Commands:
- go/move <direction>: Move in a direction (north, south, east, west, etc.)
- look: Look around the current location
- examine <object>: Look closely at an object
- inventory/i: Check your inventory
- take/get <item>: Pick up an item
- drop <item>: Drop an item from your inventory
- use <item> [on <target>]: Use an item, optionally on a target
- talk to <character>: Talk to a character

System Commands:
- save [name]: Save your game (optionally with a name)
- load [name]: Load a saved game
- list saves: Show all your saved games
- settings: View or modify game settings
- help [topic]: Show help (optionally for a specific topic)
- tutorial: Get tutorial guidance
- quit/exit: Quit the game"""

        # Topic-specific help
        topic_help = {
            "movement": "Use 'go <direction>' or just '<direction>' to move. Valid directions: north, south, east, west, up, down. You can also use abbreviations like 'n', 's', 'e', 'w'.",
            "inventory": "Use 'inventory' or 'i' to see your items. Use 'take <item>' to pick up items and 'drop <item>' to drop them.",
            "combat": "Attack enemies with 'attack <target>' or use weapons with 'use <weapon> on <target>'.",
            "save": "Save your game with 'save' or 'save <name>'. Load with 'load' or 'load <name>'. See all saves with 'list saves'.",
            "conversation": "Talk to characters with 'talk to <name>' or just 'talk <name>'. Some NPCs may have valuable information or quests.",
        }

        return topic_help.get(
            topic.lower(),
            f"Sorry, no specific help available for '{topic}'. Try 'help' for general commands.",
        )
