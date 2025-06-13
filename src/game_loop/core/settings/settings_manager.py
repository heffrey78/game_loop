"""Manage user preferences and game settings."""

import json
import uuid
from typing import Any

from ...config.manager import ConfigManager
from ...database.session_factory import DatabaseSessionFactory
from ..models.system_models import SettingDefinition, SettingInfo, SettingResult


class SettingsManager:
    """Manage user preferences and game settings."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        config_manager: ConfigManager,
    ):
        self.session_factory = session_factory
        self.config = config_manager
        self.user_settings: dict[uuid.UUID, dict[str, Any]] = {}
        self.setting_definitions: dict[str, SettingDefinition] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize settings manager by loading definitions."""
        if not self._initialized:
            await self._load_setting_definitions()
            self._initialized = True

    async def get_setting(
        self, setting_name: str, player_id: uuid.UUID | None = None
    ) -> Any:
        """Get current value of a setting."""
        try:
            if not self._initialized:
                await self.initialize()

            # Check if setting exists
            if setting_name not in self.setting_definitions:
                return None

            definition = self.setting_definitions[setting_name]

            # For global settings (no player_id required)
            if player_id is None:
                return await self._get_global_setting(
                    setting_name, definition.default_value
                )

            # For user-specific settings
            user_value = await self._get_user_setting(player_id, setting_name)
            if user_value is not None:
                return user_value

            # Fall back to default
            return definition.default_value

        except Exception:
            # Return default value if available
            definition = self.setting_definitions.get(setting_name)
            return definition.default_value if definition else None

    async def set_setting(
        self, setting_name: str, value: Any, player_id: uuid.UUID | None = None
    ) -> SettingResult:
        """Set a setting value with validation."""
        try:
            if not self._initialized:
                await self.initialize()

            # Check if setting exists
            if setting_name not in self.setting_definitions:
                return SettingResult(
                    success=False,
                    setting_name=setting_name,
                    old_value=None,
                    new_value=value,
                    message=f"Unknown setting: {setting_name}",
                    error="Setting not found",
                )

            definition = self.setting_definitions[setting_name]

            # Get current value
            old_value = await self.get_setting(setting_name, player_id)

            # Validate new value
            is_valid, validation_error = self.validate_setting_value(
                setting_name, value
            )
            if not is_valid:
                return SettingResult(
                    success=False,
                    setting_name=setting_name,
                    old_value=old_value,
                    new_value=value,
                    message=f"Invalid value for {setting_name}: {validation_error}",
                    error=validation_error,
                )

            # Convert value to appropriate type
            converted_value = self._convert_value(value, definition.value_type)

            # Store setting
            if player_id is None:
                await self._set_global_setting(setting_name, converted_value)
            else:
                await self._set_user_setting(
                    player_id, setting_name, converted_value, definition.category
                )

            return SettingResult(
                success=True,
                setting_name=setting_name,
                old_value=old_value,
                new_value=converted_value,
                message=f"Setting {setting_name} updated successfully",
            )

        except Exception as e:
            return SettingResult(
                success=False,
                setting_name=setting_name,
                old_value=None,
                new_value=value,
                message=f"Failed to update setting: {str(e)}",
                error=str(e),
            )

    async def list_settings(
        self, category: str | None = None, player_id: uuid.UUID | None = None
    ) -> list[SettingInfo]:
        """List available settings with descriptions."""
        try:
            if not self._initialized:
                await self.initialize()

            settings_info = []

            for setting_name, definition in self.setting_definitions.items():
                # Filter by category if specified
                if category and definition.category != category:
                    continue

                # Get current value
                current_value = await self.get_setting(setting_name, player_id)

                setting_info = SettingInfo(
                    name=setting_name,
                    description=definition.description,
                    current_value=current_value,
                    default_value=definition.default_value,
                    allowed_values=definition.allowed_values,
                    category=definition.category,
                    value_type=definition.value_type,
                )
                settings_info.append(setting_info)

            # Sort by category and name
            settings_info.sort(key=lambda x: (x.category, x.name))
            return settings_info

        except Exception:
            return []

    async def reset_settings(
        self, category: str | None = None, player_id: uuid.UUID | None = None
    ) -> bool:
        """Reset settings to defaults."""
        try:
            if not self._initialized:
                await self.initialize()

            settings_to_reset = []

            # Determine which settings to reset
            for setting_name, definition in self.setting_definitions.items():
                if category is None or definition.category == category:
                    settings_to_reset.append(setting_name)

            # Reset each setting
            for setting_name in settings_to_reset:
                definition = self.setting_definitions[setting_name]
                if player_id is None:
                    await self._set_global_setting(
                        setting_name, definition.default_value
                    )
                else:
                    await self._delete_user_setting(player_id, setting_name)

            return True

        except Exception:
            return False

    def validate_setting_value(
        self, setting_name: str, value: Any
    ) -> tuple[bool, str | None]:
        """Validate a setting value against constraints."""
        try:
            definition = self.setting_definitions.get(setting_name)
            if not definition:
                return False, "Setting not found"

            # Type validation
            if definition.value_type == "boolean":
                if not isinstance(value, bool) and str(value).lower() not in [
                    "true",
                    "false",
                    "1",
                    "0",
                ]:
                    return False, "Must be a boolean value (true/false)"

            elif definition.value_type == "integer":
                try:
                    int_value = int(value)
                except (ValueError, TypeError):
                    return False, "Must be an integer"

                # Check validation rules
                if "min" in definition.validation_rules:
                    if int_value < definition.validation_rules["min"]:
                        return (
                            False,
                            f"Must be at least {definition.validation_rules['min']}",
                        )
                if "max" in definition.validation_rules:
                    if int_value > definition.validation_rules["max"]:
                        return (
                            False,
                            f"Must be at most {definition.validation_rules['max']}",
                        )

            elif definition.value_type == "float":
                try:
                    float_value = float(value)
                except (ValueError, TypeError):
                    return False, "Must be a number"

                # Check validation rules
                if "min" in definition.validation_rules:
                    if float_value < definition.validation_rules["min"]:
                        return (
                            False,
                            f"Must be at least {definition.validation_rules['min']}",
                        )
                if "max" in definition.validation_rules:
                    if float_value > definition.validation_rules["max"]:
                        return (
                            False,
                            f"Must be at most {definition.validation_rules['max']}",
                        )

            elif definition.value_type == "string":
                str_value = str(value)

                # Check allowed values
                if (
                    definition.allowed_values
                    and str_value not in definition.allowed_values
                ):
                    return (
                        False,
                        f"Must be one of: {', '.join(map(str, definition.allowed_values))}",
                    )

                # Check validation rules
                if "min_length" in definition.validation_rules:
                    if len(str_value) < definition.validation_rules["min_length"]:
                        return (
                            False,
                            f"Must be at least {definition.validation_rules['min_length']} characters",
                        )
                if "max_length" in definition.validation_rules:
                    if len(str_value) > definition.validation_rules["max_length"]:
                        return (
                            False,
                            f"Must be at most {definition.validation_rules['max_length']} characters",
                        )

            # Check allowed values for any type
            if definition.allowed_values:
                converted_value = self._convert_value(value, definition.value_type)
                if converted_value not in definition.allowed_values:
                    return (
                        False,
                        f"Must be one of: {', '.join(map(str, definition.allowed_values))}",
                    )

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _convert_value(self, value: Any, value_type: str) -> Any:
        """Convert value to appropriate type."""
        if value_type == "boolean":
            if isinstance(value, bool):
                return value
            elif str(value).lower() in ["true", "1"]:
                return True
            elif str(value).lower() in ["false", "0"]:
                return False
            else:
                raise ValueError(f"Cannot convert {value} to boolean")

        elif value_type == "integer":
            return int(value)

        elif value_type == "float":
            return float(value)

        elif value_type == "string":
            return str(value)

        else:
            return value

    async def _load_setting_definitions(self) -> None:
        """Load setting definitions from database."""
        try:
            async with self.session_factory.get_session() as session:
                result = await session.execute(
                    """
                    SELECT setting_name, description, default_value, allowed_values,
                           value_type, category, validation_rules
                    FROM setting_definitions
                    ORDER BY category, setting_name
                    """
                )

                self.setting_definitions = {}
                for row in result:
                    definition = SettingDefinition(
                        name=row[0],
                        description=row[1],
                        default_value=self._parse_json_value(row[2]),
                        allowed_values=(
                            self._parse_json_value(row[3]) if row[3] else None
                        ),
                        value_type=row[4],
                        category=row[5],
                        validation_rules=self._parse_json_value(row[6]) or {},
                    )
                    self.setting_definitions[row[0]] = definition

        except Exception:
            # Load default definitions if database fails
            self._load_default_definitions()

    def _load_default_definitions(self) -> None:
        """Load default setting definitions as fallback."""
        defaults = [
            SettingDefinition(
                name="auto_save_interval",
                description="Automatic save interval in minutes",
                default_value=5,
                value_type="integer",
                category="gameplay",
                validation_rules={"min": 1, "max": 60},
            ),
            SettingDefinition(
                name="tutorial_enabled",
                description="Enable tutorial hints and guidance",
                default_value=True,
                value_type="boolean",
                category="interface",
            ),
            SettingDefinition(
                name="help_verbosity",
                description="Level of detail in help responses",
                default_value="detailed",
                allowed_values=["brief", "normal", "detailed"],
                value_type="string",
                category="interface",
            ),
            SettingDefinition(
                name="color_output",
                description="Enable colored text output",
                default_value=True,
                value_type="boolean",
                category="display",
            ),
        ]

        self.setting_definitions = {def_.name: def_ for def_ in defaults}

    def _parse_json_value(self, value: Any) -> Any:
        """Parse JSON value safely."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    async def _get_global_setting(self, setting_name: str, default_value: Any) -> Any:
        """Get global setting value."""
        # For now, use config manager for global settings
        try:
            return getattr(self.config.config, setting_name, default_value)
        except Exception:
            return default_value

    async def _set_global_setting(self, setting_name: str, value: Any) -> None:
        """Set global setting value."""
        # For now, this would update the config manager
        # In a full implementation, this might update a global settings table
        pass

    async def _get_user_setting(
        self, player_id: uuid.UUID, setting_name: str
    ) -> Any | None:
        """Get user-specific setting value."""
        try:
            async with self.session_factory.get_session() as session:
                result = await session.execute(
                    """
                    SELECT setting_value 
                    FROM user_settings 
                    WHERE player_id = $1 AND setting_name = $2
                    """,
                    (player_id, setting_name),
                )
                row = result.fetchone()

                if row:
                    return self._parse_json_value(row[0])

                return None

        except Exception:
            return None

    async def _set_user_setting(
        self, player_id: uuid.UUID, setting_name: str, value: Any, category: str
    ) -> None:
        """Set user-specific setting value."""
        try:
            async with self.session_factory.get_session() as session:
                await session.execute(
                    """
                    INSERT INTO user_settings 
                    (player_id, setting_name, setting_value, setting_category)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (player_id, setting_name) 
                    DO UPDATE SET
                        setting_value = EXCLUDED.setting_value,
                        setting_category = EXCLUDED.setting_category,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (player_id, setting_name, json.dumps(value), category),
                )
                await session.commit()

        except Exception:
            pass  # Non-critical operation

    async def _delete_user_setting(
        self, player_id: uuid.UUID, setting_name: str
    ) -> None:
        """Delete user-specific setting (revert to default)."""
        try:
            async with self.session_factory.get_session() as session:
                await session.execute(
                    """
                    DELETE FROM user_settings 
                    WHERE player_id = $1 AND setting_name = $2
                    """,
                    (player_id, setting_name),
                )
                await session.commit()

        except Exception:
            pass  # Non-critical operation
