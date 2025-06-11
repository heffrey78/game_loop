"""Quest repository for database operations."""

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.quests.models import (
    Quest,
    QuestCategory,
    QuestDifficulty,
    QuestProgress,
    QuestStatus,
    QuestStep,
)


class QuestRepository:
    """Repository for quest data persistence."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_quest(self, quest_id: str) -> Quest | None:
        """Get a quest by its ID."""
        query = text("SELECT * FROM quests WHERE quest_id = :quest_id")
        result = await self.session.execute(query, {"quest_id": quest_id})
        row = result.fetchone()

        if not row:
            return None

        return self._row_to_quest(row)

    async def create_quest(self, quest: Quest) -> bool:
        """Create a new quest."""
        try:
            # Convert quest steps to JSON format
            steps_json = json.dumps(
                [
                    {
                        "step_id": step.step_id,
                        "description": step.description,
                        "requirements": step.requirements,
                        "completion_conditions": step.completion_conditions,
                        "rewards": step.rewards,
                        "optional": step.optional,
                    }
                    for step in quest.steps
                ]
            )

            query = text(
                """
                INSERT INTO quests (
                    quest_id, title, description, category, difficulty,
                    steps, prerequisites, rewards, time_limit, repeatable
                ) VALUES (
                    :quest_id, :title, :description, :category, :difficulty,
                    :steps, :prerequisites, :rewards, :time_limit, :repeatable
                )
            """
            )

            await self.session.execute(
                query,
                {
                    "quest_id": quest.quest_id,
                    "title": quest.title,
                    "description": quest.description,
                    "category": quest.category.value,
                    "difficulty": quest.difficulty.value,
                    "steps": steps_json,
                    "prerequisites": json.dumps(quest.prerequisites),
                    "rewards": json.dumps(quest.rewards),
                    "time_limit": quest.time_limit,
                    "repeatable": quest.repeatable,
                },
            )

            await self.session.commit()
            return True

        except Exception:
            await self.session.rollback()
            return False

    async def update_quest(self, quest: Quest) -> bool:
        """Update an existing quest."""
        try:
            steps_json = json.dumps(
                [
                    {
                        "step_id": step.step_id,
                        "description": step.description,
                        "requirements": step.requirements,
                        "completion_conditions": step.completion_conditions,
                        "rewards": step.rewards,
                        "optional": step.optional,
                    }
                    for step in quest.steps
                ]
            )

            query = text(
                """
                UPDATE quests SET 
                    title = :title,
                    description = :description,
                    category = :category,
                    difficulty = :difficulty,
                    steps = :steps,
                    prerequisites = :prerequisites,
                    rewards = :rewards,
                    time_limit = :time_limit,
                    repeatable = :repeatable,
                    updated_at = CURRENT_TIMESTAMP
                WHERE quest_id = :quest_id
            """
            )

            result = await self.session.execute(
                query,
                {
                    "quest_id": quest.quest_id,
                    "title": quest.title,
                    "description": quest.description,
                    "category": quest.category.value,
                    "difficulty": quest.difficulty.value,
                    "steps": steps_json,
                    "prerequisites": json.dumps(quest.prerequisites),
                    "rewards": json.dumps(quest.rewards),
                    "time_limit": quest.time_limit,
                    "repeatable": quest.repeatable,
                },
            )

            await self.session.commit()
            return result.rowcount > 0

        except Exception:
            await self.session.rollback()
            return False

    async def delete_quest(self, quest_id: str) -> bool:
        """Delete a quest."""
        try:
            query = text("DELETE FROM quests WHERE quest_id = :quest_id")
            result = await self.session.execute(query, {"quest_id": quest_id})

            await self.session.commit()
            return result.rowcount > 0

        except Exception:
            await self.session.rollback()
            return False

    async def get_player_progress(
        self, player_id: str, quest_id: str
    ) -> QuestProgress | None:
        """Get player's progress on a specific quest."""
        query = text(
            """
            SELECT * FROM quest_progress 
            WHERE player_id = :player_id AND quest_id = :quest_id
        """
        )

        result = await self.session.execute(
            query, {"player_id": player_id, "quest_id": quest_id}
        )
        row = result.fetchone()

        if not row:
            return None

        return self._row_to_progress(row)

    async def update_progress(self, progress: QuestProgress) -> bool:
        """Update or insert quest progress."""
        try:
            # Try to update existing progress first
            update_query = text(
                """
                UPDATE quest_progress SET 
                    status = :status,
                    current_step = :current_step,
                    completed_steps = :completed_steps,
                    step_progress = :step_progress,
                    updated_at = CURRENT_TIMESTAMP
                WHERE quest_id = :quest_id AND player_id = :player_id
            """
            )

            result = await self.session.execute(
                update_query,
                {
                    "quest_id": progress.quest_id,
                    "player_id": progress.player_id,
                    "status": progress.status.value,
                    "current_step": progress.current_step,
                    "completed_steps": json.dumps(progress.completed_steps),
                    "step_progress": json.dumps(progress.step_progress),
                },
            )

            if result.rowcount == 0:
                # If no rows updated, insert new progress
                insert_query = text(
                    """
                    INSERT INTO quest_progress (
                        quest_id, player_id, status, current_step,
                        completed_steps, step_progress, started_at, updated_at
                    ) VALUES (
                        :quest_id, :player_id, :status, :current_step,
                        :completed_steps, :step_progress, 
                        to_timestamp(:started_at), to_timestamp(:updated_at)
                    )
                """
                )

                await self.session.execute(
                    insert_query,
                    {
                        "quest_id": progress.quest_id,
                        "player_id": progress.player_id,
                        "status": progress.status.value,
                        "current_step": progress.current_step,
                        "completed_steps": json.dumps(progress.completed_steps),
                        "step_progress": json.dumps(progress.step_progress),
                        "started_at": progress.started_at,
                        "updated_at": progress.updated_at,
                    },
                )

            await self.session.commit()
            return True

        except Exception:
            await self.session.rollback()
            return False

    async def get_available_quests(
        self, player_id: str, location_id: str | None = None
    ) -> list[Quest]:
        """Get available quests for a player."""
        # Build query to find quests not yet started by player
        query = text(
            """
            SELECT q.* FROM quests q
            LEFT JOIN quest_progress qp ON q.quest_id = qp.quest_id 
                AND qp.player_id = :player_id
            WHERE qp.quest_id IS NULL 
               OR qp.status IN ('available', 'failed', 'abandoned')
        """
        )

        result = await self.session.execute(query, {"player_id": player_id})
        rows = result.fetchall()

        quests = []
        for row in rows:
            quest = self._row_to_quest(row)
            if quest:
                quests.append(quest)

        return quests

    async def get_completed_quests(self, player_id: str) -> list[QuestProgress]:
        """Get all completed quests for a player."""
        query = text(
            """
            SELECT * FROM quest_progress 
            WHERE player_id = :player_id AND status = 'completed'
            ORDER BY updated_at DESC
        """
        )

        result = await self.session.execute(query, {"player_id": player_id})
        rows = result.fetchall()

        return [self._row_to_progress(row) for row in rows]

    async def get_active_quests(self, player_id: str) -> list[QuestProgress]:
        """Get all active quests for a player."""
        query = text(
            """
            SELECT * FROM quest_progress 
            WHERE player_id = :player_id AND status = 'active'
            ORDER BY started_at ASC
        """
        )

        result = await self.session.execute(query, {"player_id": player_id})
        rows = result.fetchall()

        return [self._row_to_progress(row) for row in rows]

    async def log_interaction(
        self,
        quest_id: str,
        player_id: str,
        interaction_type: str,
        interaction_data: dict[str, Any],
    ) -> bool:
        """Log a quest interaction."""
        try:
            query = text(
                """
                INSERT INTO quest_interactions (
                    quest_id, player_id, interaction_type, interaction_data
                ) VALUES (
                    :quest_id, :player_id, :interaction_type, :interaction_data
                )
            """
            )

            await self.session.execute(
                query,
                {
                    "quest_id": quest_id,
                    "player_id": player_id,
                    "interaction_type": interaction_type,
                    "interaction_data": json.dumps(interaction_data),
                },
            )

            await self.session.commit()
            return True

        except Exception:
            await self.session.rollback()
            return False

    def _row_to_quest(self, row) -> Quest | None:
        """Convert database row to Quest object."""
        try:
            # Parse steps from JSON
            steps_data = (
                json.loads(row.steps) if isinstance(row.steps, str) else row.steps
            )
            steps = []
            for step_data in steps_data:
                step = QuestStep(
                    step_id=step_data["step_id"],
                    description=step_data["description"],
                    requirements=step_data["requirements"],
                    completion_conditions=step_data["completion_conditions"],
                    rewards=step_data.get("rewards", {}),
                    optional=step_data.get("optional", False),
                )
                steps.append(step)

            # Parse other JSON fields
            prerequisites = (
                json.loads(row.prerequisites)
                if isinstance(row.prerequisites, str)
                else row.prerequisites
            )
            rewards = (
                json.loads(row.rewards) if isinstance(row.rewards, str) else row.rewards
            )

            return Quest(
                quest_id=row.quest_id,
                title=row.title,
                description=row.description,
                category=QuestCategory(row.category),
                difficulty=QuestDifficulty(row.difficulty),
                steps=steps,
                prerequisites=prerequisites or [],
                rewards=rewards or {},
                time_limit=row.time_limit,
                repeatable=row.repeatable or False,
            )

        except Exception:
            return None

    def _row_to_progress(self, row) -> QuestProgress:
        """Convert database row to QuestProgress object."""
        # Parse JSON fields
        completed_steps = (
            json.loads(row.completed_steps)
            if isinstance(row.completed_steps, str)
            else row.completed_steps
        )
        step_progress = (
            json.loads(row.step_progress)
            if isinstance(row.step_progress, str)
            else row.step_progress
        )

        # Convert timestamps to floats
        started_at = (
            row.started_at.timestamp()
            if hasattr(row.started_at, "timestamp")
            else row.started_at
        )
        updated_at = (
            row.updated_at.timestamp()
            if hasattr(row.updated_at, "timestamp")
            else row.updated_at
        )

        return QuestProgress(
            quest_id=row.quest_id,
            player_id=row.player_id,
            status=QuestStatus(row.status),
            current_step=row.current_step or 0,
            completed_steps=completed_steps or [],
            step_progress=step_progress or {},
            started_at=started_at,
            updated_at=updated_at,
        )
