"""
Container usage handler implementation.
"""

from game_loop.state.models import ActionResult, InventoryItem, Location, PlayerState

from .base import UsageHandler


class ContainerUsageHandler(UsageHandler):
    """
    Handler for "put X in Y" usage scenarios.
    Handles putting an item into a container.
    """

    async def validate(
        self,
        item_to_use: InventoryItem,
        player_state: PlayerState,
        current_location: Location,
    ) -> bool:
        """Validate if the container usage is possible."""
        # Validation would be moved here from the original _handle_put_in_container
        # For now, just return True as validation is done in handle
        return True

    async def handle(
        self,
        item_to_use: InventoryItem,
        target_name: str | None,
        player_state: PlayerState,
        current_location: Location,
    ) -> ActionResult:
        """Handle putting an item into a container."""
        if not target_name:
            return ActionResult(
                success=False, feedback_message="Not sure where you want to put that."
            )

        # Extract container name from target
        parts = []
        if " in " in target_name:
            parts = target_name.split(" in ", 1)
        elif " into " in target_name:
            parts = target_name.split(" into ", 1)

        if len(parts) != 2:
            return ActionResult(
                success=False, feedback_message="Not sure where you want to put that."
            )

        container_name = parts[1].strip()
        normalized_container_name = container_name.lower()

        # Find the container (can be in inventory or location)
        inventory_container = None
        world_container = None
        container_location_type = None
        container_id = None

        # Check player inventory first
        for inv_item in player_state.inventory:
            if (
                inv_item.name.lower() == normalized_container_name
                and inv_item.attributes.get("is_container")
            ):
                inventory_container = inv_item
                container_location_type = "inventory"
                container_id = inv_item.item_id
                break

        # Check location objects if not in inventory
        if inventory_container is None:
            for loc_obj_id, loc_obj in current_location.objects.items():
                if (
                    loc_obj.name.lower() == normalized_container_name
                    and loc_obj.is_container
                ):
                    world_container = loc_obj
                    container_location_type = "location"
                    container_id = loc_obj_id
                    break

        if not container_id:
            return ActionResult(
                success=False,
                feedback_message="You don't see any container named "
                f"'{container_name}' here.",
            )

        # Define changes for putting item in container
        inventory_remove = {"action": "remove", "item_id": item_to_use.item_id}

        if container_location_type == "inventory" and inventory_container:
            # Check if the inventory item is a container
            is_obj_container = inventory_container.attributes.get("is_container", False)
            if not is_obj_container:
                return ActionResult(
                    success=False,
                    feedback_message="You can't put things inside the "
                    f"{container_name}.",
                )

            container_update = {
                "action": "update",
                "item_id": container_id,
                "updates": {
                    "attributes": {
                        **inventory_container.attributes,
                        "contained_items": (
                            inventory_container.attributes.get("contained_items", [])
                            + [item_to_use.item_id]
                        ),
                    }
                },
            }
            return ActionResult(
                success=True,
                feedback_message=f"You put the {item_to_use.name} in the "
                f"{inventory_container.name}.",
                inventory_changes=[inventory_remove, container_update],
            )
        elif container_location_type == "location" and world_container:
            # Check if the world object is a container
            if not world_container.is_container:
                return ActionResult(
                    success=False,
                    feedback_message=f"You can't put things inside the "
                    f"{container_name}.",
                )

            # Container is in location
            object_update = {
                "action": "update",
                "location_id": current_location.location_id,
                "object_id": container_id,
                "update": {
                    "contained_items": world_container.contained_items
                    + [item_to_use.item_id]
                },
            }
            return ActionResult(
                success=True,
                feedback_message=f"You put the {item_to_use.name} in the "
                f"{world_container.name}.",
                inventory_changes=[inventory_remove],
                object_changes=[object_update],
            )

        # Should never reach here if container_id is set
        return ActionResult(
            success=False,
            feedback_message="Something went wrong while trying to use that container.",
        )
