def choose_action(state, demand):
    storage_level = state.get("storage", {}).get("level", 0.0)
    storage_capacity = state.get("storage", {}).get("capacity", 200.0)
    price = state.get("price", 100.0)
    budget = state.get("budget", 500.0)
    ships = state.get("ships", [])
    blocked_routes = state.get("blocked_routes", [])

    valid_routes = ["Suez", "Panama", "Atlantic", "Hormuz"]
    available_routes = [r for r in valid_routes if r not in blocked_routes]

    # Shortage handling: prioritize reroutes for blocked ships
    if storage_level < demand:
        shortage_amount = demand - storage_level

        for ship in ships:
            ship_id = ship.get("id")
            current_route = ship.get("route")

            # Only reroute if on blocked route AND route is actually different
            if (
                ship_id is not None
                and current_route in blocked_routes
                and available_routes
            ):
                # Pick best available route (lowest risk preferred)
                new_route = min(
                    available_routes,
                    key=lambda r: ["Suez", "Panama", "Atlantic", "Hormuz"].index(r),
                )

                if new_route != current_route:
                    return {
                        "type": "reroute",
                        "parameters": {
                            "ship_id": ship_id,
                            "new_route": new_route,
                        },
                    }

        # Release storage if available
        if storage_level > 0:
            release_amount = min(shortage_amount * 0.8, storage_level)
            release_amount = max(0.0, release_amount)
            return {"type": "release", "parameters": {"amount": release_amount}}

    # Reroute ships on blocked routes (defensive)
    for ship in ships:
        ship_id = ship.get("id")
        current_route = ship.get("route")

        if (
            ship_id is not None
            and current_route in blocked_routes
            and ship.get("status") == "moving"
            and available_routes
        ):
            # Pick best available route
            new_route = min(
                available_routes,
                key=lambda r: ["Suez", "Panama", "Atlantic", "Hormuz"].index(r),
            )

            if new_route != current_route:
                return {
                    "type": "reroute",
                    "parameters": {"ship_id": ship_id, "new_route": new_route},
                }

    # Hedge when conditions are favorable (with budget enforcement)
    if price > 120 and budget >= 10:
        return {"type": "hedge", "parameters": {}}

    # Release excess storage
    storage_ratio = storage_level / max(storage_capacity, 1.0)
    if storage_ratio > 0.85:
        release_amount = max(0.0, (storage_level - 0.7 * storage_capacity) * 0.5)
        if release_amount > 0:
            return {"type": "release", "parameters": {"amount": release_amount}}

    return {"type": "wait", "parameters": {}}