def update_ships(ships, blocked_routes):
    for ship in ships:
        if ship["status"] == "moving":

            # if route blocked then delay
            if ship["route"] in blocked_routes:
                ship["eta"] += 1   # delay
            else:
                ship["eta"] -= 1

            if ship["eta"] <= 0:
                ship["status"] = "arrived"

    return ships


def handle_arrivals(ships, storage):
    for ship in ships:
        if ship["status"] == "arrived":
            storage["level"] = min(
                storage["capacity"],
                storage["level"] + ship["capacity"]
            )
            ship["status"] = "done"

    return ships, storage