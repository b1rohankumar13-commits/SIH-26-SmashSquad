"""Write map-ready overall bust probabilities and diagnostic category tags."""

def generate_map_output(probability_grid, lead_day, valid_time):
    return {"lead_day": lead_day, "valid_time": str(valid_time),
            "overall_bust_probability": probability_grid}
