from src.inference.generate_map_output import generate_map_output

def test_map_output_contract():
    result = generate_map_output([[0.2]], 1, "2026-07-02")
    assert result["lead_day"] == 1
    assert result["overall_bust_probability"] == [[0.2]]
