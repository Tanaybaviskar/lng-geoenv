from lng_geoenv.demand import DemandGenerator

def test_demand_variability():
    dg = DemandGenerator()

    values = [dg.step() for _ in range(50)]

    # demand should not be constant
    assert max(values) != min(values)


def test_demand_non_negative():
    dg = DemandGenerator()

    for _ in range(50):
        assert dg.step() >= 0