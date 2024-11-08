
import pytest

from core.cpt import CPT
import testdata.example_cpts as cpt


def test_cpt_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        CPT()


def test_cpt_type_error_2():  # bad pmfs type
    with pytest.raises(TypeError):
        CPT(23)
    with pytest.raises(TypeError):
        CPT(-9.2)
    with pytest.raises(TypeError):
        CPT(None)
    with pytest.raises(TypeError):
        CPT(True)


def test_cpt_type_error_3():  # bad estimated type
    with pytest.raises(TypeError):
        CPT(pmfs={'1': 0.0, '0': 1.0}, estimated=1.2)
    with pytest.raises(TypeError):
        CPT(pmfs={'1': 0.0, '0': 1.0}, estimated=True)
    with pytest.raises(TypeError):
        CPT(pmfs={'1': 0.0, '0': 1.0}, estimated=[1])


def test_cpt_value_error_1():  # bad simple pmf value to constructor
    with pytest.raises(ValueError):
        CPT({})
    with pytest.raises(ValueError):
        CPT({'A': 0.25})
    with pytest.raises(ValueError):
        CPT({'A': '0.5', 'B': '0.5'})
    with pytest.raises(ValueError):
        CPT({'A': -0.1, 'B': 0.7, 'C': 0.4})
    with pytest.raises(ValueError):
        CPT({'A': 0.2, 'B': 0.799998})
    with pytest.raises(ValueError):
        CPT({'A': 0.6, 'B': 0.40001})


def test_cpt_value_error_2():  # bad multi pmf value to constructor
    with pytest.raises(ValueError):
        CPT([])
    with pytest.raises(ValueError):
        CPT([(), ()])
    with pytest.raises(ValueError):
        CPT([({}, {}), ({}, {})])
    with pytest.raises(ValueError):
        CPT([({}, {'0': 0.2, '1': 0.8}),
             ({}, {'0': 0.7, '1': 0.3})])
    with pytest.raises(ValueError):
        CPT([({'A': '0'}, {'0': 2, '1': 0.8}),
             ({'A': '1'}, {'0': 0.7, '1': 0.3})])
    with pytest.raises(ValueError):
        CPT([({'A': '0'}, {'0': 0.2, '1': 0.8, '2': 0.3}),
             ({'A': '1'}, {'0': 0.7, '1': 0.3})])
    with pytest.raises(ValueError):
        CPT([({'A': '0'}, {'0': 0.2, '1': 0.8}),
             ({'A': 1}, {'0': 0.7, '1': 0.3})])


def test_cpt_value_error_3():  # more bad multi pmf value to constructor
    with pytest.raises(ValueError):
        CPT([({'A': '0'}, {'0': 0.2, '1': 0.8}),
             ({'B': '1'}, {'0': 0.7, '1': 0.3})])
    with pytest.raises(ValueError):
        CPT([({'A': '0'}, {'0': 0.2, '1': 0.8}),
             ({'A': '1'}, {'0': 0.7, '2': 0.3})])
    with pytest.raises(ValueError):
        CPT([({'A': '0'}, {'0': 0.2, '1': 0.8}),
             ({'A': '1'}, {'0': 0.7})])
    with pytest.raises(ValueError):
        CPT([({'A': '0', 'B': '1'}, {'0': 0.2, '1': 0.8}),
             ({'A': '1'}, {'0': 0.7, '1': 0.3})])


def test_cpt_value_error_4():  # bad estimated value
    with pytest.raises(ValueError):
        CPT({'A': 0.2, 'B': 0.8}, estimated=-1)
    with pytest.raises(ValueError):
        CPT({'A': 0.2, 'B': 0.8}, estimated=2)


def test_cpt_pmf_type_error1():  # parentless node but pmf called with arg
    cpt = CPT({'A': 0.2, 'B': 0.8})
    with pytest.raises(TypeError):
        cpt.cdist({'A': 1})


def test_cpt_pmf_type_error2():  # pmf called with non-dict for node w/ parents
    cpt = CPT([({'A': '0'}, {'0': 0.2, '1': 0.8}),
               ({'A': '1'}, {'0': 0.7, '1': 0.3})])
    with pytest.raises(TypeError):
        cpt.cdist()
    with pytest.raises(TypeError):
        cpt.cdist(['A', 1])
    with pytest.raises(TypeError):
        cpt.cdist(2.7)


def test_cpt_pmf_value_error():  # pmf called with unknown parental values
    cpt = CPT([({'A': '0'}, {'0': 0.2, '1': 0.8}),
               ({'A': '1'}, {'0': 0.7, '1': 0.3})])
    with pytest.raises(ValueError):
        cpt.cdist({'A': 2})
    with pytest.raises(ValueError):
        cpt.cdist({'B': 1})


def test_p0_v2_1_ok():  # good single pmf to constructor
    cpt.p0_v2_1(cpt.p0_v2_1())


def test_p0_v2_2_ok():  # good single pmf to constructor
    cpt.p0_v2_2(cpt.p0_v2_2())


def test_p0_v3_1_ok():  # good single pmf to constructor
    cpt.p0_v3_1(cpt.p0_v3_1())


def test_p1_v2_1_ok():  # good multi pmf to constructor
    cpt.p1_v2_1(cpt.p1_v2_1())


def test_cpt_ok():  # good multi pmf to constructor
    cpt.p2_v2_1(cpt.p2_v2_1())


def test_cpt_p0_v2_1_eq():  # compare identical CPTs
    assert cpt.p0_v2_1() == cpt.p0_v2_1()
    assert (cpt.p0_v2_1() != cpt.p0_v2_1()) is False


def test_cpt_p0_v2_1a_eq():  # compare identical CPTs
    assert cpt.p0_v2_1() == cpt.p0_v2_1a()
    assert (cpt.p0_v2_1() != cpt.p0_v2_1a()) is False


def test_cpt_p0_v3_1_eq():  # compare identical CPTs
    assert cpt.p0_v3_1() == cpt.p0_v3_1()
    assert (cpt.p0_v3_1() != cpt.p0_v3_1()) is False


def test_cpt_p1_v2_1_eq():  # compare identical CPTs
    assert cpt.p1_v2_1() == cpt.p1_v2_1()
    assert (cpt.p1_v2_1() != cpt.p1_v2_1()) is False


def test_cpt_p2_v2_1_eq():  # compare identical CPTs
    assert cpt.p2_v2_1() == cpt.p2_v2_1()
    assert (cpt.p2_v2_1() != cpt.p2_v2_1()) is False


def test_cpt_p2_v2_1a_eq():  # compare identical CPTs
    assert cpt.p2_v2_1() == cpt.p2_v2_1a()
    assert (cpt.p2_v2_1() != cpt.p2_v2_1a()) is False


def test_cpt_p0_v2_1b_eq():  # compare NEARLY identical CPTs
    assert cpt.p0_v2_1() == cpt.p0_v2_1b()
    assert cpt.p0_v2_1a() == cpt.p0_v2_1b()
    assert (cpt.p0_v2_1() != cpt.p0_v2_1b()) is False
    assert (cpt.p0_v2_1a() != cpt.p0_v2_1b()) is False


def test_cpt_p2_v2_1b_eq():  # compare NEARLY identical CPTs
    assert cpt.p2_v2_1() == cpt.p2_v2_1b()
    assert cpt.p2_v2_1a() == cpt.p2_v2_1b()
    assert (cpt.p2_v2_1() != cpt.p2_v2_1b()) is False
    assert (cpt.p2_v2_1a() != cpt.p2_v2_1b()) is False


def test_cpt_ne1():  # compare simple CPTs with different probs
    assert cpt.p0_v2_1() != cpt.p0_v2_2()
    assert (cpt.p0_v2_1() == cpt.p0_v2_2()) is False


def test_cpt_ne2():  # compare simple CPTs with different node values
    assert cpt.p0_v2_1() != cpt.p0_v2_3()
    assert (cpt.p0_v2_1() == cpt.p0_v2_3()) is False


def test_cpt_ne3():  # compare different CPTs
    assert cpt.p2_v2_1() != cpt.p2_v2_2()
    assert (cpt.p2_v2_1() == cpt.p2_v2_2()) is False


def test_cpt_ne4():  # compare different CPTs
    assert cpt.p2_v2_1a() != cpt.p2_v2_2()
    assert (cpt.p2_v2_1a() == cpt.p2_v2_2()) is False


# test to_spec function including name mapping

def test_to_spec_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        cpt.p0_v2_1().to_spec()
    with pytest.raises(TypeError):
        cpt.p2_v2_1().to_spec()


def test_to_spec_type_error_2():  # name_map not a dictionary
    with pytest.raises(TypeError):
        cpt.p0_v2_1().to_spec(False)
    with pytest.raises(TypeError):
        cpt.p2_v2_1().to_spec(None)
    with pytest.raises(TypeError):
        cpt.p0_v2_1().to_spec(1)
    with pytest.raises(TypeError):
        cpt.p2_v2_1().to_spec(23.2)
    with pytest.raises(TypeError):
        cpt.p2_v2_1().to_spec(['A'])


def test_to_spec_type_error_3():  # name_map keys not strings
    with pytest.raises(TypeError):
        cpt.p0_v2_1().to_spec({1: 'A', 'B': 'S'})
    with pytest.raises(TypeError):
        cpt.p2_v2_1().to_spec({1: 'A', 'B': 'S'})


def test_to_spec_type_error_4():  # name_map values not strings
    with pytest.raises(TypeError):
        cpt.p0_v2_1().to_spec({'A': 'A', 'B': 0.05})
    with pytest.raises(TypeError):
        cpt.p2_v2_1().to_spec({'A': 'A', 'B': 0.05})


def test_to_spec_value_error_1():  # name_map doesn't include all coeff keys
    with pytest.raises(ValueError):
        cpt.p2_v2_1().to_spec({'A': 'X', 'C': 'Y'})


def test_to_spec_1_ok():  # names remaining the same
    name_map = {'A': 'A', 'B': 'B'}
    spec = cpt.p2_v2_1().to_spec(name_map)
    print('\n\nSpec is:\n{}'.format('\n'.join([s.__str__() for s in spec])))

    cpt.p2_v2_1(CPT(spec))  # check spec against original definition


def test_to_spec_2_ok():  # mapping names has no effect on orphan CPT
    name_map = {'A': 'AA', 'B': 'BB'}
    spec = cpt.p0_v2_1().to_spec(name_map)
    print('\n\nSpec is {}'.format(spec))

    cpt.p0_v2_1(CPT(spec))  # check spec against original definition


def test_to_spec_3_ok():  # mapping names correctly, reordered correctly
    name_map = {'A': 'ZA', 'B': 'YB', 'C': 'ignored'}
    spec = cpt.p2_v2_1().to_spec(name_map)
    print('\n\nSpec is:\n{}'.format('\n'.join([s.__str__() for s in spec])))

    assert spec == \
        [({'YB': '0', 'ZA': '0'}, {'0': 0.2, '1': 0.8}),
         ({'YB': '1', 'ZA': '0'}, {'0': 0.7, '1': 0.3}),
         ({'YB': '0', 'ZA': '1'}, {'0': 0.5, '1': 0.5}),
         ({'YB': '1', 'ZA': '1'}, {'1': 0.9, '0': 0.1})]
