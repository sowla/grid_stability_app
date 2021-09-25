import pytest
from grid_stability import add_input_set
import pandas as pd
import re


class TestAddInputSet(object):
    # note: basic unexpected type cases are already handled by Python and streamlit.
    
    expected_params = [("p_delay", 0.5, 10.0, 0.5, ["Producer"], 
                        pd.DataFrame({"p_delay1": 0.5}, index=[0])),
                      ("c_adapt", 0.05, 1.0, 0.5, ["Consumer1", "Consumer2", "Consumer3"], 
                        pd.DataFrame({"c_adapt" + str(i): 0.5  for i in range(1, 4)}, index=[0]))]
    @pytest.mark.parametrize("feature, min_value, max_value, value, node_names, expected", expected_params)
    def test_expected_dates(self, feature, min_value, max_value, value, node_names, expected):
        actual = add_input_set(feature, min_value, max_value, value, node_names)
        pd.testing.assert_frame_equal(actual, expected)


    expected_errors_params = [("p_delay", 0.5, 10.0, 0.5, [], "At least one string required in `node_names` list."),
                              ("p_delay", 0.5, 10.0, 0.5, [""], "Names in `node_names` list can't be empty strings.")]
    @pytest.mark.parametrize("feature, min_value, max_value, value, node_names, expected", expected_errors_params)        
    def test_expected_errors_raised(self, feature, min_value, max_value, value, node_names, expected):
        with pytest.raises(AssertionError) as error_msg:
            add_input_set(feature, min_value, max_value, value, node_names)
        assert error_msg.match(re.escape(expected))
