import pickle
from typing import List

import pandas as pd
import streamlit as st


### intro ###
"""
# Grid Stability Prediction App

In [my Kaggle notebook](https://www.kaggle.com/sowlarn/predicting-smart-grid-stability/), I used UCI's simulated
[Electrical Grid Stability data](https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+)
to predict if a given combination of power system conditions would result in an unstable grid -
and therefore risk causing blackouts or damaging equipment.

This simple app can be used to see how adjusting model inputs affect the resulting predictions.

As shown in [my data exploration](https://www.kaggle.com/sowlarn/predicting-smart-grid-stability/#quick-eda), in the
simulated conditions, lowering the response delays and willingness to adapt generally stabilises the grid - but this
is not always the case. For example, increasing the producer willingness to adapt from the default settings of this app
will *increase* stability. In this way, such an app could be used to explore possible solutions while incorporating
financial and/or technical restrictions.
"""


### side bar ###
def add_input_set(
    feature: str,
    min_value: float,
    max_value: float,
    value: float,
    node_names: List[str],
):
    assert len(node_names) > 0, "At least one string required in `node_names` list."
    assert all(
        [name != "" for name in node_names]
    ), "Names in `node_names` list can't be empty strings."

    inputs = [
        st.slider(node_name, min_value, max_value, value) for node_name in node_names
    ]
    data = {feature + str(i + 1): inputs[i] for i in range(len(node_names))}
    final_df = pd.DataFrame(data, index=[0])
    return final_df


with st.sidebar:
    st.header("Grid conditions")

    with st.expander("Response delay", expanded=False):
        st.write(
            "How long it takes for each node to adapt their production or consumption in seconds:"
        )
        p_delay_df = add_input_set("p_delay", 0.5, 10.0, 0.5, ["Producer"])
        c_delay_df = add_input_set(
            "c_delay", 0.5, 10.0, 5.0, ["Consumer1", "Consumer2", "Consumer3"]
        )

    with st.expander("Willingness to adapt", expanded=True):
        st.write(
            "Willingness of each node to adapt their consumption or production per second:"
        )
        p_adapt_df = add_input_set("p_adapt", 0.05, 1.0, 0.05, ["Producer"])
        c_adapt_df = add_input_set(
            "c_adapt", 0.05, 1.0, 0.5, ["Consumer1", "Consumer2", "Consumer3"]
        )

input_df = p_delay_df.join(c_delay_df).join(p_adapt_df).join(c_adapt_df)


### results ###
st.subheader("Grid condition summary:")
st.write(input_df)

st.subheader("Predictions:")
clf = pickle.load(open("grid_clf.pkl", "rb"))
reg = pickle.load(open("grid_reg.pkl", "rb"))

# transformation already part of pipeline
clf_pred = clf.predict(input_df)[0]
reg_pred = reg.predict(input_df)[0]

stability_dict = {0: "unstable", 1: "stable"}
st.markdown(f"Best classifier's prediction: **{stability_dict.get(clf_pred)}**")
st.markdown(
    f"Best regressor's prediction: {round(reg_pred, 3)} (**{stability_dict.get(int(reg_pred < 0))}**)"
)


# seems to only work when written separately
st.write("  \n")
st.write("  \n")
st.write(
    """You can find the source code as well as the associated tests
and github actions [here](https://github.com/sowla/grid_stability_app)."""
)
