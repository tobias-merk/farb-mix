import streamlit as st
import pandas as pd
from mix_lab import *

# Sample pigment data
sample = {
    'P01':[50.0,20.0,30.0],'P02':[60.0,-10.0,15.0],'P03':[30.0,5.0,-20.0],'P04':[70.0,10.0,5.0],
    'P05':[40.0,-25.0,10.0],'P06':[55.0,15.0,-5.0],'P07':[65.0,-5.0,-15.0],'P08':[20.0,30.0,10.0],
    'P09':[45.0,0.0,0.0],'P10':[35.0,10.0,10.0],'P11':[80.0,-20.0,20.0],'P12':[25.0,18.0,-8.0],
    'P13':[52.0,-8.0,12.0],'P14':[48.0,22.0,-6.0],'P15':[58.0,6.0,2.0],'P16':[42.0,-12.0,25.0],
    'P17':[36.0,14.0,-18.0],'P18':[66.0,-2.0,8.0]
}
pigments_df = pd.DataFrame.from_dict(sample, orient='index', columns=['L','a','b'])

# Streamlit UI
st.title("🎨 Mix Lab Optimizer")

st.sidebar.header("Input Parameters")
L = st.sidebar.number_input("Target L", value=50.0)
a = st.sidebar.number_input("Target a", value=5.0)
b = st.sidebar.number_input("Target b", value=8.0)
total_grams = st.sidebar.number_input("Total grams", value=500.0, min_value=0.01)
step = st.sidebar.number_input("Step size (g)", value=0.01, min_value=0.001)
max_components = st.sidebar.slider("Max pigments per recipe", min_value=1, max_value=6, value=4)
debug = st.sidebar.checkbox("Enable debug output", value=False)

if st.button("Find Best Recipe"):
    target_lab = [L, a, b]
    res = find_best_recipe(
        pigment_labs=pigments_df,
        target_lab=target_lab,
        total_grams=total_grams,
        step=step,
        max_components=max_components,
        debug=debug
    )

    st.subheader("🔍 Result")
    st.write(f"**DeltaE00:** {res['deltaE']:.4f}")
    st.write(f"**Mixed LAB:** {np.round(res['mixed_lab'], 4)}")
    # st.write("**Pigment Combination:**", res['combo'])

    recipe_df = pd.DataFrame(list(res['recipe'].items()), columns=["Pigment", "Grams"])
    st.table(recipe_df)
