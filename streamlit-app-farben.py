import streamlit as st
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import nnls
# prefer skimage's tested implementation; fallbacks below
from skimage.color import deltaE_ciede2000 as _skimage_deltaE


def cie_de2000(a, b):
    """
    Robust wrapper around skimage.deltaE_ciede2000.
    Accepts 1D iterables of length 3 and returns a non-negative float.
    """
    A = np.asarray(a, dtype=float).ravel()
    B = np.asarray(b, dtype=float).ravel()
    if A.size != 3 or B.size != 3:
        raise ValueError("cie_de2000 expects two 3-element LAB vectors")
    # skimage expects arrays of shape (..., 3). We pass single-pixel arrays and extract scalar.
    arr1 = A.reshape(1, 1, 3)
    arr2 = B.reshape(1, 1, 3)
    val = _skimage_deltaE(arr1, arr2)
    # val is an array with one element
    return float(val.ravel()[0])

def mix_lab(weights, labs):
    w = np.asarray(weights, dtype=float).ravel()
    L = np.asarray(labs, dtype=float)
    if L.ndim != 2 or L.shape[1] != 3:
        raise ValueError("labs must be (n,3)")
    if w.size != L.shape[0]:
        raise ValueError("weights length must equal labs rows")
    s = float(w.sum())
    if s <= 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return (L.T.dot(w) / s).ravel()

def allocate_units(frac_units, total_units):
    base = np.floor(frac_units).astype(int)
    rem = total_units - base.sum()
    if rem > 0:
        idx = np.argsort(-(frac_units - base))
        for i in idx[:rem]:
            base[i] += 1
    elif rem < 0:
        idx = np.argsort(frac_units - base)
        i = 0
        while rem < 0 and i < len(idx):
            if base[idx[i]] > 0:
                base[idx[i]] -= 1
                rem += 1
            i += 1
    return base

def find_best_recipe(pigment_labs, target_lab, total_grams=500.0, step=0.01, max_components=4, debug=False):
    """
    pigment_labs: DataFrame indexed by name with columns ['L','a','b'] OR dict name->[L,a,b]
    target_lab: sequence-like [L,a,b]
    total_grams: desired total mass in g
    step: gram resolution (e.g. 0.01)
    max_components: max number of pigments per recipe
    """
    # normalize pigments input
    if isinstance(pigment_labs, pd.DataFrame):
        names = list(pigment_labs.index)
        labs = pigment_labs[['L','a','b']].values.astype(float)
    else:
        if not isinstance(pigment_labs, dict):
            raise ValueError("pigment_labs must be a DataFrame or dict")
        names = list(pigment_labs.keys())
        labs = np.array([pigment_labs[n] for n in names], dtype=float)

    target = np.asarray(target_lab, dtype=float).ravel()
    if target.size != 3:
        raise ValueError("target_lab must be length 3 (L,a,b)")

    n = len(names)
    total_units = int(round(total_grams / step))
    if total_units <= 0:
        raise ValueError("total_grams/step yields non-positive units")

    best = {'deltaE': float('inf'), 'recipe': None, 'mixed_lab': None, 'combo': None}

    for r in range(1, min(max_components, n) + 1):
        for combo in itertools.combinations(range(n), r):
            sub = labs[list(combo), :]         # (r,3)
            A = sub.T                         # (3, r)
            # solve non-negative least squares A @ w = target
            try:
                w_cont, _ = nnls(A, target)
            except Exception:
                # fallback least squares + clamp
                w_ls, *_ = np.linalg.lstsq(A, target, rcond=None)
                w_cont = np.clip(w_ls, 0.0, None)

            if np.allclose(w_cont, 0.0):
                continue
            s = float(w_cont.sum())
            if s <= 1e-12:
                continue
            frac = w_cont / s
            units_cont = frac * total_units
            units = allocate_units(units_cont, total_units)
            grams = units.astype(int) * step
            if grams.sum() <= 0:
                continue

            mixed = mix_lab(grams, sub)
            # compute DeltaE using skimage wrapper
            try:
                dE = cie_de2000(mixed, target)
            except Exception as e:
                if debug:
                    print("DeltaE error for combo", [names[i] for i in combo], "err:", e)
                continue

            if not np.isfinite(dE):
                continue

            if dE < best['deltaE']:
                recipe = { names[combo[i]]: np.round(float(grams[i]), 4) for i in range(len(combo)) }
                best = {'deltaE': float(dE), 'recipe': recipe, 'mixed_lab': mixed.tolist(), 'combo': [names[i] for i in combo]}
                if debug:
                    # print("NEW BEST ΔE={:.4f} combo={} recipe={}".format(dE, best['combo'], best['recipe']))
                    # st.write("NEW BEST ΔE={:.4f} combo={} recipe={}, mixed_lab={}".format(dE, best['combo'], best['recipe'], np.round(best['mixed_lab'], 4)))
                    st.write(f"NEW BEST ΔE={dE:.4f} combo={best['combo']} recipe={best['recipe']}, "
                             f"mixed_lab={np.round(best['mixed_lab'], 4)}")

    # fallback: single nearest pigment if nothing found
    if best['recipe'] is None:
        dists = np.linalg.norm(labs - target.reshape(1,3), axis=1)
        idx = int(np.argmin(dists))
        recipe = { names[idx]: float(total_grams) }
        mixed = labs[idx].tolist()
        dE = cie_de2000(mixed, target)
        best = {'deltaE': float(dE), 'recipe': recipe, 'mixed_lab': mixed, 'combo': [names[idx]]}

    return best


# # Benutzerliste mit Passwörtern
# users = {
#     "Philipp": "Philipp98"
# }
#
# # Login-Formular
# st.title("🔐 Login")
# username = st.text_input("Benutzername")
# password = st.text_input("Passwort", type="password")
#
# if username in users and users[username] == password:
#     st.success(f"Willkommen, {username}!")
#     # Hier kommt deine App-Logik
# else:
#     st.warning("Zugriff verweigert. Bitte gültige Zugangsdaten eingeben.")
#     st.stop()

# Streamlit UI
st.title("🎨 Farben-Mix Optimierer")

# Sample pigment data
sample = {
    'P01':[50.0,20.0,30.0],'P02':[60.0,-10.0,15.0],'P03':[30.0,5.0,-20.0],'P04':[70.0,10.0,5.0],
    'P05':[40.0,-25.0,10.0],'P06':[55.0,15.0,-5.0],'P07':[65.0,-5.0,-15.0],'P08':[20.0,30.0,10.0],
    'P09':[45.0,0.0,0.0],'P10':[35.0,10.0,10.0],'P11':[80.0,-20.0,20.0],'P12':[25.0,18.0,-8.0],
    'P13':[52.0,-8.0,12.0],'P14':[48.0,22.0,-6.0],'P15':[58.0,6.0,2.0],'P16':[42.0,-12.0,25.0],
    'P17':[36.0,14.0,-18.0],'P18':[66.0,-2.0,8.0]
}
pigments_df = pd.DataFrame.from_dict(sample, orient='index', columns=['L','a','b'])

st.sidebar.header("Input Parameter")
L = st.sidebar.number_input("Zielwert L", value=50.0)
a = st.sidebar.number_input("Zielwert a", value=5.0)
b = st.sidebar.number_input("Zielwert b", value=8.0)
total_grams = st.sidebar.number_input("Gewünschte Menge [gramm]", value=500.0, min_value=0.01)
step = st.sidebar.number_input("Genauigkeit [gramm]", value=0.01, min_value=0.001)
max_components = st.sidebar.slider("Maximale Anzahl Farben", min_value=1, max_value=6, value=4)
debug = st.sidebar.checkbox("Debug Modus", value=False)

if st.button("Finde das beste Rezept"):
    target_lab = [L, a, b]
    res = find_best_recipe(
        pigment_labs=pigments_df,
        target_lab=target_lab,
        total_grams=total_grams,
        step=step,
        max_components=max_components,
        debug=debug
    )

    st.subheader("🔍 Ergebnis")
    st.write(f"**DeltaE Wert:** {res['deltaE']:.4f}")
    st.write(f"**Berechnete LAB Werte:** {np.round(res['mixed_lab'], 4)}")

    recipe_df = pd.DataFrame(list(res['recipe'].items()), columns=["Pigment", "Gram"])
    st.table(recipe_df)
