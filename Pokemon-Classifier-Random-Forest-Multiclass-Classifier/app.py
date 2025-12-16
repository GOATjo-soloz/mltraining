import streamlit as st
import pandas as pd
import joblib
import os

# ✨ THE VERIFIED MAPPING ✨
# This is the correct Type 2 mapping (Superset) derived from LabelEncoder's
# alphabetical sorting on the 'Type 2_Fill' column ('No_Type_2' landed at index 12).
TYPE_MAPPING = {
    0: 'Bug', 1: 'Dark', 2: 'Dragon', 3: 'Electric', 4: 'Fairy',
    5: 'Fighting', 6: 'Fire', 7: 'Flying', 8: 'Ghost', 9: 'Grass',
    10: 'Ground', 11: 'Ice',
    12: None,  # The verified placement of the filler type
    13: 'Normal', 14: 'Poison', 15: 'Psychic', 16: 'Rock', 17: 'Steel', 18: 'Water'
}


@st.cache_resource
def load_model():
    """Loads the pre-trained Random Forest model."""
    try:
        CURR_DIR = os.path.dirname(__file__)
        MODEL_PATH = os.path.join(CURR_DIR, "pokemon_classifier.pkl")
        with open(MODEL_PATH, "rb") as file:
            model = joblib.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'pokemon_classifier.pkl' not found. Make sure the model file is in the same directory.")
        return None


# Load the model
Rf = load_model()

if Rf is not None:
    st.title('✨ Unfiltered Oracle: Pokémon Type Classifier')
    st.markdown("---")

    st.header("Stat Check: Feed the Machine")
    st.markdown("Adjust the sliders to input the Pokémon's base stats.")

    # --- Input Fields ---
    col1, col2, col3 = st.columns(3)

    # Max stats are usually around 255 for HP/Att/Def and 180 for Spe/SpA/SpD
    with col1:
        hp = st.slider('HP', 1, 255, 70)
        attack = st.slider('Attack', 1, 255, 75)
        defense = st.slider('Defense', 1, 255, 75)

    with col2:
        sp_atk = st.slider('Sp. Atk', 1, 255, 65)
        sp_def = st.slider('Sp. Def', 1, 255, 65)
        speed = st.slider('Speed', 1, 255, 60)

    with col3:
        generation = st.selectbox('Generation', [1, 2, 3, 4, 5, 6], index=0)
        legendary = st.checkbox('Legendary', value=False)

        # Prepare the input data for the model
        is_legendary = 1 if legendary else 0

    # --- Prediction Button ---
    st.markdown("---")
    if st.button('Classify & Spill the Tea'):

        # Calculate Total Stat
        total = hp + attack + defense + sp_atk + sp_def + speed

        # Create a DataFrame matching the training feature order
        input_data = pd.DataFrame([[
            total, hp, attack, defense, sp_atk, sp_def, speed, generation, is_legendary
        ]], columns=['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary'])

        # Make the prediction (Returns a numpy array with three columns)
        prediction = Rf.predict(input_data)[0]

        # Decode the results
        type_1_code = int(prediction[0])
        has_dual_type = bool(prediction[1])
        type_2_code = int(prediction[2])

        # Find the type names using the CORRECTED mapping
        type_1_name = TYPE_MAPPING.get(type_1_code, f'Unknown Code {type_1_code}')

        # --- Output Display ---
        st.subheader(f'📝 The Oracle Has Spoken')
        st.markdown(f"**Primary Vibe (Type 1):** **{type_1_name}**")

        if has_dual_type:
            type_2_name = TYPE_MAPPING.get(type_2_code, f'Unknown Code {type_2_code}')

            if type_2_name == 'No_Type_2':
                st.info(
                    f"**Dual Type Predicted:** The model is having an identity crisis. It thinks it has a dual type, but its Type 2 prediction is the fill value: **{type_2_name}**.")
            elif type_2_name == type_1_name:
                st.info(
                    "The model predicted a Type 1 and Type 2 that are identical. This is uncommon, but the model has spoken.")
            else:
                st.markdown(f"**Secondary Vibe (Type 2):** **{type_2_name}**")
        else:
            st.markdown(f"**Secondary Vibe (Type 2):** **None** (Monotype Realness)")

    st.markdown("---")
    st.caption(
        "Disclaimer: This model is a Random Forest classifier built on basic stats. It's not *actually* omniscient. Verify before you make it official.")