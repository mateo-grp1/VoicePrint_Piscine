# app.py — Démo v0 de Voiceprint (Streamlit)
# Enrôler une voix, vérifier une nouvelle voix, supprimer l'empreinte.
# À lancer depuis la racine du projet : streamlit run app.py
import io
import numpy as np
import librosa
import torch
import streamlit as st
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier

# ---------- Réglages de la démo v0 ----------
SEUIL = 0.409                                   # seuil EER choisi sur pairs_val (provisoire)
DUREE_MIN = 10                                  # secondes de parole minimum à l'enrôlement
EMPREINTE = Path("data/enrolement/empreinte.npy")   # data/ est exclu de GitHub

st.set_page_config(page_title="Voiceprint — démo v0", page_icon="🎙️")


# ---------- Outils ----------
@st.cache_resource                               # chargé une seule fois, pas à chaque clic
def charger_modele():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="models/ecapa")


def lire_audio(fichier):
    """Audio du micro -> signal 16 kHz mono, sans les silences du début et de la fin."""
    y, _ = librosa.load(io.BytesIO(fichier.getvalue()), sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y


def calculer_empreinte(y):
    """192 nombres ECAPA, ramenés à une longueur de 1 (seule la direction compte)."""
    with torch.no_grad():
        e = charger_modele().encode_batch(torch.tensor(y).unsqueeze(0)).squeeze().numpy()
    return e / np.linalg.norm(e)


# ---------- Page ----------
st.title("🎙️ Voiceprint — démo v0")
st.warning("Démonstration pédagogique, **pas un système d'authentification sécurisé**. "
           "L'empreinte vocale est une donnée biométrique : seule l'empreinte est stockée "
           "(jamais l'audio), sur cet ordinateur, et elle peut être supprimée à tout moment.")
st.caption(f"Modèle : ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb). Seuil = {SEUIL} : "
           "seuil de l'EER mesuré sur les paires de validation Common Voice (provisoire).")

# 1. Enrôlement
st.header("1. Enrôlement")
if EMPREINTE.exists():
    st.success("Une empreinte est enregistrée. Passe à l'étape 2, ou supprime-la (étape 3).")
else:
    accord = st.checkbox("J'accepte que mon empreinte vocale soit calculée et stockée "
                         "sur cet ordinateur jusqu'à sa suppression.")
    audio = st.audio_input(f"Parle librement pendant au moins {DUREE_MIN} s",
                           key="enrolement", disabled=not accord)
    if audio:
        y = lire_audio(audio)
        duree = len(y) / 16000
        st.write(f"Durée utile : **{duree:.1f} s**")
        if duree < DUREE_MIN:
            st.error(f"Trop court : il faut au moins {DUREE_MIN} s. Recommence.")
        elif st.button("Enregistrer mon empreinte"):
            EMPREINTE.parent.mkdir(parents=True, exist_ok=True)
            np.save(EMPREINTE, calculer_empreinte(y))
            st.rerun()                           # recharge la page pour passer à l'étape 2

# 2. Vérification
st.header("2. Vérification")
if not EMPREINTE.exists():
    st.info("Fais d'abord l'enrôlement.")
else:
    audio = st.audio_input("Quelqu'un parle quelques secondes", key="verification")
    if audio:
        y = lire_audio(audio)
        if len(y) / 16000 < 1:
            st.error("Trop court : parle au moins 1 seconde.")
        else:
            score = float(np.dot(np.load(EMPREINTE), calculer_empreinte(y)))   # cosinus
            st.metric("Score de similarité", f"{score:.3f}", f"seuil {SEUIL}", delta_color="off")
            if score >= SEUIL:
                st.success("✅ C'est la personne enrôlée.")
            else:
                st.error("❌ Ce n'est pas la personne enrôlée.")

# 3. Suppression
st.header("3. Supprimer l'empreinte")
if EMPREINTE.exists() and st.button("Supprimer mon empreinte"):
    EMPREINTE.unlink()
    st.success("Empreinte supprimée.")
    st.rerun()