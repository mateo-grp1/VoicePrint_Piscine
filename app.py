# app.py — Démo de Voiceprint (Streamlit), interface épurée en 3 étapes
#   1. Enregistrer une voix (3, 5 ou 10 s, avec consentement)
#   2. Tester des voix, autant de fois qu'on veut (décision, confiance, "Pourquoi ?")
#   3. Supprimer la voix enregistrée
# Configuration figée : ECAPA + moyenne soustraite, seuils à 1 % d'imposteurs (0,428 / 0,447 / 0,482), confiance calibrée.
# La voix est préparée par src/preparation.py : la MÊME recette que les bancs d'essai.
# Prérequis : models/reference_demo.npz (src/eval_v3.py puis src/figer_seuils.py)
# À lancer depuis la racine du projet : streamlit run app.py
import io
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, "src")
from ecapa_eval import charger_modele                      # noqa: E402
from normalisation import preparer, seuil_validation        # noqa: E402
from caracteristiques import extraire, expliquer            # noqa: E402
from preparation import (SR, charger_brut, empreinte_brute,  # noqa: E402
                         enrolement, garder_parole, test as decouper_test)

DUREES = [3, 5, 10]
EMPREINTE = Path("data/enrolement/empreinte.npz")    # data/ est exclu de GitHub
REFERENCE = Path("models/reference_demo.npz")

st.set_page_config(page_title="Voiceprint", page_icon="🎙️", layout="centered",
                   initial_sidebar_state="collapsed")

# ---------- Apparence : épurée, aérée ----------
st.markdown("""
<style>
  .block-container {max-width: 680px; padding-top: 3.5rem; padding-bottom: 5rem;}
  #MainMenu, footer {visibility: hidden;}
  .titre {font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0;}
  .sous-titre {color: #6b7280; font-size: 1.05rem; margin-top: 0.2rem;}
  .avertissement {color: #9ca3af; font-size: 0.8rem; margin-top: 0.8rem;}
  .etape {display: flex; align-items: center; gap: 0.8rem; margin: 0 0 1.2rem 0;}
  .pastille {width: 2rem; height: 2rem; border-radius: 50%; background: #1f3a5f; color: white;
             display: flex; align-items: center; justify-content: center; font-weight: 700;}
  .pastille.inactive {background: #d1d5db;}
  .etape-titre {font-size: 1.35rem; font-weight: 650;}
  .decision {text-align: center; padding: 1.6rem 1rem; border-radius: 18px; margin: 1.2rem 0 0.6rem 0;}
  .decision .verdict {font-size: 1.9rem; font-weight: 700;}
  .decision .confiance {font-size: 1.05rem; margin-top: 0.3rem; opacity: 0.85;}
  .oui {background: #ecf7f0; color: #1b6b3a;}
  .non {background: #fbeeee; color: #9b2c2c;}
  .detail {text-align: center; color: #9ca3af; font-size: 0.8rem;}
  .etat {padding: 0.9rem 1.1rem; border-radius: 14px; background: #f3f6fa; color: #1f3a5f;}
  .vide {color: #9ca3af;}
  button[kind="primary"] {background: #1f3a5f !important; border-color: #1f3a5f !important; color: white !important;}
  button[kind="primary"]:hover {background: #2c5282 !important; border-color: #2c5282 !important;}
  button[data-variant="segmented_control"][aria-checked="true"] {background: #1f3a5f !important; border-color: #1f3a5f !important; color: white !important;}
  button[data-variant="segmented_control"][aria-checked="true"] * {color: white !important;}
</style>
""", unsafe_allow_html=True)


def espace(hauteur=2.5):
    st.markdown(f"<div style='height:{hauteur}rem'></div>", unsafe_allow_html=True)


def entete_etape(numero, titre, active=True):
    classe = "pastille" if active else "pastille inactive"
    couleur = "" if active else "color:#9ca3af"
    st.markdown(f"<div class='etape'><div class='{classe}'>{numero}</div>"
                f"<div class='etape-titre' style='{couleur}'>{titre}</div></div>",
                unsafe_allow_html=True)


# ---------- Chargements (une seule fois) ----------
@st.cache_resource
def modele():
    return charger_modele()


def lire_npz(chemin):
    """Lit un fichier .npz, copie son contenu puis le FERME (sinon Windows interdit de le supprimer)."""
    with np.load(chemin) as f:
        return {k: f[k].copy() for k in f.files}


@st.cache_resource
def reference():
    r = lire_npz(REFERENCE)
    if "seuils" in r:
        seuils = {int(d): float(s) for d, s in json.loads(str(r["seuils"])).items()}
    else:
        seuils = {d: seuil_validation(d) for d in DUREES}
    return {"moyenne": r["moyenne"], "a": float(r["calib_a"]), "b": float(r["calib_b"]),
            "ecart_intra": json.loads(str(r["ecart_intra"])), "seuils": seuils}


# ---------- Outils ----------
def lire_audio(fichier):
    """Audio du micro -> 16 kHz mono, puis on ne garde QUE la parole (silences et pauses retirés)."""
    brut = charger_brut(io.BytesIO(fichier.getvalue()))
    return garder_parole(brut)


def empreinte(y):
    return preparer(empreinte_brute(modele(), y), reference()["moyenne"])[0]


def confiance(score):
    ref = reference()
    return 1 / (1 + np.exp(-(ref["a"] * score + ref["b"])))


def supprimer():
    EMPREINTE.unlink(missing_ok=True)
    st.session_state["passage"], st.session_state["historique"] = 1, []


# ---------- En-tête ----------
st.markdown("<div class='titre'>🎙️ Voiceprint</div>"
            "<div class='sous-titre'>Ta voix, ton identité</div>"
            "<div class='avertissement'>Démonstration pédagogique, pas un système de sécurité. "
            "Seule l'empreinte de la voix est gardée sur cet ordinateur, jamais l'enregistrement.</div>",
            unsafe_allow_html=True)

if not REFERENCE.exists():
    st.error("Réglages introuvables (models/reference_demo.npz) : lance `python src/eval_v3.py` "
             "puis `python src/figer_seuils.py`.")
    st.stop()

st.session_state.setdefault("passage", 1)
st.session_state.setdefault("historique", [])
enrole = EMPREINTE.exists()
espace(3)

# ---------- 1. Enregistrer une voix ----------
entete_etape(1, "Enregistrer une voix")
if enrole:
    d_enrol = int(lire_npz(EMPREINTE)["duree"])
    st.markdown(f"<div class='etat'>✅ Voix enregistrée ({d_enrol} s de parole)</div>", unsafe_allow_html=True)
else:
    duree = st.segmented_control("Durée de parole", DUREES, default=3, format_func=lambda d: f"{d} s",
                                 selection_mode="single") or 3
    accord = st.checkbox("J'accepte que l'empreinte de ma voix soit enregistrée sur cet ordinateur "
                         "(elle peut être supprimée à tout moment).")
    audio = st.audio_input(f"Parlez librement pendant au moins {duree} secondes",
                           key=f"enrol_{duree}", disabled=not accord)
    if audio:
        y = lire_audio(audio)
        parole = len(y) / SR
        if parole < duree:
            st.warning(f"Seulement {parole:.1f} s de voix détectée, il en faut {duree}. Recommencez.")
        elif st.button("Enregistrer ma voix", type="primary", use_container_width=True):
            y = enrolement(y, duree)
            EMPREINTE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(EMPREINTE, emb=empreinte(y), duree=duree, caracteristiques=json.dumps(extraire(y)))
            st.rerun()

espace()

# ---------- 2. Tester une voix ----------
entete_etape(2, "Tester une voix", active=enrole)
if not enrole:
    st.markdown("<div class='vide'>Enregistrez d'abord une voix à l'étape 1.</div>", unsafe_allow_html=True)
else:
    enr = lire_npz(EMPREINTE)
    seuil = reference()["seuils"][int(enr["duree"])]
    n = st.session_state["passage"]
    audio = st.audio_input(f"Personne n° {n} : parlez quelques secondes", key=f"test_{n}")
    if audio:
        y = lire_audio(audio)
        parole = len(y) / SR
        if parole < 1.5:
            st.warning("Trop court : parlez au moins 3 secondes.")
        else:
            y = decouper_test(y)
            score = float(enr["emb"] @ empreinte(y))
            accepte = score >= seuil
            if accepte:
                st.markdown(f"<div class='decision oui'><div class='verdict'>✅ Même personne</div>"
                            f"<div class='confiance'>Confiance {confiance(score):.0%}</div></div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='decision non'><div class='verdict'>❌ Personne différente</div></div>",
                            unsafe_allow_html=True)
            st.markdown(f"<div class='detail'>score {score:.3f} · seuil {seuil:.3f}</div>", unsafe_allow_html=True)

            with st.expander("Pourquoi ?"):
                a = json.loads(str(enr["caracteristiques"]))
                for ligne in expliquer(a, extraire(y), reference()["ecart_intra"]):
                    st.write("• " + ligne)
                st.caption("La décision repose sur l'empreinte complète de la voix ; "
                           "ces mesures servent seulement à l'expliquer.")

            espace(0.8)
            if st.button("Personne suivante →", type="primary", use_container_width=True):
                st.session_state["historique"].append({
                    "Personne": n, "Résultat": "✅ même personne" if accepte else "❌ différente",
                    "Confiance": f"{confiance(score):.0%}" if accepte else "—"})
                st.session_state["passage"] = n + 1
                st.rerun()

    if st.session_state["historique"]:
        espace(1)
        with st.expander(f"Historique ({len(st.session_state['historique'])} personnes)"):
            st.dataframe(st.session_state["historique"], hide_index=True, use_container_width=True)

espace()

# ---------- 3. Supprimer la voix ----------
entete_etape(3, "Supprimer la voix", active=enrole)
if enrole:
    if st.button("Supprimer l'empreinte enregistrée", use_container_width=True):
        supprimer()
        st.toast("Empreinte supprimée.")
        st.rerun()
else:
    st.markdown("<div class='vide'>Aucune voix enregistrée.</div>", unsafe_allow_html=True)

espace(4)
with st.expander("Détails techniques"):
    seuils = reference()["seuils"]
    st.caption("Modèle : ECAPA-TDNN pré-entraîné (speechbrain/spkrec-ecapa-voxceleb), empreinte de 192 nombres, "
               "moyenne soustraite, comparaison par cosinus.")
    st.caption("Seuils (au plus 1 % d'imposteurs acceptés, voix proches) : "
               + " · ".join(f"{d} s : {s:.3f}" for d, s in seuils.items()))
    st.caption("Confiance calibrée sur la validation, en supposant une chance sur deux au départ.")