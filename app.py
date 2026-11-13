# app.py — Démo v1 de Voiceprint (Streamlit)
# Enrôler une voix (3, 5 ou 10 s), vérifier une nouvelle voix, expliquer, supprimer l'empreinte.
# Configuration retenue sur la validation : ECAPA + moyenne soustraite, enrôlement en un bloc,
# seuil à 1 % d'imposteurs acceptés, confiance calibrée.
# Prérequis : avoir lancé src/calibration.py (crée models/reference_demo.npz).
# À lancer depuis la racine du projet : streamlit run app.py
import io
import json
import sys
from pathlib import Path
import numpy as np
import librosa
import torch
import streamlit as st

sys.path.insert(0, "src")
from ecapa_eval import charger_modele                      # noqa: E402
from normalisation import preparer, seuil_validation        # noqa: E402
from caracteristiques import extraire, expliquer            # noqa: E402

SR = 16000
DUREES = [3, 5, 10]                                  # durées d'enrôlement proposées (secondes)
TEST_MAX = 10                                        # parole analysée au maximum pour un test (secondes)
AFFICHER_CONSEILS = False                            # conseils de qualité (volume, bruit) : désactivés pour l'instant
EMPREINTE = Path("data/enrolement/empreinte.npz")    # data/ est exclu de GitHub
REFERENCE = Path("models/reference_demo.npz")

st.set_page_config(page_title="Voiceprint — démo v1", page_icon="🎙️")


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
    return {"moyenne": r["moyenne"], "a": float(r["calib_a"]), "b": float(r["calib_b"]),
            "ecart_intra": json.loads(str(r["ecart_intra"])),
            "seuils": {d: seuil_validation(d) for d in DUREES}}


# ---------- Outils audio ----------
def lire_audio(fichier):
    """Audio du micro -> 16 kHz mono, puis on ne garde QUE la parole :
    les silences du début, de la fin ET les pauses au milieu sont retirés, puis les morceaux de parole sont recollés.
    Ainsi, "3 s d'enrôlement" = 3 s de voix réelle, pas 3 s dont une partie de silence."""
    brut, _ = librosa.load(io.BytesIO(fichier.getvalue()), sr=SR, mono=True)
    morceaux = librosa.effects.split(brut, top_db=30)
    parole = np.concatenate([brut[a:b] for a, b in morceaux]) if len(morceaux) else brut[:0]
    return brut, parole


def qualite(brut):
    """Conseils sur l'enregistrement : volume et bruit de fond (rapport signal/bruit estimé)."""
    conseils = []
    pic = float(np.abs(brut).max()) if len(brut) else 0.0
    rms = librosa.feature.rms(y=brut, frame_length=1024, hop_length=512)[0] + 1e-9
    snr = 20 * np.log10(np.percentile(rms, 90) / np.percentile(rms, 10))
    if pic > 0.99:
        conseils.append(("warning", "Le son sature : éloigne-toi un peu du micro ou parle moins fort."))
    elif pic < 0.05:
        conseils.append(("warning", "Le son est très faible : rapproche-toi du micro."))
    if snr < 10:
        conseils.append(("error", f"Beaucoup de bruit autour (rapport voix/bruit ≈ {snr:.0f} dB). "
                                  "Rapproche-toi du micro ou trouve un endroit plus calme : "
                                  "le système risque de ne pas reconnaître la personne."))
    elif snr < 20:
        conseils.append(("warning", f"Un peu de bruit autour (rapport voix/bruit ≈ {snr:.0f} dB). "
                                    "Parler plus près du micro améliore la reconnaissance."))
    return snr, conseils


def afficher_qualite(brut):
    if not AFFICHER_CONSEILS:      # le micro intégré du PC déclenchait des alertes de saturation injustifiées
        return
    snr, conseils = qualite(brut)
    for niveau, texte in conseils:
        (st.error if niveau == "error" else st.warning)(texte)
    if not conseils:
        texte = "> 40 dB" if snr > 40 else f"≈ {snr:.0f} dB"
        st.caption(f"Qualité d'enregistrement correcte (rapport voix/bruit {texte}).")


def empreinte(y):
    """Empreinte ECAPA avec la moyenne du train soustraite (réglage retenu au jour 3)."""
    with torch.no_grad():
        e = modele().encode_batch(torch.tensor(y).unsqueeze(0)).squeeze().numpy()
    return preparer(e, reference()["moyenne"])[0]


def confiance(score):
    ref = reference()
    return 1 / (1 + np.exp(-(ref["a"] * score + ref["b"])))


# ---------- Page ----------
st.title("🎙️ Voiceprint — démo v1")
st.warning("Démonstration pédagogique, **pas un système d'authentification sécurisé** "
           "(il ne détecte pas un enregistrement rejoué ni une voix clonée). "
           "L'empreinte vocale est une donnée biométrique : seule l'empreinte est stockée "
           "(jamais l'audio), sur cet ordinateur, et elle peut être supprimée à tout moment.")

if not REFERENCE.exists():
    st.error("Fichier models/reference_demo.npz introuvable : lance d'abord `python src/calibration.py`.")
    st.stop()

with st.sidebar:
    st.header("Réglages documentés")
    st.markdown("**Modèle** : ECAPA-TDNN pré-entraîné (speechbrain/spkrec-ecapa-voxceleb)\n\n"
                "**Normalisation** : soustraction de l'empreinte moyenne (locuteurs d'entraînement)\n\n"
                "**Seuils** (validation Common Voice, voix proches, 1 % d'imposteurs acceptés) :")
    for d, s in reference()["seuils"].items():
        st.markdown(f"- enrôlement {d} s : **{s:.3f}**")
    st.caption("La confiance (%) est calibrée sur la validation, en supposant une chance sur deux au départ.")

# 1. Enrôlement
st.header("1. Enrôlement")
if EMPREINTE.exists():
    d_enrol = int(lire_npz(EMPREINTE)["duree"])
    st.success(f"Une empreinte est enregistrée (enrôlement de {d_enrol} s). "
               "Passe à l'étape 2, ou supprime-la (étape 3) pour changer de personne ou de durée.")
else:
    duree = st.radio("Durée d'enrôlement", DUREES, horizontal=True, format_func=lambda d: f"{d} s",
                     help="3 s est la durée visée pour la démo ; 5 et 10 s permettent de comparer.")
    accord = st.checkbox("J'accepte que mon empreinte vocale soit calculée et stockée "
                         "sur cet ordinateur jusqu'à sa suppression.")
    audio = st.audio_input(f"Parle librement pendant au moins {duree} s", key=f"enrol_{duree}",
                           disabled=not accord)
    if audio:
        brut, y = lire_audio(audio)
        afficher_qualite(brut)
        parole = len(y) / SR
        if parole < duree:
            st.error(f"Seulement {parole:.1f} s de parole détectée : il en faut au moins {duree}. Recommence.")
        else:
            st.write(f"Parole détectée : {parole:.1f} s → on garde les **{duree} premières secondes**.")
            if st.button("Enregistrer mon empreinte"):
                y = y[:duree * SR]
                EMPREINTE.parent.mkdir(parents=True, exist_ok=True)
                np.savez(EMPREINTE, emb=empreinte(y), duree=duree,
                         caracteristiques=json.dumps(extraire(y)))
                st.rerun()

# 2. Vérification
st.header("2. Vérification")
if not EMPREINTE.exists():
    st.info("Fais d'abord l'enrôlement.")
else:
    enr = lire_npz(EMPREINTE)
    d_enrol = int(enr["duree"])
    seuil = reference()["seuils"][d_enrol]
    st.session_state.setdefault("passage", 1)
    st.session_state.setdefault("historique", [])
    n = st.session_state["passage"]

    st.markdown(f"**Passage n° {n}** : la personne suivante parle, l'enrôlement reste le même.")
    audio = st.audio_input(f"Quelqu'un parle (au moins 3 s, jusqu'à {TEST_MAX} s analysées)",
                           key=f"verification_{n}")
    if audio:
        brut, y = lire_audio(audio)
        afficher_qualite(brut)
        parole = len(y) / SR
        if parole < 1.5:
            st.error("Trop court : parle au moins 3 secondes.")
        else:
            y = y[:TEST_MAX * SR]
            score = float(enr["emb"] @ empreinte(y))
            accepte = score >= seuil
            if accepte:
                st.success("## ✅ C'est la personne enrôlée")
                st.metric("Confiance", f"{confiance(score):.0%}")
            else:
                st.error("## ❌ Ce n'est pas la personne enrôlée")
            st.caption(f"Score brut {score:.3f} · seuil {seuil:.3f} · enrôlement {d_enrol} s · "
                       f"parole analysée {min(parole, TEST_MAX):.1f} s")
            if parole < 3:
                st.caption("Moins de 3 s de parole : la décision est moins fiable.")

            with st.expander("Pourquoi ? Les plus grandes différences avec la voix enrôlée"):
                a = json.loads(str(enr["caracteristiques"]))
                for ligne in expliquer(a, extraire(y), reference()["ecart_intra"]):
                    st.write("• " + ligne)
                st.caption("Classées selon l'écart par rapport à la variation habituelle chez une même personne. "
                           "L'empreinte ECAPA combine bien plus d'indices que ces quelques mesures.")

            if st.button("➡️ Passage suivant", type="primary"):
                st.session_state["historique"].append({
                    "passage": n, "décision": "✅ reconnue" if accepte else "❌ autre personne",
                    "confiance": f"{confiance(score):.0%}" if accepte else "—",
                    "score brut": round(score, 3), "parole (s)": round(min(parole, TEST_MAX), 1)})
                st.session_state["passage"] = n + 1
                st.rerun()

    if st.session_state["historique"]:
        st.subheader("Passages précédents")
        st.dataframe(st.session_state["historique"], hide_index=True, use_container_width=True)

# 3. Suppression
st.header("3. Supprimer l'empreinte")
if EMPREINTE.exists() and st.button("Supprimer mon empreinte"):
    EMPREINTE.unlink()
    st.session_state["passage"], st.session_state["historique"] = 1, []
    st.success("Empreinte supprimée.")
    st.rerun()