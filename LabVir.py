import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import hashlib
import json
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="BioNexus Hub", page_icon="🧬", layout="wide")

api_key = None
if "OPENROUTER_API_KEY" in st.secrets:
    api_key = st.secrets["OPENROUTER_API_KEY"]
else:
    with st.sidebar:
        st.markdown("### 🔑 Configuration API")
        api_key = st.text_input("Entrez votre clé OpenRouter :", type="password")

client = None
if api_key:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# ==========================================
# DICTIONNAIRE DES DISCIPLINES (Le cerveau de l'IA)
# ==========================================
DISCIPLINES = {
    # --- DÉPARTEMENT GÉNÉTIQUE & OMHIQUES ---
    "Génétique": {"icon": "🧬", "prompt": "Expert en génétique formelle, transmission des caractères et variants pathogènes.", "tools": ["Benchling", "UCSC Genome"]},
    "Épigénétique": {"icon": "🔓", "prompt": "Expert en méthylation de l'ADN, modifications d'histones et régulation non-codante.", "tools": ["Galaxy", "IGV"]},
    "Bioinformatique Génomique": {"icon": "💻", "prompt": "Expert en analyse de données de séquençage NGS (GWAS, RNA-Seq, variant calling).", "tools": ["Galaxy", "NCBI BLAST"]},
    "Génie Génétique": {"icon": "✂️", "prompt": "Expert en CRISPR-Cas9, clonage moléculaire, construction de vecteurs et design de primers.", "tools": ["Benchling", "SnapGene"]},
    
    # --- DÉPARTEMENT BIOLOGIE CELLULAIRE & MOLÉCULAIRE ---
    "Biologie Moléculaire": {"icon": "🧪", "prompt": "Expert en expression génique, PCR, extraction d'ARN/ADN et électrophorèse.", "tools": ["Benchling", "Protocoles.io"]},
    "Régulation Cellulaire": {"icon": "♻️", "prompt": "Expert en voies de signalisation (JAK-STAT, MAPK), cycle cellulaire et apoptose.", "tools": ["KEGG", "Reactome"]},
    "Immunologie": {"icon": "🛡️", "prompt": "Expert en immunité innée/adaptative, anticorps, cytométrie en flux et réponses inflammatoires.", "tools": ["FlowJo", "IMGT"]},
    "Microbiologie": {"icon": "🦠", "prompt": "Expert en culture bactérienne, identification de souches, microbiome et antibiorésistance.", "tools": ["API Systems", "MEGAN"]},

    # --- DÉPARTEMENT BIOCHIMIE & STRUCTURALE ---
    "Biochimie": {"icon": "⚗️", "prompt": "Expert en cinétique enzymatique (Michaëlis-Menten), purification de protéines et métabolisme.", "tools": ["ChemDraw", "Protocoles.io"]},
    "Biochimie Structurale": {"icon": "🧱", "prompt": "Expert en repliement des protéines, domaines fonctionnels et interactions moléculaires.", "tools": ["AlphaFold DB", "PyMOL"]},
    "Cristallographie": {"icon": "💎", "prompt": "Expert en diffraction aux rayons X, résolution de structures 3D macromoléculaires et facteurs B.", "tools": ["RCSB PDB", "CCP4"]},

    # --- DÉPARTEMENT ÉCOLOGIE & ÉVOLUTION ---
    "Écologie": {"icon": "🌳", "prompt": "Expert en dynamique des populations, interactions biotiques et modèles mathématiques d'écosystèmes.", "tools": ["QGIS", "R (Vegan)"]},
    "Biodiversité": {"icon": "🐒", "prompt": "Expert en taxonomie, phylogénie, indices de diversité et conservation.", "tools": ["GBIF", "iNaturalist"]},
}

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================
def generate_discovery_certificate(data_context, ai_interpretation):
    timestamp = datetime.now().isoformat()
    cert_string = f"{data_context}|{ai_interpretation}|{timestamp}"
    hash_object = hashlib.sha256(cert_string.encode())
    return {
        "FINGERPRINT": hash_object.hexdigest(),
        "TIMESTAMP": timestamp,
        "DATA_HASH": hashlib.sha256(data_context.encode()).hexdigest(),
        "AI_INTERPRETATION_HASH": hashlib.sha256(ai_interpretation.encode()).hexdigest(),
        "ISSUER": "BioNexus Hub"
    }

def analyze_data_with_ai(data_context, user_question, domain_prompt):
    system_prompt = f"Tu es un {domain_prompt}. L'utilisateur te fournit des résultats bruts. Fais une interprétation biologique précise et actionnable."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Résultats:\n{data_context}\n\nQuestion: {user_question}"}
    ]
    response = client.chat.completions.create(model="mistralai/mistral-7b-instruct:free", messages=messages, temperature=0.3, stream=True)
    full_response = ""
    placeholder = st.empty()
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            placeholder.markdown(full_response + "▌")
    placeholder.markdown(full_response)
    return full_response

# ==========================================
# INTERFACE SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/biomass.png", width=80)
    st.title("BioNexus Hub")
    st.caption("Laboratoire Virtuel Multidisciplinaire")
    st.divider()
    
    st.subheader("🏢 Départements")
    module = st.radio("Choisissez le département :", 
                      ("🧬 Génétique & Omiques", 
                       "🔬 BioCell & Moléculaire",
                       "💎 Biochimie & Structurale",
                       "🌳 Écologie & Biodiversité",
                       "💬 Chat Libre"), label_visibility="collapsed")

# ==========================================
# CONTENU PRINCIPAL
# ==========================================

st.title(module)

# ------------------------------------------
# DÉPARTEMENT GÉNÉTIQUE & OMIQUES
# ------------------------------------------
if module == "🧬 Génétique & Omiques":
    sub_discipline = st.selectbox("Spécialité :", ["Bioinformatique Génomique", "Génétique", "Épigénétique", "Génie Génétique"])
    disc_info = DISCIPLINES[sub_discipline]
    
    st.info(f"🛠️ Outils recommandés : {', '.join(disc_info['tools'])}")
    
    analysis_type = st.selectbox("Type d'analyse de données :", ["GWAS (Manhattan Plot)", "RNA-Seq (Volcano Plot)"])
    
    uploaded_file = st.file_uploader(f"Uploadez votre fichier CSV ({analysis_type})", type="csv", key="genomique")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        context = ""
        
        with st.spinner("Calcul en cours..."):
            if analysis_type == "GWAS (Manhattan Plot)" and all(col in df.columns for col in ['CHR', 'BP', 'SNP', 'P']):
                df['-log10(P)'] = -np.log10(df['P'])
                df['CHR_NUM'] = df['CHR'].astype(str).str.replace('chr', '', case=False).replace({'X':23,'Y':24}).astype(int)
                df = df.sort_values('CHR_NUM')
                fig, ax = plt.subplots(figsize=(12, 5))
                x_pos, x_ticks, x_labels, color = 0, [], [], ['#1f77b4', '#aec7e8']
                for i, (chr_num, chr_data) in enumerate(df.groupby('CHR_NUM')):
                    ax.scatter(x_pos + np.arange(len(chr_data)), chr_data['-log10(P)'], color=color[i%2], s=8)
                    x_ticks.append(x_pos + len(chr_data)/2); x_labels.append(f"chr{chr_data['CHR'].iloc[0]}"); x_pos += len(chr_data)
                ax.axhline(y=-np.log10(5e-8), color='r', linestyle='--')
                ax.set_xticks(x_ticks); ax.set_xticklabels(x_labels, rotation=45)
                st.pyplot(fig)
                sig = df[df['P'] < 5e-8].head(15)
                if not sig.empty: st.dataframe(sig); context = sig.to_string(index=False)
                    
            elif analysis_type == "RNA-Seq (Volcano Plot)" and all(col in df.columns for col in ['Gene', 'Log2FC', 'PValue']):
                df['PValue'] = df['PValue'].clip(lower=1e-300)
                df['-log10(P)'] = -np.log10(df['PValue'])
                sig = (df['PValue'] < 0.05) & (abs(df['Log2FC']) > 1)
                fig, ax = plt.subplots()
                ax.scatter(df['Log2FC'], df['-log10(P)'], color='grey', s=10)
                ax.scatter(df[sig]['Log2FC'], df[sig]['-log10(P)'], color='red', s=15)
                ax.axvline(x=-1, color='b', linestyle='--'); ax.axvline(x=1, color='b', linestyle='--')
                st.pyplot(fig)
                sig_genes = df[sig].head(15)
                if not sig_genes.empty: st.dataframe(sig_genes); context = sig_genes.to_string(index=False)
            else:
                st.error("Colonnes manquantes pour cette analyse.")

        if context and api_key:
            user_q = st.text_input("🧠 Interprétation IA :", f"Analyse ces résultats de {sub_discipline} et propose des hypothèses biologiques.")
            if st.button("🤖 Interpréter & Certifier", key="gen_btn"):
                ai_resp = analyze_data_with_ai(context, user_q, disc_info['prompt'])
                cert = generate_discovery_certificate(context, ai_resp)
                st.json(cert)
                st.download_button("📥 Certificat", json.dumps(cert, indent=4), f"cert_{cert['TIMESTAMP'][:10]}.json", "application/json")

# ------------------------------------------
# DÉPARTEMENT BIOCHIMIE & STRUCTURALE
# ------------------------------------------
elif module == "💎 Biochimie & Structurale":
    sub_discipline = st.selectbox("Spécialité :", ["Biochimie", "Biochimie Structurale", "Cristallographie"])
    disc_info = DISCIPLINES[sub_discipline]
    st.info(f"🛠️ Outils recommandés : {', '.join(disc_info['tools'])}")
    
    st.subheader("Calculatrice de Cinétique Enzymatique (Michaëlis-Menten)")
    st.markdown("Entrez vos données de vitesse (V) en fonction de la concentration en substrat ([S]) pour fitter la courbe.")
    
    # Saisie simplifiée
    col1, col2 = st.columns(2)
    with col1:
        s_data = st.text_input("Concentrations [S] (séparées par virgules)", "1,2,5,10,20,50")
    with col2:
        v_data = st.text_input("Vitesses V (séparées par virgules)", "0.5,0.9,1.5,1.8,2.1,2.3")
        
    if st.button("Fitter la Courbe"):
        try:
            S = np.array([float(x) for x in s_data.split(',')])
            V = np.array([float(x) for x in v_data.split(',')])
            
            # Modèle de Michaëlis-Menten
            from scipy.optimize import curve_fit
            def michaelis_menten(S, Vmax, Km):
                return (Vmax * S) / (Km + S)
            
            popt, _ = curve_fit(michaelis_menten, S, V)
            Vmax_fit, Km_fit = popt
            
            S_fit = np.linspace(0, max(S)*1.2, 100)
            V_fit = michaelis_menten(S_fit, Vmax_fit, Km_fit)
            
            fig, ax = plt.subplots()
            ax.scatter(S, V, color='red', label='Données expérimentales')
            ax.plot(S_fit, V_fit, color='blue', label=f'Fit: Vmax={Vmax_fit:.2f}, Km={Km_fit:.2f}')
            ax.set_xlabel('[S]'); ax.set_ylabel('V'); ax.legend()
            st.pyplot(fig)
            
            context = f"Vmax calculé: {Vmax_fit:.4f}\nKm calculé: {Km_fit:.4f}"
            st.success(f"Paramètres calculés : Vmax = {Vmax_fit:.2f}, Km = {Km_fit:.2f}")
            
            if api_key:
                user_q = st.text_input("🧠 Interprétation IA :", "Que signifient ces paramètres Vmax et Km pour l'efficacité de cette enzyme ?")
                if st.button("🤖 Interpréter & Certifier", key="bioch_btn"):
                    ai_resp = analyze_data_with_ai(context, user_q, disc_info['prompt'])
                    cert = generate_discovery_certificate(context, ai_resp)
                    st.json(cert)
        except Exception as e:
            st.error(f"Erreur de calcul. Vérifiez vos données. {e}")

# ------------------------------------------
# DÉPARTEMENT BIOLOGIE CELLULAIRE & MOLÉCULAIRE
# ------------------------------------------
elif module == "🔬 BioCell & Moléculaire":
    sub_discipline = st.selectbox("Spécialité :", ["Biologie Moléculaire", "Régulation Cellulaire", "Immunologie", "Microbiologie"])
    disc_info = DISCIPLINES[sub_discipline]
    st.info(f"🛠️ Outils recommandés : {', '.join(disc_info['tools'])}")
    
    st.subheader("Analyse de Courbe de Croissance Bactérienne/Cellulaire")
    st.markdown("Format CSV attendu : Colonnes `Temps` (heures) et `OD600` (Densité Optique)")
    uploaded_file = st.file_uploader("Uploadez vos données de croissance (.csv)", type="csv", key="micro")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Temps' in df.columns and 'OD600' in df.columns:
            fig, ax = plt.subplots()
            ax.plot(df['Temps'], df['OD600'], marker='o', linestyle='-')
            ax.set_xlabel('Temps (h)'); ax.set_ylabel('OD600'); ax.set_title('Courbe de Croissance')
            st.pyplot(fig)
            
            # Calcul temps de génération (phase exponentielle approximative)
            log_phase = df[(df['OD600'] > df['OD600'].max()*0.2) & (df['OD600'] < df['OD600'].max()*0.8)]
            context = f"Données complètes:\n{df.to_string(index=False)}\nPhase exponentielle détectée:\n{log_phase.to_string(index=False)}"
            
            if api_key:
                user_q = st.text_input("🧠 Interprétation IA :", "Analyse cette courbe de croissance et identifie les phases (lag, log, stationnaire).")
                if st.button("🤖 Interpréter & Certifier", key="micro_btn"):
                    ai_resp = analyze_data_with_ai(context, user_q, disc_info['prompt'])
                    cert = generate_discovery_certificate(context, ai_resp)
                    st.json(cert)
        else:
            st.error("Colonnes 'Temps' et 'OD600' requises.")

# ------------------------------------------
# DÉPARTEMENT ÉCOLOGIE & BIODIVERSITÉ
# ------------------------------------------
elif module == "🌳 Écologie & Biodiversité":
    sub_discipline = st.selectbox("Spécialité :", ["Écologie", "Biodiversité"])
    disc_info = DISCIPLINES[sub_discipline]
    st.info(f"🛠️ Outils recommandés : {', '.join(disc_info['tools'])}")
    
    st.subheader("Analyse de Communauté (Abondance)")
    st.markdown("Format CSV attendu : Colonnes `Espece` et `Abondance`")
    uploaded_file = st.file_uploader("Uploadez vos données de terrain (.csv)", type="csv", key="eco")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Espece' in df.columns and 'Abondance' in df.columns:
            df_sorted = df.sort_values('Abondance', ascending=False)
            fig, ax = plt.subplots()
            ax.bar(df_sorted['Espece'].head(15), df_sorted['Abondance'].head(15), color='green')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            # Calcul indice de Shannon (si demandé)
            p = df['Abondance'] / df['Abondance'].sum()
            shannon = -sum(p * np.log(p))
            context = f"Indice de Shannon calculé: {shannon:.2f}\n\nTop espèces:\n{df_sorted.head(10).to_string(index=False)}"
            st.metric("Indice de Shannon (Diversité)", f"{shannon:.2f}")
            
            if api_key:
                user_q = st.text_input("🧠 Interprétation IA :", "Interprète cet indice de Shannon et la structure de cette communauté.")
                if st.button("🤖 Interpréter & Certifier", key="eco_btn"):
                    ai_resp = analyze_data_with_ai(context, user_q, disc_info['prompt'])
                    cert = generate_discovery_certificate(context, ai_resp)
                    st.json(cert)

# ------------------------------------------
# CHAT LIBRE
# ------------------------------------------
elif module == "💬 Chat Libre":
    if not api_key:
        st.warning("Clé API requise.")
    else:
        sub_discipline = st.selectbox("Dans quelle spécialité ?", list(DISCIPLINES.keys()))
        disc_info = DISCIPLINES[sub_discipline]
        st.info(f"Vous discutez avec un expert en **{sub_discipline}** {disc_info['icon']}")
        
        if "msg_free" not in st.session_state:
            st.session_state.msg_free = []
            
        for msg in st.session_state.msg_free:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
                
        if prompt := st.chat_input("Posez votre question..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.msg_free.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                messages_api = [{"role": "system", "content": disc_info['prompt']}] + st.session_state.msg_free
                resp = client.chat.completions.create(model="mistralai/mistral-7b-instruct:free", messages=messages_api, stream=True, temperature=0.3)
                full = ""
                ph = st.empty()
                for c in resp:
                    if c.choices[0].delta.content:
                        full += c.choices[0].delta.content
                        ph.markdown(full + "▌")
                ph.markdown(full)
            st.session_state.msg_free.append({"role": "assistant", "content": full})