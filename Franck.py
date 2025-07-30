import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import docx2txt
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings , HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Importation pour Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
# Importations pour traiter les liens web
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
# Importations pour traiter les fichiers Excel
import pandas as pd
import openpyxl
import xlrd

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

import os
import warnings
import io

# Supprimer les avertissements de dépréciation
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration des modèles disponibles
EMBEDDING_MODELS = {
    "OpenAI": {
        "class": OpenAIEmbeddings,
        "params": {},
        "requires_api_key": True
    },
    "HuggingFace Instructor": {
        "class": HuggingFaceInstructEmbeddings,
        "params": {"model_name": "hku-nlp/instructor-xl"},
        "requires_api_key": False
    },
    "HuggingFace Sentence Transformers": {
        "class": HuggingFaceEmbeddings,
        "params": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"}
        },
        "requires_api_key": False
    },
    "HuggingFace Multilingual": {
        "class": HuggingFaceEmbeddings,
        "params": {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "model_kwargs": {"device": "cpu"}
        },
        "requires_api_key": False
    }
}

# Configuration des modèles LLM disponibles
LLM_MODELS = {
    "Gemini 2.0 Flash": {
        "model_name": "gemini-2.0-flash-exp",
        "api_key_env": "GOOGLE_API_KEY",
        "description": "Google Gemini Pro - Modèle puissant et rapide"
    },
    "Gemini Pro 1.5": {
        "model_name": "gemini-1.5-pro",
        "api_key_env": "GOOGLE_API_KEY",
        "description": "Google Gemini 1.5 Pro - Modèle le plus avancé"
    }
}

def is_valid_url(url):
    """Vérifie si l'URL est valide"""
    try:
        parsed = urllib.parse.urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except:
        return False

def extract_text_from_url(url):
    """Extrait le texte d'une URL donnée"""
    try:
        # Headers pour éviter les blocages
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Faire la requête
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parser le HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Supprimer les scripts et styles
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extraire le texte
        text = soup.get_text()
        
        # Nettoyer le texte
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        raise Exception(f"Erreur lors de l'extraction du texte de {url}: {str(e)}")

def get_web_content(urls):
    """Extrait le contenu de plusieurs URLs"""
    all_text = ""
    successful_urls = []
    failed_urls = []
    
    for url in urls:
        if not is_valid_url(url):
            failed_urls.append((url, "URL invalide"))
            continue
            
        try:
            with st.spinner(f"Extraction du contenu de {url}..."):
                text = extract_text_from_url(url)
                if text.strip():
                    all_text += f"\n\n=== Contenu de {url} ===\n\n{text}\n\n"
                    successful_urls.append(url)
                else:
                    failed_urls.append((url, "Aucun texte extrait"))
        except Exception as e:
            failed_urls.append((url, str(e)))
    
    return all_text, successful_urls, failed_urls

def get_pdf_text(pdf_docs):
    """Extrait le texte de tous les PDFs uploadés"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_excel_text(excel_files):
    """Extrait le texte de tous les fichiers Excel uploadés"""
    text = ""
    for excel_file in excel_files:
        try:
            # Lire le fichier Excel
            if excel_file.name.endswith('.xlsx'):
                # Pour les fichiers .xlsx
                df_dict = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
            else:
                # Pour les fichiers .xls
                df_dict = pd.read_excel(excel_file, sheet_name=None, engine='xlrd')
            
            # Traiter chaque feuille
            for sheet_name, df in df_dict.items():
                if not df.empty:
                    text += f"\n\n=== Feuille: {sheet_name} ===\n\n"
                    
                    # Convertir le DataFrame en texte structuré
                    # Ajouter les en-têtes de colonnes
                    headers = list(df.columns)
                    text += "Colonnes: " + " | ".join(str(h) for h in headers) + "\n\n"
                    
                    # Ajouter les données ligne par ligne
                    for index, row in df.iterrows():
                        row_text = []
                        for col in headers:
                            cell_value = row[col]
                            if pd.notna(cell_value):  # Ignorer les valeurs NaN
                                row_text.append(f"{col}: {cell_value}")
                        
                        if row_text:  # Ajouter seulement si la ligne contient des données
                            text += "Ligne " + str(index + 1) + " - " + " | ".join(row_text) + "\n"
                    
                    text += "\n"
        except Exception as e:
            st.warning(f"Erreur lors de la lecture du fichier Excel {excel_file.name}: {str(e)}")
            continue
    return text
    """Extrait le texte de tous les fichiers Word uploadés"""
def get_docx_text(docx_docs):    
    
    text = ""
    for docx_file in docx_docs:
        try:
            # Méthode 1: Utiliser docx2txt (plus simple et robuste)
            docx_text = docx2txt.process(docx_file)
            if docx_text:
                text += docx_text + "\n"
            else:
                # Méthode 2: Utiliser python-docx comme fallback
                doc = Document(docx_file)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
        except Exception as e:
            st.warning(f"Erreur lors de la lecture d'un fichier Word: {str(e)}")
            continue
    return text

def get_documents_text(uploaded_files):
    """Extrait le texte de tous les documents uploadés (PDF, Word et Excel)"""
    text = ""
    pdf_files = []
    word_files = []
    excel_files = []
    
    # Séparer les fichiers par type
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_files.append(file)
        elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                          "application/msword"]:
            word_files.append(file)
        elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                          "application/vnd.ms-excel"]:
            excel_files.append(file)
    
    # Extraire le texte des PDFs
    if pdf_files:
        with st.spinner(f"Extraction du texte de {len(pdf_files)} fichier(s) PDF..."):
            text += get_pdf_text(pdf_files)
    
    # Extraire le texte des fichiers Word
    if word_files:
        with st.spinner(f"Extraction du texte de {len(word_files)} fichier(s) Word..."):
            text += get_docx_text(word_files)
    
    # Extraire le texte des fichiers Excel
    if excel_files:
        with st.spinner(f"Extraction du texte de {len(excel_files)} fichier(s) Excel..."):
            text += get_excel_text(excel_files)
    
    return text

def get_text_chunks(text):
    """Divise le texte en chunks plus petits"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def load_embedding_model(model_name):
    """Charge le modèle d'embedding avec mise en cache"""
    try:
        model_config = EMBEDDING_MODELS[model_name]
        
        # Charger le modèle avec suppression des warnings
        with st.spinner(f"Chargement du modèle {model_name}..."):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                embeddings = model_config["class"](**model_config["params"])
            st.success(f"Modèle {model_name} chargé avec succès!")
            return embeddings
            
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle {model_name}: {str(e)}")
        return None

def get_vectorstore(text_chunks, embeddings):
    """Crée le vector store avec les embeddings"""
    try:
        with st.spinner("Création du vector store..."):
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            st.success("Vector store créé avec succès!")
            return vectorstore
    except Exception as e:
        st.error(f"Erreur lors de la création du vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore, llm_model="Gemini Pro"):
    """Crée la chaîne de conversation avec le LLM Gemini"""
    try:
        # Vérifier que la clé API Google est disponible
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("❌ Clé API Google manquante! Ajoutez GOOGLE_API_KEY dans votre fichier .env")
            return None
        
        # Configuration du modèle Gemini
        model_config = LLM_MODELS[llm_model]
        
        llm = ChatGoogleGenerativeAI(
            model=model_config["model_name"],
            google_api_key=google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True  # Pour la compatibilité avec Gemini
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        return conversation_chain
        
    except Exception as e:
        st.error(f"Erreur lors de la création de la chaîne de conversation: {str(e)}")
        return None

def handle_userinput(user_question):
    """Traite la question de l'utilisateur et génère une réponse"""
    if st.session_state.conversation is None:
        st.warning("⚠️ Veuillez d'abord traiter vos documents/liens avant de poser une question.")
        return
    
    try:
        with st.spinner("Génération de la réponse..."):
            response = st.session_state.conversation({'question': user_question})
            
        # Stocker l'historique de chat
        st.session_state.chat_history = response['chat_history']
        
        # Afficher la conversation
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Erreur lors de la génération de la réponse: {str(e)}")
        st.error("Vérifiez votre clé API Google et votre connexion internet.")

def main():
    load_dotenv()
    
    st.set_page_config(
        page_title="Chat avec vos documents et liens",
        page_icon="📚",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)
    
    # Initialiser les variables de session
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("💬 Chat avec vos documents et liens web")
    st.markdown("Uploadez vos PDFs, fichiers Word, Excel et ajoutez des liens web, puis posez des questions sur leur contenu!")
    
    # Interface principale
    user_question = st.text_input("❓ Posez une question sur vos documents/liens:")
    if user_question:
        handle_userinput(user_question)
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.subheader("📄 Vos sources")
        
        # Onglets pour séparer les types de sources
        tab1, tab2 = st.tabs(["📁 Fichiers", "🔗 Liens Web"])
        
        with tab1:
            st.subheader("📁 Documents")
            uploaded_files = st.file_uploader(
                "Uploadez vos documents ici",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'doc', 'xlsx', 'xls'],
                help="Formats supportés: PDF, DOCX, DOC, XLSX, XLS"
            )
        
        with tab2:
            st.subheader("🔗 Liens Web")
            st.markdown("Ajoutez des URLs (une par ligne):")
            urls_input = st.text_area(
                "URLs:",
                height=100,
                placeholder="https://example.com/page1\nhttps://example.com/page2\n...",
                help="Entrez chaque URL sur une nouvelle ligne"
            )
        
        # Sélection du modèle d'embedding
        st.subheader("🤖 Configuration du modèle")
        selected_model = st.selectbox(
            "Choisissez un modèle d'embedding:",
            list(EMBEDDING_MODELS.keys()),
            index=2,  # Par défaut sur HuggingFace Sentence Transformers
            help="Les modèles HuggingFace sont gratuits et fonctionnent hors ligne"
        )
        
        # Sélection du modèle LLM
        selected_llm = st.selectbox(
            "Choisissez un modèle LLM:",
            list(LLM_MODELS.keys()),
            index=0,  # Par défaut sur Gemini Pro
            help="Modèles Google Gemini pour la génération de réponses"
        )
        
        # Affichage des informations sur les modèles sélectionnés
        model_info = EMBEDDING_MODELS[selected_model]
        if model_info["requires_api_key"]:
            st.warning("⚠️ Ce modèle d'embedding nécessite une clé API")
        else:
            st.info("✅ Ce modèle d'embedding est gratuit et fonctionne hors ligne")
        
        # Information sur le LLM
        st.subheader("🧠 LLM Configuration")
        llm_info = LLM_MODELS[selected_llm]
        st.info(f"🤖 LLM utilisé: {llm_info['description']}")
        st.warning("⚠️ Une clé API Google est requise pour Gemini")
        
        # Bouton de traitement
        if st.button("🔄 Traiter les sources", type="primary"):
            # Préparer les URLs
            urls = []
            if urls_input.strip():
                urls = [url.strip() for url in urls_input.strip().split('\n') if url.strip()]
            
            # Vérifier qu'au moins une source est fournie
            if not uploaded_files and not urls:
                st.warning("⚠️ Veuillez uploader au moins un fichier ou fournir au moins une URL")
                return
            
            with st.spinner("Traitement en cours..."):
                # Charger le modèle d'embedding
                embeddings = load_embedding_model(selected_model)
                
                if embeddings is not None:
                    all_text = ""
                    processing_stats = {
                        'pdf_count': 0,
                        'word_count': 0,
                        'excel_count': 0,
                        'url_count': 0,
                        'successful_urls': [],
                        'failed_urls': []
                    }
                    
                    # Traiter les fichiers uploadés
                    if uploaded_files:
                        file_text = get_documents_text(uploaded_files)
                        if file_text.strip():
                            all_text += file_text
                            
                            # Compter les types de fichiers
                            processing_stats['pdf_count'] = len([f for f in uploaded_files if f.type == "application/pdf"])
                            processing_stats['word_count'] = len([f for f in uploaded_files if f.type in [
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                "application/msword"
                            ]])
                            processing_stats['excel_count'] = len([f for f in uploaded_files if f.type in [
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                "application/vnd.ms-excel"
                            ]])
                    
                    # Traiter les URLs
                    if urls:
                        web_text, successful_urls, failed_urls = get_web_content(urls)
                        if web_text.strip():
                            all_text += web_text
                            
                        processing_stats['url_count'] = len(successful_urls)
                        processing_stats['successful_urls'] = successful_urls
                        processing_stats['failed_urls'] = failed_urls
                    
                    if all_text.strip():
                        # Créer les chunks de texte
                        text_chunks = get_text_chunks(all_text)
                        
                        # Créer le vector store
                        vectorstore = get_vectorstore(text_chunks, embeddings)
                        
                        if vectorstore is not None:
                            # Créer la chaîne de conversation avec le modèle sélectionné
                            conversation_chain = get_conversation_chain(vectorstore, selected_llm)
                            
                            if conversation_chain is not None:
                                # Stocker dans la session
                                st.session_state.conversation = conversation_chain
                                st.session_state.vectorstore = vectorstore
                                st.session_state.processed = True
                                
                                # Afficher les statistiques
                                st.success("✅ Sources traitées avec succès!")
                                st.info(f"📊 Statistiques:")
                                st.info(f"- Fichiers PDF: {processing_stats['pdf_count']}")
                                st.info(f"- Fichiers Word: {processing_stats['word_count']}")
                                st.info(f"- Fichiers Excel: {processing_stats['excel_count']}")
                                st.info(f"- URLs réussies: {processing_stats['url_count']}")
                                st.info(f"- Total documents: {processing_stats['pdf_count'] + processing_stats['word_count'] + processing_stats['excel_count']}")
                                st.info(f"- Nombre de chunks: {len(text_chunks)}")
                                st.info(f"- Modèle d'embedding: {selected_model}")
                                st.info(f"- LLM: {selected_llm}")
                                
                                # Afficher les URLs traitées avec succès
                                if processing_stats['successful_urls']:
                                    st.success("✅ URLs traitées avec succès:")
                                    for url in processing_stats['successful_urls']:
                                        st.success(f"• {url}")
                                
                                # Afficher les URLs échouées
                                if processing_stats['failed_urls']:
                                    st.error("❌ URLs échouées:")
                                    for url, error in processing_stats['failed_urls']:
                                        st.error(f"• {url}: {error}")
                            else:
                                st.error("❌ Impossible de créer la chaîne de conversation")
                        else:
                            st.error("❌ Impossible de créer le vector store")
                    else:
                        st.error("❌ Aucun texte extrait des sources")
                else:
                    st.error("❌ Impossible de charger le modèle d'embedding")
        
        # Bouton pour effacer l'historique
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.chat_history = None
            st.rerun()
    
    # Affichage du statut
    if 'processed' in st.session_state and st.session_state.processed:
        st.success("🎉 Sources prêtes pour les questions!")
        st.info("💡 Vous pouvez maintenant poser des questions sur vos documents et liens!")
    else:
        st.info("👈 Commencez par uploader vos documents et/ou ajouter des liens dans la sidebar")
        
        # Affichage d'exemple si aucun document n'est traité
        st.markdown("### 📝 Exemple d'utilisation:")
        st.markdown("""
        1. **Uploadez vos documents** dans l'onglet "📁 Fichiers" (PDF, DOCX, DOC, XLSX, XLS)
        2. **Ajoutez des liens web** dans l'onglet "🔗 Liens Web" (une URL par ligne)
        3. **Choisissez un modèle d'embedding** (recommandé: HuggingFace Sentence Transformers)
        4. **Choisissez un modèle LLM** (Gemini Pro ou Gemini Pro 1.5)
        5. **Cliquez sur 'Traiter les sources'**
        6. **Posez vos questions** dans le champ de texte ci-dessus
        
        **Sources supportées:**
        - 📄 PDF (.pdf)
        - 📝 Word (.docx, .doc)
        - 📊 Excel (.xlsx, .xls)
        - 🔗 Pages web (HTTP/HTTPS)
        
        **Exemples d'URLs:**
        ```
        https://fr.wikipedia.org/wiki/Intelligence_artificielle
        https://www.example.com/article
        https://blog.example.com/post/123
        ```
        
        **Note:** Vous devez avoir une clé API Google dans votre fichier `.env` :
        ```
        GOOGLE_API_KEY=your_google_api_key_here
        ```
        
        **Bibliothèques requises:**
        ```bash
        pip install python-docx docx2txt langchain-google-genai beautifulsoup4 requests pandas openpyxl xlrd
        ```
        
        **Fonctionnalités web:**
        - Extraction automatique du contenu textuel des pages web
        - Support des sites web standards (HTML)
        - Gestion des erreurs et timeouts
        - Headers personnalisés pour éviter les blocages
        
        **Fonctionnalités Excel:**
        - Support des fichiers .xlsx et .xls
        - Extraction de toutes les feuilles de calcul
        - Préservation de la structure des données (colonnes et lignes)
        - Gestion des cellules vides et des erreurs
        """)

if __name__ == '__main__':
    main()