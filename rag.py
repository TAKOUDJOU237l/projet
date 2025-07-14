import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings , HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Importation pour Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

import os
import warnings

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

def get_pdf_text(pdf_docs):
    """Extrait le texte de tous les PDFs uploadés"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
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
        st.warning("⚠️ Veuillez d'abord traiter vos documents avant de poser une question.")
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
        page_title="Chat avec PDFs multiples",
        page_icon="📚",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)
    
    # Initialiser les variables de session
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("💬 Chat avec PDFs multiples")
    st.markdown("Uploadez vos PDFs et posez des questions sur leur contenu!")
    
    # Interface principale
    user_question = st.text_input("❓ Posez une question sur vos documents:")
    if user_question:
        handle_userinput(user_question)
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.subheader("📄 Vos documents")
        
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
        
        # Upload des fichiers PDF
        pdf_docs = st.file_uploader(
            "📁 Uploadez vos PDFs ici et cliquez sur 'Traiter'",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        # Bouton de traitement
        if st.button("🔄 Traiter les documents", type="primary"):
            if pdf_docs:
                with st.spinner("Traitement en cours..."):
                    # Charger le modèle d'embedding
                    embeddings = load_embedding_model(selected_model)
                    
                    if embeddings is not None:
                        # Extraire le texte des PDFs
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if raw_text.strip():
                            # Créer les chunks de texte
                            text_chunks = get_text_chunks(raw_text)
                            
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
                                    st.success("✅ Documents traités avec succès!")
                                    st.info(f"📊 Statistiques:")
                                    st.info(f"- Nombre de documents: {len(pdf_docs)}")
                                    st.info(f"- Nombre de chunks: {len(text_chunks)}")
                                    st.info(f"- Modèle d'embedding: {selected_model}")
                                    st.info(f"- LLM: {selected_llm}")
                                else:
                                    st.error("❌ Impossible de créer la chaîne de conversation")
                            else:
                                st.error("❌ Impossible de créer le vector store")
                        else:
                            st.error("❌ Aucun texte extrait des PDFs")
                    else:
                        st.error("❌ Impossible de charger le modèle d'embedding")
            else:
                st.warning("⚠️ Veuillez uploader au moins un fichier PDF")
        
        # Bouton pour effacer l'historique
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.chat_history = None
            st.rerun()
    
    # Affichage du statut
    if 'processed' in st.session_state and st.session_state.processed:
        st.success("🎉 Documents prêts pour les questions!")
        st.info("💡 Vous pouvez maintenant poser des questions sur vos documents!")
    else:
        st.info("👈 Commencez par uploader et traiter vos documents dans la sidebar")
        
        # Affichage d'exemple si aucun document n'est traité
        st.markdown("### 📝 Exemple d'utilisation:")
        st.markdown("""
        1. **Uploadez vos PDFs** dans la sidebar
        2. **Choisissez un modèle d'embedding** (recommandé: HuggingFace Sentence Transformers)
        3. **Choisissez un modèle LLM** (Gemini Pro ou Gemini Pro 1.5)
        4. **Cliquez sur 'Traiter les documents'**
        5. **Posez vos questions** dans le champ de texte ci-dessus
        
        **Note:** Vous devez avoir une clé API Google dans votre fichier `.env` :
        ```
        GOOGLE_API_KEY=your_google_api_key_here
        ```
        
        **Comment obtenir une clé API Google:**
        1. Allez sur [Google AI Studio](https://aistudio.google.com/)
        2. Créez un nouveau projet ou sélectionnez un projet existant
        3. Générez une clé API
        4. Ajoutez la clé dans votre fichier `.env`
        """)

if __name__ == '__main__':
    main()