import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Import pour Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

import os
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration de la page en premier
st.set_page_config(
    page_title="Chat avec PDFs multiples - Gemini 2.0 Flash", 
    page_icon="📚", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Configuration des modèles Gemini disponibles
GEMINI_MODELS = {
    "Gemini 2.0 Flash": {
        "model_name": "gemini-2.0-flash-exp",
        "description": "Modèle ultra-rapide et performant de Google",
        "features": ["Multimodal", "Très rapide", "Économique"],
        "recommended": True
    },
    "Gemini 1.5 Pro": {
        "model_name": "gemini-1.5-pro",
        "description": "Modèle puissant avec large contexte",
        "features": ["2M tokens de contexte", "Très performant"],
        "recommended": False
    },
    "Gemini 1.5 Flash": {
        "model_name": "gemini-1.5-flash",
        "description": "Version rapide de Gemini 1.5",
        "features": ["Rapide", "Économique", "1M tokens"],
        "recommended": False
    }
}

def initialize_session_state():
    """Initialise les variables de session de manière sécurisée"""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "processing" not in st.session_state:
        st.session_state.processing = False

def check_gemini_api_key():
    """Vérifie si la clé API Gemini est configurée"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False
    
    try:
        genai.configure(api_key=api_key)
        # Test simple pour vérifier la validité de la clé
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello")
        return True
    except Exception as e:
        st.error(f"Erreur avec la clé API Gemini: {str(e)}")
        return False

def get_pdf_text(pdf_docs):
    """Extrait le texte des PDFs"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.warning(f"Erreur lors de la lecture d'un PDF: {str(e)}")
            continue
    return text

def get_text_chunks(text):
    """Divise le texte en chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

@st.cache_resource
def load_embedding_model(model_name):
    """Charge le modèle d'embedding avec cache"""
    try:
        model_config = EMBEDDING_MODELS[model_name]
        if model_config["requires_api_key"] and model_name == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("Clé API OpenAI manquante. Veuillez l'ajouter dans votre fichier .env")
                return None
        
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
    """Crée le vector store"""
    try:
        with st.spinner("Création du vector store..."):
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            st.success("Vector store créé avec succès!")
            return vectorstore
    except Exception as e:
        st.error(f"Erreur lors de la création du vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore, selected_llm):
    """Crée la chaîne de conversation avec Gemini"""
    try:
        # Vérification de la clé API
        if not check_gemini_api_key():
            st.error("❌ Clé API Gemini manquante ou invalide!")
            st.info("💡 Ajoutez GOOGLE_API_KEY dans votre fichier .env")
            st.info("🔑 Obtenez votre clé sur: https://makersuite.google.com/app/apikey")
            return None
        
        # Configuration du modèle Gemini
        model_name = GEMINI_MODELS[selected_llm]["model_name"]
        
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            max_tokens=1024,
            timeout=30,
            max_retries=2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        st.success(f"✅ Modèle {selected_llm} configuré avec succès!")
        return conversation_chain
        
    except Exception as e:
        st.error(f"Erreur avec le modèle Gemini: {str(e)}")
        if "api_key" in str(e).lower():
            st.info("💡 Vérifiez que votre clé API Gemini est correcte")
        elif "quota" in str(e).lower():
            st.info("💡 Quota API dépassé, attendez ou vérifiez votre facturation")
        return None

def handle_userinput(user_question):
    """Gère l'input utilisateur avec gestion d'erreurs améliorée"""
    if st.session_state.conversation is None:
        st.warning("⚠️ Veuillez d'abord traiter vos documents avant de poser une question.")
        return
    
    try:
        with st.spinner("Génération de la réponse avec Gemini 2.0 Flash..."):
            response = st.session_state.conversation({'question': user_question})
        
        # Récupération de l'historique depuis la mémoire
        chat_history = st.session_state.conversation.memory.chat_memory.messages
        
        # Affichage direct de la conversation
        for i, message in enumerate(chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                
        # Affichage optionnel des sources
        if hasattr(response, 'source_documents') and response['source_documents']:
            with st.expander("📄 Sources utilisées"):
                for i, doc in enumerate(response['source_documents']):
                    st.write(f"**Source {i+1}:**")
                    st.write(doc.page_content[:200] + "...")
                    st.write("---")
                
    except Exception as e:
        st.error(f"Erreur lors de la génération de la réponse: {str(e)}")
        if "api_key" in str(e).lower():
            st.info("💡 Vérifiez votre clé API Gemini")
        elif "quota" in str(e).lower():
            st.info("💡 Quota API dépassé")
        elif "safety" in str(e).lower():
            st.info("💡 La réponse a été bloquée par les filtres de sécurité")

def reset_session():
    """Remet à zéro la session de manière sécurisée"""
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.vectorstore = None
    st.session_state.processed = False
    st.session_state.processing = False

def main():
    """Fonction principale"""
    load_dotenv()
    
    # Initialisation des variables de session
    initialize_session_state()
    
    # CSS et templates
    st.write(css, unsafe_allow_html=True)

    st.header("💬 Chat avec PDFs multiples - Gemini 2.0 Flash")
    st.markdown("Uploadez vos PDFs et posez des questions avec la puissance de Gemini 2.0 Flash !")

    # Zone de question
    user_question = st.text_input("❓ Posez une question sur vos documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar
    with st.sidebar:
        st.subheader("📄 Vos documents")
        
        # Vérification du statut Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            if check_gemini_api_key():
                st.success("✅ Gemini API connectée")
            else:
                st.error("❌ Problème avec l'API Gemini")
        else:
            st.error("❌ Clé API Gemini manquante")
            st.info("💡 Ajoutez GOOGLE_API_KEY dans votre .env")

        st.subheader("🤖 Configuration du modèle")

        # Sélection du modèle d'embedding
        selected_model = st.selectbox(
            "Choisissez un modèle d'embedding:",
            list(EMBEDDING_MODELS.keys()),
            index=2,
            help="Les modèles HuggingFace sont gratuits et fonctionnent hors ligne"
        )
        
        model_info = EMBEDDING_MODELS[selected_model]
        if model_info["requires_api_key"]:
            st.warning("⚠️ Ce modèle nécessite une clé API")
        else:
            st.info("✅ Ce modèle est gratuit et fonctionne hors ligne")

        st.subheader("🧠 LLM Configuration")
        
        # Sélection du modèle Gemini
        selected_llm = st.selectbox(
            "Choisissez un modèle Gemini:",
            list(GEMINI_MODELS.keys()),
            index=0,
            help="Gemini 2.0 Flash est recommandé pour sa rapidité"
        )
        
        llm_info = GEMINI_MODELS[selected_llm]
        st.info(f"🤖 {llm_info['description']}")
        
        # Affichage des caractéristiques
        st.markdown("**Caractéristiques:**")
        for feature in llm_info["features"]:
            st.markdown(f"• {feature}")
        
        if llm_info["recommended"]:
            st.success("⭐ Modèle recommandé")

        # Upload des fichiers
        pdf_docs = st.file_uploader(
            "📁 Uploadez vos PDFs ici et cliquez sur 'Traiter'",
            accept_multiple_files=True,
            type=['pdf']
        )

        # Bouton de traitement
        if st.button("🔄 Traiter les documents", type="primary", disabled=st.session_state.processing):
            if pdf_docs:
                st.session_state.processing = True
                
                try:
                    with st.spinner("Traitement en cours..."):
                        # Chargement du modèle d'embedding
                        embeddings = load_embedding_model(selected_model)
                        if embeddings is not None:
                            # Extraction du texte
                            raw_text = get_pdf_text(pdf_docs)
                            if raw_text.strip():
                                # Création des chunks
                                text_chunks = get_text_chunks(raw_text)
                                # Création du vector store
                                vectorstore = get_vectorstore(text_chunks, embeddings)
                                if vectorstore is not None:
                                    # Création de la chaîne de conversation
                                    conversation_chain = get_conversation_chain(vectorstore, selected_llm)
                                    if conversation_chain is not None:
                                        # Mise à jour de la session
                                        st.session_state.conversation = conversation_chain
                                        st.session_state.vectorstore = vectorstore
                                        st.session_state.processed = True
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
                
                except Exception as e:
                    st.error(f"Erreur lors du traitement: {str(e)}")
                
                finally:
                    st.session_state.processing = False
                    
            else:
                st.warning("⚠️ Veuillez uploader au moins un fichier PDF")

        # Boutons de gestion
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Effacer l'historique"):
                if st.session_state.conversation and st.session_state.conversation.memory:
                    st.session_state.conversation.memory.clear()
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset complet"):
                reset_session()
                st.rerun()

        # Informations sur Gemini
        st.markdown("### 🚀 Avantages Gemini 2.0 Flash:")
        st.markdown("""
        - ⚡ **Ultra-rapide** - Latence très faible
        - 🧠 **Très performant** - Dernière génération
        - 🌐 **Multimodal** - Texte, images, audio
        - 💰 **Économique** - Tarification attractive
        - 🔒 **Sécurisé** - Filtres de sécurité intégrés
        """)

        # Informations sur les coûts
        st.markdown("### 💰 Tarification:")
        st.markdown("""
        - **Gemini 2.0 Flash**: ~$0.075/1M tokens
        - **Gemini 1.5 Flash**: ~$0.35/1M tokens
        - **Gemini 1.5 Pro**: ~$3.50/1M tokens
        """)

    # Affichage du statut
    if st.session_state.processed:
        st.success("🎉 Documents prêts pour les questions avec Gemini 2.0 Flash!")
        st.info("💡 Vous pouvez maintenant poser des questions sur vos documents!")
    else:
        st.info("👈 Commencez par uploader et traiter vos documents dans la sidebar")
        
        # Guide d'installation
        with st.expander("📝 Guide de configuration Gemini API"):
            st.markdown("""
            **1. Obtenir une clé API Gemini:**
            - Allez sur [Google AI Studio](https://makersuite.google.com/app/apikey)
            - Créez un nouveau projet ou sélectionnez un existant
            - Cliquez sur "Create API Key"
            - Copiez votre clé API
            
            **2. Configurer votre environnement:**
            Créez un fichier `.env` dans votre projet avec:
            ```
            GOOGLE_API_KEY=votre_clé_api_ici
            ```
            
            **3. Installer les dépendances:**
            ```bash
            pip install langchain-google-genai google-generativeai
            ```
            
            **4. Utiliser votre RAG:**
            1. Uploadez vos PDFs
            2. Choisissez Gemini 2.0 Flash
            3. Traitez les documents  
            4. Posez vos questions !
            
            **Note:** Gemini 2.0 Flash est actuellement en preview expérimental.
            """)

if __name__ == '__main__':
    main()