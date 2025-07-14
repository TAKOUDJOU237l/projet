import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template

# Import pour Ollama
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

import os
import warnings
import requests
import time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration de la page en premier
st.set_page_config(
    page_title="Chat avec PDFs multiples", 
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

# Configuration des modèles Ollama disponibles
OLLAMA_MODELS = {
    "Llama 3.1 8B": {
        "model_name": "llama3.1:8b",
        "description": "Rapide et performant - 8B paramètres",
        "size": "4.7 GB",
        "recommended": True
    },
    "Llama 3.1 70B": {
        "model_name": "llama3.1:70b", 
        "description": "Très performant - 70B paramètres",
        "size": "40 GB",
        "recommended": False
    },
    "Mistral 7B": {
        "model_name": "mistral:7b",
        "description": "Léger et efficace - 7B paramètres",
        "size": "4.1 GB",
        "recommended": True
    },
    "Codellama 7B": {
        "model_name": "codellama:7b",
        "description": "Spécialisé pour le code - 7B paramètres",
        "size": "3.8 GB",
        "recommended": False
    },
    "Phi-3 Mini": {
        "model_name": "phi3:mini",
        "description": "Très léger - 3.8B paramètres",
        "size": "2.3 GB",
        "recommended": True
    },
    "Gemma 7B": {
        "model_name": "gemma:7b",
        "description": "Modèle de Google - 7B paramètres",
        "size": "5.0 GB",
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

def create_custom_prompt():
    """Crée un prompt personnalisé pour forcer le LLM à rester dans le contexte"""
    system_prompt = """
    Vous êtes un assistant IA spécialisé dans l'analyse de documents. Votre rôle est de répondre aux questions UNIQUEMENT en vous basant sur le contenu fourni dans le contexte.

    RÈGLES IMPORTANTES :
    1. Répondez UNIQUEMENT aux questions dont la réponse se trouve dans le contexte fourni
    2. Si l'information n'est pas disponible dans le contexte, répondez exactement : "Je ne trouve pas cette information dans le document fourni."
    3. Ne jamais utiliser vos connaissances générales ou externes
    4. Citez toujours les parties pertinentes du document dans votre réponse
    5. Soyez précis et factuel
    6. Si la question est partiellement répondue dans le document, précisez quelle partie vous ne pouvez pas répondre

    Contexte disponible : {context}

    Question : {question}

    Réponse :
    """
    return system_prompt

def check_ollama_connection():
    """Vérifie si Ollama est en cours d'exécution"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_ollama_models():
    """Récupère la liste des modèles Ollama installés"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return [model["name"] for model in models.get("models", [])]
        return []
    except:
        return []

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
    """Crée le vector store avec métadonnées améliorées"""
    try:
        with st.spinner("Création du vector store..."):
            # Ajouter des métadonnées aux chunks
            texts_with_metadata = []
            metadatas = []
            
            for i, chunk in enumerate(text_chunks):
                texts_with_metadata.append(chunk)
                metadatas.append({
                    "chunk_id": i,
                    "chunk_length": len(chunk),
                    "source": f"chunk_{i}"
                })
            
            vectorstore = FAISS.from_texts(
                texts=texts_with_metadata,
                embedding=embeddings,
                metadatas=metadatas
            )
            
            st.success("Vector store créé avec succès!")
            return vectorstore
    except Exception as e:
        st.error(f"Erreur lors de la création du vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore, selected_llm):
    """Crée la chaîne de conversation avec prompt personnalisé"""
    try:
        # Vérification de la connexion Ollama
        if not check_ollama_connection():
            st.error("❌ Ollama n'est pas en cours d'exécution!")
            st.info("💡 Démarrez Ollama avec la commande : `ollama serve`")
            return None
        
        # Vérification que le modèle est installé
        installed_models = get_ollama_models()
        model_name = OLLAMA_MODELS[selected_llm]["model_name"]
        
        if model_name not in installed_models:
            st.error(f"❌ Le modèle {model_name} n'est pas installé!")
            st.info(f"💡 Installez-le avec : `ollama pull {model_name}`")
            return None
        
        # Configuration du modèle Ollama
        llm = ChatOllama(
            model=model_name,
            temperature=0.1,  # Très faible pour plus de déterminisme
            num_ctx=2046,     # Augmenté pour plus de contexte
            num_predict=128,  # Augmenté pour des réponses plus complètes
            repeat_penalty=1.1,
            top_k=20,
            top_p=0.8,
            base_url="http://localhost:11434",
            mirostat=1,
            mirostat_eta=0.1,
            mirostat_tau=5.0,
            num_thread=4
        )
        
        # Création du prompt personnalisé
        custom_prompt = PromptTemplate(
            template=create_custom_prompt(),
            input_variables=["context", "question"]
        )
        
        # Configuration de la mémoire
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        # Création de la chaîne avec prompt personnalisé
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 3,  # Augmenté pour plus de contexte
                    "fetch_k": 5  # Récupère plus de documents candidats
                }
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        
        st.success(f"✅ Modèle {selected_llm} configuré avec succès!")
        return conversation_chain
        
    except Exception as e:
        st.error(f"Erreur avec le modèle Ollama: {str(e)}")
        if "connection" in str(e).lower():
            st.info("💡 Assurez-vous qu'Ollama est en cours d'exécution avec `ollama serve`")
        return None

def validate_answer_relevance(answer, source_documents, threshold=0.5):
    """Valide si la réponse est pertinente par rapport au contexte"""
    
    # Mots-clés indiquant une réponse non pertinente
    irrelevant_keywords = [
        "je ne sais pas",
        "je ne trouve pas",
        "information non disponible",
        "pas mentionné dans le document",
        "selon mes connaissances générales"
    ]
    
    answer_lower = answer.lower()
    
    # Vérifier si la réponse contient des mots-clés de non-pertinence
    for keyword in irrelevant_keywords:
        if keyword in answer_lower:
            return False, "Information non disponible dans le document"
    
    # Vérifier si des sources ont été trouvées
    if not source_documents:
        return False, "Aucune source pertinente trouvée dans le document"
    
    return True, answer

def analyze_context_quality(question, retrieved_docs):
    """Analyse la qualité du contexte récupéré"""
    if not retrieved_docs:
        return 0.0
    
    question_words = set(question.lower().split())
    
    scores = []
    for doc in retrieved_docs:
        doc_words = set(doc.page_content.lower().split())
        # Calcul simple de similarité lexicale
        intersection = question_words.intersection(doc_words)
        score = len(intersection) / len(question_words) if question_words else 0
        scores.append(score)
    
    return max(scores) if scores else 0.0

def handle_userinput(user_question):
    """Gère l'input utilisateur avec validation de pertinence"""
    if st.session_state.conversation is None:
        st.warning("⚠️ Veuillez d'abord traiter vos documents avant de poser une question.")
        return
    
    try:
        with st.spinner("Génération de la réponse avec Ollama..."):
            response = st.session_state.conversation({'question': user_question})
        
        # Vérification de la pertinence de la réponse
        answer = response.get('answer', '')
        source_documents = response.get('source_documents', [])
        
        # Affichage de la question utilisateur
        st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
        
        # Affichage de la réponse
        st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
        
        # Affichage des sources si disponibles
        if source_documents and "Je ne trouve pas cette information" not in answer:
            with st.expander("📚 Sources utilisées"):
                for i, doc in enumerate(source_documents):
                    st.write(f"**Source {i+1}:**")
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.write("---")
        
        # Analyse de la qualité du contexte
        context_quality = analyze_context_quality(user_question, source_documents)
        if context_quality < 0.2:
            st.warning("⚠️ La question semble peu liée au contenu du document. Essayez d'être plus spécifique.")
                
    except Exception as e:
        st.error(f"Erreur lors de la génération de la réponse: {str(e)}")
        if "connection" in str(e).lower():
            st.info("💡 Vérifiez qu'Ollama est en cours d'exécution")
        elif "output_key" in str(e).lower():
            st.info("💡 Problème de configuration de la mémoire - redémarrez l'application")

def display_context_info():
    """Affiche des informations sur le contexte disponible"""
    if st.session_state.vectorstore is not None:
        st.sidebar.subheader("📊 Informations sur le contexte")
        
        # Nombre de documents dans le vector store
        if hasattr(st.session_state.vectorstore, 'index'):
            num_docs = st.session_state.vectorstore.index.ntotal
            st.sidebar.info(f"📄 Documents indexés: {num_docs}")
        
        # Conseils pour les questions
        st.sidebar.info("""
        💡 **Conseils pour de meilleures réponses:**
        - Posez des questions spécifiques au contenu du document
        - Utilisez des mots-clés présents dans le document
        - Évitez les questions trop générales
        - Le système ne répondra qu'avec les informations du document
        """)

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

    st.header("💬 Chat avec PDFs multiples")
    st.markdown("Uploadez vos PDFs et posez des questions sur leur contenu !")

    # Zone de question
    user_question = st.text_input("❓ Posez une question sur vos documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar
    with st.sidebar:
        st.subheader("📄 Vos documents")
        
        # Vérification du statut Ollama
        if check_ollama_connection():
            st.success("✅ Ollama est connecté")
            installed_models = get_ollama_models()
            if installed_models:
                st.info(f"📦 Modèles installés: {len(installed_models)}")
            else:
                st.warning("⚠️ Aucun modèle installé")
        else:
            st.error("❌ Ollama non connecté")
            st.info("💡 Démarrez avec: `ollama serve`")

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
        
        # Sélection du modèle Ollama
        selected_llm = st.selectbox(
            "Choisissez un modèle Ollama:",
            list(OLLAMA_MODELS.keys()),
            index=0,
            help="Llama 3.1 8B est recommandé pour un bon équilibre performance/vitesse"
        )
        
        llm_info = OLLAMA_MODELS[selected_llm]
        st.info(f"🤖 {llm_info['description']}")
        st.info(f"💾 Taille: {llm_info['size']}")
        
        if llm_info["recommended"]:
            st.success("⭐ Modèle recommandé")
        
        # Vérification si le modèle est installé
        installed_models = get_ollama_models()
        model_name = llm_info["model_name"]
        
        if model_name in installed_models:
            st.success(f"✅ {model_name} est installé")
        else:
            st.warning(f"⚠️ {model_name} n'est pas installé")
            if st.button(f"📥 Installer {model_name}"):
                st.info(f"Exécutez cette commande dans votre terminal:")
                st.code(f"ollama pull {model_name}")

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

        # Affichage des informations sur le contexte
        display_context_info()

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

        # Informations sur Ollama
        st.markdown("### 🚀 Avantages Ollama:")
        st.markdown("""
        - ✅ **100% gratuit** et local
        - ✅ **Pas de limites** d'utilisation
        - ✅ **Données privées** (tout reste sur votre machine)
        - ✅ **Fonctionne offline**
        - ✅ **Modèles open source**
        """)

        # Informations sur la contextualisation
        st.markdown("### 🎯 Contextualisation:")
        st.markdown("""
        - ✅ **Réponses strictement basées** sur vos documents
        - ✅ **Indication claire** quand l'info n'est pas disponible
        - ✅ **Sources affichées** pour chaque réponse
        - ✅ **Pas de connaissances externes** utilisées
        """)

    # Affichage du statut
    if st.session_state.processed:
        st.success("🎉 Documents prêts pour les questions!")
        st.info("💡 Vous pouvez maintenant poser des questions sur vos documents!")
        st.info("🎯 Le système répondra UNIQUEMENT avec les informations contenues dans vos documents.")
    else:
        st.info("👈 Commencez par uploader et traiter vos documents dans la sidebar")
        
        # Guide d'installation
        with st.expander("📝 Guide d'installation Ollama"):
            st.markdown("""
            **1. Installer Ollama:**
            ```bash
            # Sur macOS/Linux
            curl -fsSL https://ollama.com/install.sh | sh
            
            # Sur Windows
            # Téléchargez depuis https://ollama.com/download
            ```
            
            **2. Démarrer Ollama:**
            ```bash
            ollama serve
            ```
            
            **3. Installer un modèle (recommandé):**
            ```bash
            ollama pull llama3.1:8b
            ```
            
            **4. Utiliser votre RAG:**
            1. Uploadez vos PDFs
            2. Traitez les documents  
            3. Posez vos questions !
            
            **5. Contextualisation stricte:**
            - Le système ne répondra qu'avec les infos de vos documents
            - Si l'info n'est pas disponible, il vous le dira clairement
            - Les sources sont toujours affichées
            """)

if __name__ == '__main__':
    main()