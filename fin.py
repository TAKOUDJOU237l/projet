import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
import urllib.parse
import pandas as pd
import openpyxl
import xlrd
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document as LangchainDocument
from htmlTemplates import css, bot_template, user_template

import os
import warnings
import hashlib
import json
import shutil
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supprimer les avertissements
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Répertoires de stockage
VECTORSTORE_BASE_DIR = "vectorstores"
CACHE_DIR = "cache"

# Configuration des modèles
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
    "HuggingFace BGE Small": {
        "class": HuggingFaceEmbeddings,
        "params": {
            "model_name": "BAAI/bge-small-en-v1.5",
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
    },
    "Gemini Pro": {
        "model_name": "gemini-pro",
        "api_key_env": "GOOGLE_API_KEY",
        "description": "Google Gemini Pro - Modèle standard performant"
    }
}

class VectorStoreManager:
    """Gestionnaire des vector stores FAISS avec sauvegarde en répertoire"""
    
    def __init__(self, base_dir: str = VECTORSTORE_BASE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        
    def generate_vectorstore_id(self, documents: List[LangchainDocument], 
                              embedding_model: str, chunk_params: Dict) -> str:
        """Génère un ID unique pour le vector store basé sur le contenu et les paramètres"""
        content_hash = hashlib.md5()
        
        # Hash du contenu des documents
        for doc in documents:
            content_hash.update(doc.page_content.encode('utf-8'))
            # Inclure les métadonnées principales
            if 'source' in doc.metadata:
                content_hash.update(doc.metadata['source'].encode('utf-8'))
        
        # Hash des paramètres
        params_str = f"{embedding_model}_{chunk_params['chunk_size']}_{chunk_params['chunk_overlap']}"
        content_hash.update(params_str.encode('utf-8'))
        
        # Ajouter timestamp pour unicité
        timestamp = str(int(time.time()))
        content_hash.update(timestamp.encode('utf-8'))
        
        return content_hash.hexdigest()[:16]
    
    def get_vectorstore_path(self, vectorstore_id: str) -> Path:
        """Retourne le chemin du répertoire du vector store"""
        return self.base_dir / f"faiss_db_{vectorstore_id}"
    
    def vectorstore_exists(self, vectorstore_id: str) -> bool:
        """Vérifie si un vector store existe déjà"""
        vectorstore_path = self.get_vectorstore_path(vectorstore_id)
        return vectorstore_path.exists() and (vectorstore_path / "index.faiss").exists()
    
    def save_vectorstore_metadata(self, vectorstore_id: str, metadata: Dict):
        """Sauvegarde les métadonnées du vector store"""
        vectorstore_path = self.get_vectorstore_path(vectorstore_id)
        metadata_file = vectorstore_path / "metadata.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, indent=2, ensure_ascii=False, fp=f)
    
    def load_vectorstore_metadata(self, vectorstore_id: str) -> Optional[Dict]:
        """Charge les métadonnées du vector store"""
        vectorstore_path = self.get_vectorstore_path(vectorstore_id)
        metadata_file = vectorstore_path / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def save_vectorstore(self, vectorstore: FAISS, vectorstore_id: str, 
                        documents: List[LangchainDocument], embedding_model: str,
                        chunk_params: Dict, custom_name: str = None) -> bool:
        """Sauvegarde le vector store en répertoire avec métadonnées"""
        try:
            vectorstore_path = self.get_vectorstore_path(vectorstore_id)
            
            # Supprimer le répertoire existant si nécessaire
            if vectorstore_path.exists():
                shutil.rmtree(vectorstore_path)
            
            # Créer le répertoire
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le vector store
            vectorstore.save_local(str(vectorstore_path))
            
            # Générer un nom personnalisé si non fourni
            if not custom_name:
                sources = list(set([doc.metadata.get('source', 'Unknown')[:20] for doc in documents[:3]]))
                custom_name = f"VS_{len(sources)}docs_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Créer les métadonnées
            metadata = {
                'vectorstore_id': vectorstore_id,
                'custom_name': custom_name,
                'created_at': datetime.now().isoformat(),
                'embedding_model': embedding_model,
                'chunk_params': chunk_params,
                'document_count': len(documents),
                'vector_count': vectorstore.index.ntotal,
                'documents_info': []
            }
            
            # Ajouter les informations des documents
            doc_sources = {}
            for doc in documents:
                source = doc.metadata.get('source', 'Unknown')
                doc_type = doc.metadata.get('type', 'unknown')
                
                if source not in doc_sources:
                    doc_sources[source] = {
                        'source': source,
                        'type': doc_type,
                        'chunks': 0,
                        'total_size': 0
                    }
                
                doc_sources[source]['chunks'] += 1
                doc_sources[source]['total_size'] += len(doc.page_content)
            
            metadata['documents_info'] = list(doc_sources.values())
            
            # Sauvegarder les métadonnées
            self.save_vectorstore_metadata(vectorstore_id, metadata)
            
            logger.info(f"Vector store sauvegardé: {vectorstore_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde vector store {vectorstore_id}: {str(e)}")
            return False
    
    def load_vectorstore(self, vectorstore_id: str, embeddings) -> Optional[FAISS]:
        """Charge un vector store depuis le répertoire"""
        try:
            vectorstore_path = self.get_vectorstore_path(vectorstore_id)
            
            if not self.vectorstore_exists(vectorstore_id):
                logger.warning(f"Vector store {vectorstore_id} n'existe pas")
                return None
            
            # Charger le vector store
            vectorstore = FAISS.load_local(
                str(vectorstore_path), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"Vector store chargé: {vectorstore_path}")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Erreur chargement vector store {vectorstore_id}: {str(e)}")
            return None
    
    def list_vectorstores(self) -> List[Dict]:
        """Liste tous les vector stores disponibles avec leurs métadonnées"""
        vectorstores = []
        
        for path in self.base_dir.iterdir():
            if path.is_dir() and path.name.startswith("faiss_db_"):
                vectorstore_id = path.name.replace("faiss_db_", "")
                metadata = self.load_vectorstore_metadata(vectorstore_id)
                
                if metadata:
                    # Ajouter des informations sur la taille
                    total_size = sum(path.stat().st_size for path in path.rglob('*') if path.is_file())
                    metadata['storage_size_mb'] = round(total_size / (1024 * 1024), 2)
                    metadata['path'] = str(path)
                    vectorstores.append(metadata)
        
        # Trier par date de création (plus récent d'abord)
        vectorstores.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return vectorstores
    
    def delete_vectorstore(self, vectorstore_id: str) -> bool:
        """Supprime un vector store"""
        try:
            vectorstore_path = self.get_vectorstore_path(vectorstore_id)
            if vectorstore_path.exists():
                shutil.rmtree(vectorstore_path)
                logger.info(f"Vector store supprimé: {vectorstore_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Erreur suppression vector store {vectorstore_id}: {str(e)}")
            return False
    
    def cleanup_old_vectorstores(self, max_age_days: int = 30, max_count: int = 10):
        """Nettoie les anciens vector stores"""
        vectorstores = self.list_vectorstores()
        current_time = datetime.now()
        deleted_count = 0
        
        # Supprimer les vector stores trop anciens
        for vs in vectorstores:
            created_at = datetime.fromisoformat(vs['created_at'])
            age_days = (current_time - created_at).days
            
            if age_days > max_age_days:
                self.delete_vectorstore(vs['vectorstore_id'])
                deleted_count += 1
        
        # Garder seulement les N plus récents
        if len(vectorstores) > max_count:
            for vs in vectorstores[max_count:]:
                self.delete_vectorstore(vs['vectorstore_id'])
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Nettoyage: {deleted_count} vector stores supprimés")

class DocumentProcessor:
    """Classe pour traiter différents types de documents avec métadonnées"""
    
    def __init__(self):
        self.processed_docs = []
        self.errors = []
        self.stats = {
            'total_docs': 0,
            'successful_docs': 0,
            'failed_docs': 0,
            'processing_time': 0,
            'file_types': {}
        }
        # Dictionnaire pour stocker tous les documents par leur ID unique
        self.document_registry = {}
    
    def reset(self):
        """Remet à zéro le processor pour traiter de nouveaux documents"""
        self.processed_docs = []
        self.errors = []
        self.stats = {
            'total_docs': 0,
            'successful_docs': 0,
            'failed_docs': 0,
            'processing_time': 0,
            'file_types': {}
        }
        self.document_registry = {}
    
    def create_metadata(self, source: str, doc_type: str, size: int = 0, 
                       additional_info: Dict = None) -> Dict:
        """Crée les métadonnées pour un document avec ID unique"""
        doc_id = f"{doc_type}_{hashlib.md5(f'{source}_{size}_{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"

        
        metadata = {
            'doc_id': doc_id,
            'source': source,
            'type': doc_type,
            'size': size,
            'processed_at': datetime.now().isoformat(),
            'hash': hashlib.md5(source.encode()).hexdigest()[:8]
        }
        if additional_info:
            metadata.update(additional_info)
        return metadata
    
    def register_document(self, doc_metadata: Dict, content: str):
        """Enregistre un document dans le registre global"""
        doc_id = doc_metadata['doc_id']
        self.document_registry[doc_id] = {
            'metadata': doc_metadata,
            'content': content,
            'chunks': []
        }
    
    def process_pdf(self, pdf_file) -> List[LangchainDocument]:
        """Traite un fichier PDF avec métadonnées"""
        documents = []
        try:
            pdf_reader = PdfReader(pdf_file)
            full_text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    full_text += page_text + "\n\n"
                    
                    # Créer un document par page avec métadonnées
                    metadata = self.create_metadata(
                        source=pdf_file.name,
                        doc_type="pdf",
                        size=len(page_text),
                        additional_info={'page_number': page_num + 1}
                    )
                    
                    # Enregistrer dans le registre
                    self.register_document(metadata, page_text)
                    
                    documents.append(LangchainDocument(
                        page_content=page_text,
                        metadata=metadata
                    ))
            
            self.stats['file_types']['pdf'] = self.stats['file_types'].get('pdf', 0) + 1
            logger.info(f"PDF traité avec succès: {pdf_file.name} ({len(documents)} pages)")
            
        except Exception as e:
            error_msg = f"Erreur PDF {pdf_file.name}: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            
        return documents
    
    def process_docx(self, docx_file) -> List[LangchainDocument]:
        """Traite un fichier DOCX avec métadonnées"""
        documents = []
        try:
            # Essayer d'abord avec docx2txt
            text = docx2txt.process(docx_file)
            if not text.strip():
                # Fallback avec python-docx
                doc = Document(docx_file)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            if text.strip():
                metadata = self.create_metadata(
                    source=docx_file.name,
                    doc_type="docx",
                    size=len(text),
                    additional_info={'word_count': len(text.split())}
                )
                
                # Enregistrer dans le registre
                self.register_document(metadata, text)
                
                documents.append(LangchainDocument(
                    page_content=text,
                    metadata=metadata
                ))
                
                self.stats['file_types']['docx'] = self.stats['file_types'].get('docx', 0) + 1
                logger.info(f"DOCX traité avec succès: {docx_file.name}")
            
        except Exception as e:
            error_msg = f"Erreur DOCX {docx_file.name}: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            
        return documents
    
    def process_excel(self, excel_file) -> List[LangchainDocument]:
        """Traite un fichier Excel avec métadonnées"""
        documents = []
        try:
            # Déterminer le moteur selon l'extension
            engine = 'openpyxl' if excel_file.name.endswith('.xlsx') else 'xlrd'
            df_dict = pd.read_excel(excel_file, sheet_name=None, engine=engine)
            
            for sheet_name, df in df_dict.items():
                if not df.empty:
                    # Créer un texte structuré pour chaque feuille
                    text_content = f"=== Feuille: {sheet_name} ===\n\n"
                    
                    # Ajouter les en-têtes
                    headers = list(df.columns)
                    text_content += "Colonnes: " + " | ".join(str(h) for h in headers) + "\n\n"
                    
                    # Ajouter les données
                    for index, row in df.iterrows():
                        row_data = []
                        for col in headers:
                            cell_value = row[col]
                            if pd.notna(cell_value):
                                row_data.append(f"{col}: {cell_value}")
                        
                        if row_data:
                            text_content += f"Ligne {index + 1} - " + " | ".join(row_data) + "\n"
                    
                    # Créer le document avec métadonnées
                    metadata = self.create_metadata(
                        source=excel_file.name,
                        doc_type="excel",
                        size=len(text_content),
                        additional_info={
                            'sheet_name': sheet_name,
                            'rows': len(df),
                            'columns': len(df.columns)
                        }
                    )
                    
                    # Enregistrer dans le registre
                    self.register_document(metadata, text_content)
                    
                    documents.append(LangchainDocument(
                        page_content=text_content,
                        metadata=metadata
                    ))
            
            self.stats['file_types']['excel'] = self.stats['file_types'].get('excel', 0) + 1
            logger.info(f"Excel traité avec succès: {excel_file.name} ({len(documents)} feuilles)")
            
        except Exception as e:
            error_msg = f"Erreur Excel {excel_file.name}: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            
        return documents
    
    def process_url(self, url: str) -> List[LangchainDocument]:
        """Traite une URL avec métadonnées"""
        documents = []
        try:
            if not self.is_valid_url(url):
                raise ValueError("URL invalide")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parser le contenu
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Supprimer les éléments indésirables
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            
            # Extraire le titre
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Sans titre"
            
            # Extraire le texte principal
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            if clean_text.strip():
                processing_time = time.time() - start_time
                
                metadata = self.create_metadata(
                    source=url,
                    doc_type="web",
                    size=len(clean_text),
                    additional_info={
                        'title': title_text,
                        'status_code': response.status_code,
                        'processing_time': round(processing_time, 2),
                        'content_type': response.headers.get('content-type', 'unknown')
                    }
                )
                
                # Enregistrer dans le registre
                self.register_document(metadata, clean_text)
                
                documents.append(LangchainDocument(
                    page_content=clean_text,
                    metadata=metadata
                ))
                
                self.stats['file_types']['web'] = self.stats['file_types'].get('web', 0) + 1
                logger.info(f"URL traitée avec succès: {url} ({len(clean_text)} caractères)")
                
        except Exception as e:
            error_msg = f"Erreur URL {url}: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            
        return documents
    
    def is_valid_url(self, url: str) -> bool:
        """Valide une URL"""
        try:
            parsed = urllib.parse.urlparse(url)
            return all([parsed.scheme, parsed.netloc])
        except:
            return False
    
    def process_all_sources(self, uploaded_files: List = None, urls: List[str] = None) -> List[LangchainDocument]:
        """Traite toutes les sources avec gestion d'erreurs et statistiques"""
        start_time = time.time()
        all_documents = []
        
        self.stats['total_docs'] = len(uploaded_files or []) + len(urls or [])
        
        # Traiter les fichiers uploadés
        if uploaded_files:
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        docs = self.process_pdf(file)
                    elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                                      "application/msword"]:
                        docs = self.process_docx(file)
                    elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                      "application/vnd.ms-excel"]:
                        docs = self.process_excel(file)
                    else:
                        continue
                    
                    all_documents.extend(docs)
                    if docs:
                        self.stats['successful_docs'] += 1
                    else:
                        self.stats['failed_docs'] += 1
                        
                except Exception as e:
                    self.stats['failed_docs'] += 1
                    logger.error(f"Erreur traitement fichier {file.name}: {str(e)}")
        
        # Traiter les URLs
        if urls:
            for url in urls:
                try:
                    docs = self.process_url(url)
                    all_documents.extend(docs)
                    if docs:
                        self.stats['successful_docs'] += 1
                    else:
                        self.stats['failed_docs'] += 1
                        
                except Exception as e:
                    self.stats['failed_docs'] += 1
                    logger.error(f"Erreur traitement URL {url}: {str(e)}")
        
        self.stats['processing_time'] = round(time.time() - start_time, 2)
        self.processed_docs = all_documents
        
        logger.info(f"Traitement terminé: {len(all_documents)} documents traités en {self.stats['processing_time']}s")
        
        return all_documents

class OptimizedTextSplitter:
    """Diviseur de texte optimisé avec préservation du contexte"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, processor: DocumentProcessor = None):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.processor = processor
        self.chunk_params = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
    
    def split_documents(self, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """Divise les documents en chunks en préservant les métadonnées et en enregistrant les chunks"""
        chunks = []
        
        for doc in documents:
            # Diviser le document
            text_chunks = self.splitter.split_text(doc.page_content)
            doc_id = doc.metadata.get('doc_id')
            
            # Créer des chunks avec métadonnées enrichies
            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = doc.metadata.copy()
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_index': i,
                    'total_chunks': len(text_chunks),
                    'chunk_size': len(chunk_text),
                    'parent_doc_id': doc_id
                })
                
                chunk_doc = LangchainDocument(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk_doc)
                
                # Enregistrer le chunk dans le registre du processor
                if self.processor and doc_id in self.processor.document_registry:
                    self.processor.document_registry[doc_id]['chunks'].append({
                        'chunk_id': chunk_id,
                        'content': chunk_text,
                        'metadata': chunk_metadata
                    })
        
        return chunks

class MultiVectorStoreRetriever:
    """Retriever personnalisé pour interroger plusieurs vector stores simultanément"""
    
    def __init__(self, vectorstores: Dict[str, FAISS], k: int = 5):
        self.vectorstores = vectorstores
        self.k = k
    
    def get_relevant_documents(self, query: str) -> List[LangchainDocument]:
        """Récupère les documents pertinents de tous les vector stores"""
        all_docs = []
        
        for vs_id, vectorstore in self.vectorstores.items():
            try:
                # Rechercher dans chaque vector store
                docs = vectorstore.similarity_search(query, k=self.k)
                
                # Ajouter l'ID du vector store aux métadonnées
                for doc in docs:
                    doc.metadata['vectorstore_id'] = vs_id
                
                all_docs.extend(docs)
                
            except Exception as e:
                logger.error(f"Erreur recherche dans vector store {vs_id}: {str(e)}")
        
        # Trier par pertinence (score de similarité si disponible)
        # Pour l'instant, on limite au nombre total de documents demandés
        return all_docs[:self.k * len(self.vectorstores)]
    
    def aget_relevant_documents(self, query: str):
        """Version asynchrone (non implémentée pour cet exemple)"""
        return self.get_relevant_documents(query)

