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
        "model_name": "gemini-1.5-pro-latest",
        "api_key_env": "GOOGLE_API_KEY",
        "description": "Google Gemini 1.5 Pro - Modèle le plus avancé"
    },
    "Gemini Pro": {
        "model_name": "gemini-pro",
        "api_key_env": "GOOGLE_API_KEY",
        "description": "Google Gemini Pro - Modèle standard performant"
    }
}

class SegmentedVectorStoreManager:
    """Gestionnaire des vector stores FAISS avec segmentation par document"""
    
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
    
    def save_segmented_vectorstore(self, vectorstore: FAISS, vectorstore_id: str, 
                                 documents: List[LangchainDocument], embedding_model: str,
                                 chunk_params: Dict, custom_name: str = None) -> bool:
        """Sauvegarde le vector store avec segmentation par document"""
        try:
            vectorstore_path = self.get_vectorstore_path(vectorstore_id)
            
            # Supprimer le répertoire existant si nécessaire
            if vectorstore_path.exists():
                shutil.rmtree(vectorstore_path)
            
            # Créer le répertoire
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le vector store principal
            vectorstore.save_local(str(vectorstore_path))
            
            # Créer des vector stores séparés pour chaque document source
            document_segments = self._create_document_segments(documents, embedding_model)
            
            # Sauvegarder chaque segment
            segments_dir = vectorstore_path / "segments"
            segments_dir.mkdir(exist_ok=True)
            
            for doc_id, segment_data in document_segments.items():
                segment_path = segments_dir / doc_id
                segment_path.mkdir(exist_ok=True)
                segment_data['vectorstore'].save_local(str(segment_path))
            
            # Générer un nom personnalisé si non fourni
            if not custom_name:
                sources = list(set([doc.metadata.get('source', 'Unknown')[:20] for doc in documents[:3]]))
                custom_name = f"VS_{len(sources)}docs_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Créer les métadonnées avec information de segmentation
            metadata = {
                'vectorstore_id': vectorstore_id,
                'custom_name': custom_name,
                'created_at': datetime.now().isoformat(),
                'embedding_model': embedding_model,
                'chunk_params': chunk_params,
                'document_count': len(documents),
                'vector_count': vectorstore.index.ntotal,
                'has_segments': True,
                'segments': {},
                'documents_info': []
            }
            
            # Ajouter les informations des documents et segments
            doc_sources = {}
            for doc in documents:
                source = doc.metadata.get('source', 'Unknown')
                doc_type = doc.metadata.get('type', 'unknown')
                doc_id = doc.metadata.get('doc_id', f"doc_{hashlib.md5(source.encode()).hexdigest()[:8]}")
                
                if source not in doc_sources:
                    doc_sources[source] = {
                        'source': source,
                        'type': doc_type,
                        'doc_id': doc_id,
                        'chunks': 0,
                        'total_size': 0
                    }
                
                doc_sources[source]['chunks'] += 1
                doc_sources[source]['total_size'] += len(doc.page_content)
            
            metadata['documents_info'] = list(doc_sources.values())
            
            # Ajouter les informations des segments
            for doc_id, segment_data in document_segments.items():
                metadata['segments'][doc_id] = {
                    'doc_id': doc_id,
                    'source': segment_data['source'],
                    'type': segment_data['type'],
                    'chunk_count': segment_data['chunk_count'],
                    'vector_count': segment_data['vectorstore'].index.ntotal
                }
            
            # Sauvegarder les métadonnées
            self.save_vectorstore_metadata(vectorstore_id, metadata)
            
            logger.info(f"Vector store segmenté sauvegardé: {vectorstore_path} avec {len(document_segments)} segments")
            return True
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde vector store segmenté {vectorstore_id}: {str(e)}")
            return False
    
    def _create_document_segments(self, documents: List[LangchainDocument], 
                                embedding_model: str) -> Dict[str, Dict]:
        """Crée des segments de vector store pour chaque document"""
        # Regrouper les chunks par document source
        doc_groups = {}
        
        for doc in documents:
            doc_id = doc.metadata.get('doc_id')
            if not doc_id:
                # Générer un doc_id basé sur la source
                source = doc.metadata.get('source', 'Unknown')
                doc_id = f"doc_{hashlib.md5(source.encode()).hexdigest()[:8]}"
                doc.metadata['doc_id'] = doc_id
            
            if doc_id not in doc_groups:
                doc_groups[doc_id] = {
                    'documents': [],
                    'source': doc.metadata.get('source', 'Unknown'),
                    'type': doc.metadata.get('type', 'unknown')
                }
            
            doc_groups[doc_id]['documents'].append(doc)
        
        # Créer un vector store pour chaque groupe
        document_segments = {}
        embeddings = self._get_embeddings_instance(embedding_model)
        
        if embeddings:
            for doc_id, group_data in doc_groups.items():
                try:
                    # Créer le vector store pour ce document
                    segment_vectorstore = FAISS.from_documents(group_data['documents'], embeddings)
                    
                    document_segments[doc_id] = {
                        'vectorstore': segment_vectorstore,
                        'source': group_data['source'],
                        'type': group_data['type'],
                        'chunk_count': len(group_data['documents'])
                    }
                    
                except Exception as e:
                    logger.error(f"Erreur création segment pour {doc_id}: {str(e)}")
        
        return document_segments
    
    def _get_embeddings_instance(self, embedding_model: str):
        """Obtient une instance des embeddings pour la segmentation"""
        try:
            model_config = EMBEDDING_MODELS.get(embedding_model)
            if not model_config:
                return None
            
            # Vérifier si une clé API est requise
            if model_config["requires_api_key"]:
                if embedding_model == "OpenAI":
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        return None
            
            # Créer l'objet embeddings
            embedding_class = model_config["class"]
            params = model_config["params"].copy()
            
            embeddings = embedding_class(**params)
            return embeddings
            
        except Exception as e:
            logger.error(f"Erreur création embeddings pour segmentation: {str(e)}")
            return None
    
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
    
    def load_document_segment(self, vectorstore_id: str, doc_id: str, embeddings) -> Optional[FAISS]:
        """Charge un segment spécifique d'un vector store"""
        try:
            vectorstore_path = self.get_vectorstore_path(vectorstore_id)
            segment_path = vectorstore_path / "segments" / doc_id
            
            if not segment_path.exists():
                logger.warning(f"Segment {doc_id} n'existe pas dans {vectorstore_id}")
                return None
            
            # Charger le segment
            segment_vectorstore = FAISS.load_local(
                str(segment_path), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"Segment chargé: {segment_path}")
            return segment_vectorstore
            
        except Exception as e:
            logger.error(f"Erreur chargement segment {doc_id}: {str(e)}")
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
            doc_id = f"pdf_{hashlib.md5(f'{pdf_file.name}_{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    full_text += page_text + "\n\n"
                    
                    # Créer un document par page avec métadonnées
                    metadata = self.create_metadata(
                        source=pdf_file.name,
                        doc_type="pdf",
                        size=len(page_text),
                        additional_info={
                            'page_number': page_num + 1,
                            'doc_id': doc_id  # Même doc_id pour toutes les pages
                        }
                    )
                    metadata['doc_id'] = doc_id  # Forcer le même doc_id
                    
                    documents.append(LangchainDocument(
                        page_content=page_text,
                        metadata=metadata
                    ))
            
            # Enregistrer le document complet dans le registre
            if full_text.strip():
                complete_metadata = self.create_metadata(
                    source=pdf_file.name,
                    doc_type="pdf",
                    size=len(full_text),
                    additional_info={'total_pages': len(pdf_reader.pages)}
                )
                complete_metadata['doc_id'] = doc_id
                self.register_document(complete_metadata, full_text)
            
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
                doc_id = f"docx_{hashlib.md5(f'{docx_file.name}_{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
                
                metadata = self.create_metadata(
                    source=docx_file.name,
                    doc_type="docx",
                    size=len(text),
                    additional_info={'word_count': len(text.split())}
                )
                metadata['doc_id'] = doc_id
                
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
            
            doc_id = f"excel_{hashlib.md5(f'{excel_file.name}_{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
            
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
                    metadata['doc_id'] = doc_id  # Même doc_id pour toutes les feuilles
                    
                    documents.append(LangchainDocument(
                        page_content=text_content,
                        metadata=metadata
                    ))
            
            # Enregistrer le document complet dans le registre
            if documents:
                complete_text = "\n\n".join([doc.page_content for doc in documents])
                complete_metadata = self.create_metadata(
                    source=excel_file.name,
                    doc_type="excel",
                    size=len(complete_text),
                    additional_info={'total_sheets': len(df_dict)}
                )
                complete_metadata['doc_id'] = doc_id
                self.register_document(complete_metadata, complete_text)
            
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
                doc_id = f"web_{hashlib.md5(f'{url}_{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
                
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
                metadata['doc_id'] = doc_id
                
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
    
    def process_text(self, text_content: str, source_name: str = "Texte personnalisé") -> List[LangchainDocument]:
        """Traite du texte brut avec métadonnées"""
        documents = []
        try:
            if text_content.strip():
                doc_id = f"text_{hashlib.md5(f'{source_name}_{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
                
                metadata = self.create_metadata(
                    source=source_name,
                    doc_type="text",
                    size=len(text_content),
                    additional_info={'word_count': len(text_content.split())}
                )
                metadata['doc_id'] = doc_id
                
                # Enregistrer dans le registre
                self.register_document(metadata, text_content)
                
                documents.append(LangchainDocument(
                    page_content=text_content,
                    metadata=metadata
                ))
                
                self.stats['file_types']['text'] = self.stats['file_types'].get('text', 0) + 1
                logger.info(f"Texte traité avec succès: {source_name}")
                
        except Exception as e:
            error_msg = f"Erreur Texte {source_name}: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            
        return documents
    
    def is_valid_url(self, url: str) -> bool:
        """Valide une URL"""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def process_multiple_sources(self, uploaded_files, urls, text_inputs) -> List[LangchainDocument]:
        """Traite plusieurs sources de données"""
        start_time = time.time()
        all_documents = []
        
        # Traiter les fichiers uploadés
        if uploaded_files:
            for uploaded_file in uploaded_files:
                self.stats['total_docs'] += 1
                
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension == 'pdf':
                        docs = self.process_pdf(uploaded_file)
                    elif file_extension in ['docx', 'doc']:
                        docs = self.process_docx(uploaded_file)
                    elif file_extension in ['xlsx', 'xls']:
                        docs = self.process_excel(uploaded_file)
                    else:
                        # Essayer de lire comme du texte
                        content = str(uploaded_file.read(), "utf-8")
                        docs = self.process_text(content, uploaded_file.name)
                    
                    if docs:
                        all_documents.extend(docs)
                        self.stats['successful_docs'] += 1
                    else:
                        self.stats['failed_docs'] += 1
                        
                except Exception as e:
                    self.stats['failed_docs'] += 1
                    error_msg = f"Erreur fichier {uploaded_file.name}: {str(e)}"
                    self.errors.append(error_msg)
                    logger.error(error_msg)
        
        # Traiter les URLs
        if urls:
            for url in urls:
                if url.strip():
                    self.stats['total_docs'] += 1
                    
                    try:
                        docs = self.process_url(url.strip())
                        if docs:
                            all_documents.extend(docs)
                            self.stats['successful_docs'] += 1
                        else:
                            self.stats['failed_docs'] += 1
                    except Exception as e:
                        self.stats['failed_docs'] += 1
                        error_msg = f"Erreur URL {url}: {str(e)}"
                        self.errors.append(error_msg)
                        logger.error(error_msg)
        
        # Traiter les textes
        if text_inputs:
            for i, text_input in enumerate(text_inputs):
                if text_input.strip():
                    self.stats['total_docs'] += 1
                    
                    try:
                        docs = self.process_text(text_input.strip(), f"Texte_{i+1}")
                        if docs:
                            all_documents.extend(docs)
                            self.stats['successful_docs'] += 1
                        else:
                            self.stats['failed_docs'] += 1
                    except Exception as e:
                        self.stats['failed_docs'] += 1
                        error_msg = f"Erreur Texte_{i+1}: {str(e)}"
                        self.errors.append(error_msg)
                        logger.error(error_msg)
        
        # Mettre à jour les statistiques
        self.stats['processing_time'] = time.time() - start_time
        self.processed_docs = all_documents
        
        logger.info(f"Traitement terminé: {len(all_documents)} documents traités en {self.stats['processing_time']:.2f}s")
        return all_documents
    
    def get_processing_summary(self) -> Dict:
        """Retourne un résumé du traitement"""
        return {
            'stats': self.stats,
            'errors': self.errors,
            'document_count': len(self.processed_docs),
            'total_size': sum(len(doc.page_content) for doc in self.processed_docs),
            'document_registry_size': len(self.document_registry)
        }

def get_text_chunks(documents: List[LangchainDocument], chunk_size: int = 1000, 
                   chunk_overlap: int = 200) -> List[LangchainDocument]:
    """Divise les documents en chunks avec préservation des métadonnées"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunked_documents = []
        
        for doc in documents:
            # Diviser le document en chunks
            chunks = text_splitter.split_text(doc.page_content)
            
            # Créer des documents pour chaque chunk en préservant les métadonnées
            for i, chunk in enumerate(chunks):
                # Copier les métadonnées originales
                chunk_metadata = doc.metadata.copy()
                
                # Ajouter des informations sur le chunk
                chunk_metadata.update({
                    'chunk_index': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks),
                    'original_size': len(doc.page_content)
                })
                
                chunked_documents.append(LangchainDocument(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        logger.info(f"Documents divisés en {len(chunked_documents)} chunks")
        return chunked_documents
        
    except Exception as e:
        logger.error(f"Erreur division en chunks: {str(e)}")
        return documents

def get_embeddings_model(model_name: str):
    """Crée et retourne le modèle d'embeddings sélectionné"""
    try:
        model_config = EMBEDDING_MODELS.get(model_name)
        if not model_config:
            st.error(f"Modèle d'embeddings non trouvé: {model_name}")
            return None
        
        # Vérifier si une clé API est requise
        if model_config["requires_api_key"]:
            if model_name == "OpenAI":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    st.error("Clé API OpenAI requise mais non trouvée dans les variables d'environnement")
                    return None
        
        # Créer l'objet embeddings
        embedding_class = model_config["class"]
        params = model_config["params"].copy()
        
        with st.spinner(f"Initialisation du modèle {model_name}..."):
            embeddings = embedding_class(**params)
        
        logger.info(f"Modèle d'embeddings créé: {model_name}")
        return embeddings
        
    except Exception as e:
        st.error(f"Erreur création modèle d'embeddings {model_name}: {str(e)}")
        logger.error(f"Erreur embeddings {model_name}: {str(e)}")
        return None

def create_vectorstore(documents: List[LangchainDocument], embeddings, 
                      vectorstore_manager: SegmentedVectorStoreManager,
                      chunk_params: Dict, embedding_model: str,
                      custom_name: str = None) -> Tuple[Optional[FAISS], Optional[str]]:
    """Crée un vector store FAISS avec sauvegarde automatique"""
    try:
        if not documents:
            st.error("Aucun document à traiter")
            return None, None
        
        # Générer un ID unique pour ce vector store
        vectorstore_id = vectorstore_manager.generate_vectorstore_id(
            documents, embedding_model, chunk_params
        )
        
        with st.spinner(f"Création du vector store avec {len(documents)} chunks..."):
            # Créer le vector store
            vectorstore = FAISS.from_documents(documents, embeddings)
            
            # Sauvegarder automatiquement
            success = vectorstore_manager.save_segmented_vectorstore(
                vectorstore, vectorstore_id, documents, 
                embedding_model, chunk_params, custom_name
            )
            
            if success:
                st.success(f"✅ Vector store créé et sauvegardé avec succès!")
                st.info(f"📊 {len(documents)} chunks indexés | ID: {vectorstore_id}")
                logger.info(f"Vector store créé: {vectorstore_id} avec {len(documents)} documents")
                return vectorstore, vectorstore_id
            else:
                st.error("❌ Erreur lors de la sauvegarde du vector store")
                return vectorstore, None
        
    except Exception as e:
        st.error(f"❌ Erreur création vector store: {str(e)}")
        logger.error(f"Erreur création vector store: {str(e)}")
        return None, None

def get_llm_model(model_name: str):
    """Crée et retourne le modèle LLM sélectionné"""
    try:
        model_config = LLM_MODELS.get(model_name)
        if not model_config:
            st.error(f"Modèle LLM non trouvé: {model_name}")
            return None
        
        # Vérifier la clé API
        api_key = os.getenv(model_config["api_key_env"])
        if not api_key:
            st.error(f"Clé API {model_config['api_key_env']} requise mais non trouvée")
            return None
        
        # Créer le modèle
        llm = ChatGoogleGenerativeAI(
            model=model_config["model_name"],
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        logger.info(f"Modèle LLM créé: {model_name}")
        return llm
        
    except Exception as e:
        st.error(f"Erreur création modèle LLM {model_name}: {str(e)}")
        logger.error(f"Erreur LLM {model_name}: {str(e)}")
        return None

def get_conversation_chain(vectorstore: FAISS, llm):
    """Crée la chaîne de conversation avec mémoire"""
    try:
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        logger.info("Chaîne de conversation créée")
        return conversation_chain
        
    except Exception as e:
        st.error(f"Erreur création chaîne de conversation: {str(e)}")
        logger.error(f"Erreur conversation chain: {str(e)}")
        return None

def handle_user_input(user_question: str, conversation_chain):
    """Traite la question de l'utilisateur et affiche la réponse"""
    try:
        if not user_question.strip():
            return
        
        with st.spinner("🤔 Recherche d'informations..."):
            start_time = time.time()
            
            # Obtenir la réponse
            response = conversation_chain({'question': user_question})
            
            processing_time = time.time() - start_time
            
            # Afficher la réponse
            st.write(bot_template.replace("{{MSG}}", response['answer']), unsafe_allow_html=True)
            
            # Afficher les sources si disponibles
            if response.get('source_documents'):
                with st.expander("📚 Sources utilisées", expanded=False):
                    for i, doc in enumerate(response['source_documents']):
                        st.write(f"**Source {i+1}:**")
                        
                        # Afficher les métadonnées
                        metadata = doc.metadata
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"📄 **Fichier:** {metadata.get('source', 'Unknown')}")
                            st.write(f"🔖 **Type:** {metadata.get('type', 'unknown')}")
                            if metadata.get('page_number'):
                                st.write(f"📖 **Page:** {metadata['page_number']}")
                        
                        with col2:
                            st.write(f"📏 **Taille:** {metadata.get('size', 0)} caractères")
                            if metadata.get('chunk_index') is not None:
                                st.write(f"🧩 **Chunk:** {metadata['chunk_index'] + 1}/{metadata.get('total_chunks', 1)}")
                        
                        # Afficher un extrait du contenu
                        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        st.text_area(f"Extrait {i+1}:", content_preview, height=100, key=f"source_{i}")
                        st.divider()
            
            # Afficher les statistiques
            st.caption(f"⏱️ Temps de traitement: {processing_time:.2f}s")
            
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement: {str(e)}")
        logger.error(f"Erreur traitement question: {str(e)}")

def display_vectorstore_info(vectorstore_manager: SegmentedVectorStoreManager):
    """Affiche les informations sur les vector stores sauvegardés"""
    st.header("📚 Vector Stores Sauvegardés")
    
    # Rafraîchir la liste des vector stores
    vectorstores = vectorstore_manager.list_vectorstores()
    
    if not vectorstores:
        st.info("Aucun vector store sauvegardé trouvé.")
        return
    
    # Statistiques générales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Vector Stores", len(vectorstores))
    with col2:
        total_docs = sum(vs.get('document_count', 0) for vs in vectorstores)
        st.metric("Documents Total", total_docs)
    with col3:
        total_size = sum(vs.get('storage_size_mb', 0) for vs in vectorstores)
        st.metric("Taille Totale", f"{total_size:.1f} MB")
    
    st.divider()
    
    # Afficher chaque vector store
    for vs in vectorstores:
        with st.expander(f"📚 {vs.get('custom_name', vs['vectorstore_id'])}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**ID:** `{vs['vectorstore_id']}`")
                st.write(f"**Créé le:** {vs.get('created_at', 'Unknown')[:19]}")
                st.write(f"**Documents:** {vs.get('document_count', 0)}")
                st.write(f"**Modèle:** {vs.get('embedding_model', 'Unknown')}")
            
            with col2:
                st.write(f"**Taille:** {vs.get('storage_size_mb', 0):.2f} MB")
                st.write(f"**Vecteurs:** {vs.get('vector_count', 0):,}")
                chunk_params = vs.get('chunk_params', {})
                st.write(f"**Chunk Size:** {chunk_params.get('chunk_size', 'N/A')}")
            
            # Boutons d'action
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Charger", key=f"load_{vs['vectorstore_id']}"):
                    try:
                        # Obtenir le modèle d'embedding approprié
                        embedding_model = vs.get('embedding_model', 'HuggingFace Sentence Transformers')
                        embeddings = get_embeddings_model(embedding_model)
                        
                        if embeddings:
                            with st.spinner("Chargement du vector store..."):
                                vectorstore = vectorstore_manager.load_vectorstore(
                                    vs['vectorstore_id'], 
                                    embeddings
                                )
                                if vectorstore:
                                    # Mettre à jour la session
                                    st.session_state.vectorstore = vectorstore
                                    st.session_state.vectorstore_id = vs['vectorstore_id']
                                    st.session_state.current_embedding_model = embedding_model
                                    
                                    # Créer une nouvelle conversation
                                    llm = get_llm_model(list(LLM_MODELS.keys())[0])
                                    if llm:
                                        st.session_state.conversation = get_conversation_chain(vectorstore, llm)
                                    
                                    st.success(f"✅ Vector store chargé : {vs.get('custom_name', vs['vectorstore_id'])}")
                                    time.sleep(1)  # Petit délai pour l'affichage
                                    st.rerun()  # Recharger la page
                    except Exception as e:
                        st.error(f"Erreur lors du chargement : {str(e)}")
            
            with col2:
                if st.button("🗑️ Supprimer", key=f"delete_{vs['vectorstore_id']}"):
                    if vectorstore_manager.delete_vectorstore(vs['vectorstore_id']):
                        st.success("Vector store supprimé!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Erreur lors de la suppression")
            
            with col3:
                if st.button("ℹ️ Détails", key=f"details_{vs['vectorstore_id']}"):
                    st.json(vs)

def main():
    """Fonction principale de l'application"""
    load_dotenv()
    
    st.set_page_config(
        page_title="RAG Documentaire Avancé",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.write(css, unsafe_allow_html=True)
    
    # Titre principal
    st.title("📚 Système RAG Documentaire Avancé")
    st.markdown("---")
    
    # Initialiser les objets
    if 'vectorstore_manager' not in st.session_state:
        st.session_state.vectorstore_manager = SegmentedVectorStoreManager()
    
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Section Modèles
        st.subheader("🤖 Modèles")
        
        # Sélection du modèle d'embeddings
        embedding_model = st.selectbox(
            "Modèle d'Embeddings:",
            options=list(EMBEDDING_MODELS.keys()),
            index=2,  # Par défaut HuggingFace Sentence Transformers
            help="Choisissez le modèle pour créer les embeddings des documents"
        )
        
        # Sélection du modèle LLM
        llm_model = st.selectbox(
            "Modèle LLM:",
            options=list(LLM_MODELS.keys()),
            index=0,  # Par défaut Gemini 2.0 Flash
            help="Choisissez le modèle de langage pour les réponses"
        )
        
        st.divider()
        
        # Section Paramètres de chunking
        st.subheader("✂️ Paramètres de Chunking")
        
        chunk_size = st.slider(
            "Taille des chunks:",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Nombre de caractères par chunk"
        )
        
        chunk_overlap = st.slider(
            "Chevauchement:",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Nombre de caractères de chevauchement entre chunks"
        )
        
        chunk_params = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
        
        st.divider()
        
        # Section Vector Stores dans la sidebar
        st.subheader("🗂️ Vector Stores")
        
        # Lister les vector stores existants
        vectorstores = st.session_state.vectorstore_manager.list_vectorstores()
        
        if vectorstores:
            st.write(f"**{len(vectorstores)} vector stores disponibles**")
            
            for vs in vectorstores:
                if st.button(
                    f"📚 {vs.get('custom_name', vs['vectorstore_id'])}",
                    key=f"sidebar_vs_{vs['vectorstore_id']}",
                    use_container_width=True
                ):
                    # Charger le vector store sélectionné
                    embeddings = get_embeddings_model(embedding_model)
                    if embeddings:
                        with st.spinner("Chargement du vector store..."):
                            vectorstore = st.session_state.vectorstore_manager.load_vectorstore(
                                vs['vectorstore_id'],
                                embeddings
                            )
                            if vectorstore:
                                st.session_state.vectorstore = vectorstore
                                st.session_state.vectorstore_id = vs['vectorstore_id']
                                st.session_state.current_embedding_model = embedding_model
                                st.success(f"✅ Vector store chargé: {vs.get('custom_name', vs['vectorstore_id'])}")
                                st.rerun()
                            else:
                                st.error("❌ Erreur lors du chargement")
        else:
            st.info("Aucun vector store sauvegardé")
        
        if st.button("🧹 Nettoyer Anciens", use_container_width=True):
            with st.spinner("Nettoyage en cours..."):
                st.session_state.vectorstore_manager.cleanup_old_vectorstores()
                st.success("✅ Nettoyage effectué!")
                st.rerun()
    
    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["📄 Traitement Documents", "💬 Chat", "📊 Vector Stores"])
    
    with tab1:
        st.header("📄 Traitement des Documents")
        
        # Section upload de fichiers
        st.subheader("📁 Upload de Fichiers")
        uploaded_files = st.file_uploader(
            "Choisissez vos fichiers:",
            type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt'],
            accept_multiple_files=True,
            help="Formats supportés: PDF, DOCX, XLSX, TXT"
        )
        
        # Section URLs
        st.subheader("🌐 URLs Web")
        url_input = st.text_area(
            "URLs (une par ligne):",
            height=100,
            placeholder="https://example.com\nhttps://another-site.com"
        )
        
        # Section texte personnalisé
        st.subheader("✍️ Texte Personnalisé")
        text_input = st.text_area(
            "Votre texte:",
            height=200,
            placeholder="Collez ici votre texte personnalisé..."
        )
        
        # Nom personnalisé pour le vector store
        custom_name = st.text_input(
            "Nom du Vector Store (optionnel):",
            placeholder="Mon_Projet_RAG_2024"
        )
        
        # Bouton de traitement
        if st.button("🚀 Traiter et Créer Vector Store", type="primary", use_container_width=True):
            # Préparer les données
            urls = [url.strip() for url in url_input.split('\n') if url.strip()] if url_input else []
            texts = [text_input] if text_input.strip() else []
            
            if not uploaded_files and not urls and not texts:
                st.error("❌ Veuillez fournir au moins un document, une URL ou du texte.")
                return
            
            # Réinitialiser le processor
            st.session_state.document_processor.reset()
            
            # Traiter les documents
            with st.spinner("📖 Traitement des documents..."):
                documents = st.session_state.document_processor.process_multiple_sources(
                    uploaded_files, urls, texts
                )
            
            if not documents:
                st.error("❌ Aucun document n'a pu être traité.")
                return
            
            # Afficher le résumé du traitement
            summary = st.session_state.document_processor.get_processing_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Documents Traités", summary['stats']['successful_docs'])
            with col2:
                st.metric("Erreurs", summary['stats']['failed_docs'])
            with col3:
                st.metric("Taille Totale", f"{summary['total_size']:,} car.")
            with col4:
                st.metric("Temps", f"{summary['stats']['processing_time']:.1f}s")
            
            # Afficher les erreurs s'il y en a
            if summary['errors']:
                with st.expander("⚠️ Erreurs de traitement", expanded=False):
                    for error in summary['errors']:
                        st.error(error)
            
            # Diviser en chunks
            with st.spinner("✂️ Division en chunks..."):
                chunked_documents = get_text_chunks(documents, chunk_size, chunk_overlap)
            
            st.success(f"✅ {len(chunked_documents)} chunks créés")
            
            # Créer les embeddings
            embeddings = get_embeddings_model(embedding_model)
            if not embeddings:
                return
            
            # Créer le vector store
            vectorstore, vectorstore_id = create_vectorstore(
                chunked_documents, embeddings, 
                st.session_state.vectorstore_manager,
                chunk_params, embedding_model, custom_name
            )
            
            if vectorstore and vectorstore_id:
                st.session_state.vectorstore = vectorstore
                st.session_state.vectorstore_id = vectorstore_id
                st.session_state.current_embedding_model = embedding_model
                
                # Afficher les détails du vector store créé
                with st.expander("📊 Détails du Vector Store", expanded=True):
                    metadata = st.session_state.vectorstore_manager.load_vectorstore_metadata(vectorstore_id)
                    if metadata:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**ID:** `{vectorstore_id}`")
                            st.write(f"**Nom:** {metadata.get('custom_name', 'N/A')}")
                            st.write(f"**Documents:** {metadata.get('document_count', 0)}")
                            st.write(f"**Vecteurs:** {metadata.get('vector_count', 0):,}")
                        with col2:
                            st.write(f"**Modèle:** {metadata.get('embedding_model', 'N/A')}")
                            st.write(f"**Segments:** {'Oui' if metadata.get('has_segments') else 'Non'}")
                            st.write(f"**Chunk Size:** {chunk_params['chunk_size']}")
                            st.write(f"**Chunk Overlap:** {chunk_params['chunk_overlap']}")
                
                st.balloons()
    
    with tab2:
        st.header("💬 Chat avec vos Documents")
        
        # Vérifier si un vector store est disponible
        if 'vectorstore' not in st.session_state and 'selected_vectorstore' not in st.session_state:
            st.info("👆 Veuillez d'abord traiter des documents dans l'onglet 'Traitement Documents' ou charger un vector store existant.")
            return
        
        # Charger un vector store sélectionné si nécessaire
        if 'selected_vectorstore' in st.session_state and 'vectorstore' not in st.session_state:
            vectorstore_id = st.session_state['selected_vectorstore']
            embeddings = get_embeddings_model(embedding_model)
            
            if embeddings:
                with st.spinner("🔄 Chargement du vector store..."):
                    vectorstore = st.session_state.vectorstore_manager.load_vectorstore(
                        vectorstore_id, embeddings
                    )
                
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.vectorstore_id = vectorstore_id
                    st.session_state.current_embedding_model = embedding_model
                    st.success(f"✅ Vector store chargé: {vectorstore_id}")
                else:
                    st.error("❌ Erreur lors du chargement du vector store")
                    return
        
        # Créer la chaîne de conversation
        if 'conversation' not in st.session_state or st.session_state.get('current_embedding_model') != embedding_model:
            llm = get_llm_model(llm_model)
            if llm and 'vectorstore' in st.session_state:
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, llm)
                st.session_state.current_llm_model = llm_model
        
        # Interface de chat
        if 'conversation' in st.session_state:
            # Afficher les informations du vector store actuel
            if st.session_state.get('vectorstore_id'):
                metadata = st.session_state.vectorstore_manager.load_vectorstore_metadata(
                    st.session_state.vectorstore_id
                )
                if metadata:
                    st.info(f"📚 Vector Store actuel: **{metadata.get('custom_name', st.session_state.vectorstore_id)}** "
                           f"({metadata.get('document_count', 0)} documents, {metadata.get('vector_count', 0):,} vecteurs)")
            
            # Zone de saisie de question
            user_question = st.text_input(
                "🤔 Posez votre question:",
                placeholder="Que souhaitez-vous savoir à propos de vos documents ?",
                key="question_input"
            )
            
            if user_question:
                handle_user_input(user_question, st.session_state.conversation)
            
            # Historique des conversations
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                st.divider()
                st.subheader("📝 Historique des conversations")
                
                for i, entry in enumerate(reversed(st.session_state.chat_history)):
                    # Question de l'utilisateur
                    st.write(user_template.replace("{{MSG}}", entry["question"]), unsafe_allow_html=True)
                    
                    # Réponse du bot
                    st.write(bot_template.replace("{{MSG}}", entry["answer"]), unsafe_allow_html=True)
                    
                    # Sources utilisées
                    if entry.get("source_documents"):
                        with st.expander(f"📚 Sources de la réponse {i+1}", expanded=False):
                            for j, doc in enumerate(entry["source_documents"]):
                                st.write(f"**Source {j+1}:** {doc.metadata.get('source', 'Unknown')}")
                                # Afficher un aperçu du contenu
                                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                st.text_area(
                                    f"Extrait {j+1}:",
                                    preview,
                                    height=100,
                                    key=f"history_source_{i}_{j}",
                                    disabled=True
                                )
                    st.divider()
    
    with tab3:
        st.header("📊 Gestion des Vector Stores")
        
        # Afficher les vector stores existants
        display_vectorstore_info(st.session_state.vectorstore_manager)
        
        # Actions globales pour les vector stores
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Nettoyer les anciens vector stores", use_container_width=True):
                with st.spinner("Nettoyage en cours..."):
                    st.session_state.vectorstore_manager.cleanup_old_vectorstores()
                st.success("✅ Nettoyage terminé")
                st.rerun()
        
        with col2:
            if st.button("🔄 Rafraîchir la liste", use_container_width=True):
                st.rerun()
        
        # Informations système
        st.divider()
        st.subheader("ℹ️ Informations système")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Variables d'environnement:**")
            env_vars = {
                "GOOGLE_API_KEY": "✅" if os.getenv("GOOGLE_API_KEY") else "❌",
                "OPENAI_API_KEY": "✅" if os.getenv("OPENAI_API_KEY") else "⚠️ (Optionnel)"
            }
            for var, status in env_vars.items():
                st.write(f"- {var}: {status}")
        
        with col2:
            st.write("**Modèles disponibles:**")
            st.write(f"- Embeddings: {len(EMBEDDING_MODELS)} modèles")
            st.write(f"- LLM: {len(LLM_MODELS)} modèles")
            
        # Logs et débogage
        with st.expander("🔍 Logs et débogage", expanded=False):
            st.write("**Derniers logs:**")
            # Afficher les derniers logs si disponibles
            if hasattr(st.session_state, "logs"):
                for log in st.session_state.logs[-10:]:
                    st.text(log)
            else:
                st.info("Aucun log disponible")

if __name__ == "__main__":
    main()