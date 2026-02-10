"""
Advanced Memory Management System (AMMS)
A production-ready hierarchical memory system for conversational AI
Based on EMMS architecture but focused on practical functionality
"""

import os
import json
import pickle
import time
import asyncio
import numpy as np
import requests
import feedparser
import networkx as nx
import hashlib
import logging
import math
import random
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import warnings
import sys
from scipy.spatial.distance import cosine
from scipy import stats
import threading
import queue

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================

class Config:
    """System configuration"""
    # Memory limits
    WORKING_MEMORY_CAPACITY = 7
    SHORT_TERM_MEMORY_CAPACITY = 50
    LONG_TERM_MEMORY_UNLIMITED = True
    
    # Consolidation parameters
    CONSOLIDATION_THRESHOLD = 0.7
    DECAY_RATE = 0.1  # per hour
    
    # Performance settings
    CHUNK_SIZE = 50
    COMPRESSION_RATIO = 0.8
    
    # Real-time integration
    UPDATE_INTERVAL = 30  # seconds
    API_TIMEOUT = 5
    
    # LLM settings
    CONTEXT_WINDOW = 32000
    TEMPERATURE = 0.7
    
    # File paths
    MEMORY_SAVE_DIR = "amms_memories"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        os.makedirs(cls.MEMORY_SAVE_DIR, exist_ok=True)

# ===================== DATA STRUCTURES =====================

@dataclass
class Experience:
    """Represents a single experience with metadata"""
    content: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    domain: str = "conversation"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.id = self._generate_id()
        
    def _generate_id(self) -> str:
        """Generate unique ID for experience"""
        return hashlib.md5(
            f"{self.content}{self.timestamp}".encode()
        ).hexdigest()[:12]

@dataclass
class MemoryItem:
    """Enhanced memory item with multi-modal features"""
    experience: Experience
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    consolidation_score: float = 0.0
    associations: List[str] = field(default_factory=list)
    features: Optional['MultiModalFeatures'] = None
    
    @property
    def id(self) -> str:
        return self.experience.id
        
    @property
    def content(self) -> str:
        return self.experience.content
        
    @property
    def timestamp(self) -> float:
        return self.experience.timestamp
        
    @property
    def importance(self) -> float:
        return self.experience.importance

@dataclass
class MultiModalFeatures:
    """Multi-modal feature representation"""
    textual: np.ndarray = field(default_factory=lambda: np.zeros(16))
    temporal: np.ndarray = field(default_factory=lambda: np.zeros(16))
    emotional: np.ndarray = field(default_factory=lambda: np.zeros(16))
    semantic: np.ndarray = field(default_factory=lambda: np.zeros(16))
    spatial: np.ndarray = field(default_factory=lambda: np.zeros(16))
    domain: np.ndarray = field(default_factory=lambda: np.zeros(16))
    
    def get_combined(self) -> np.ndarray:
        """Get combined feature vector"""
        return np.concatenate([
            self.textual, self.temporal, self.emotional,
            self.semantic, self.spatial, self.domain
        ])
        
    def calculate_similarity(self, other: 'MultiModalFeatures') -> float:
        """Calculate similarity with another feature set"""
        if not isinstance(other, MultiModalFeatures):
            return 0.0
        
        similarities = []
        for attr in ['textual', 'temporal', 'emotional', 'semantic', 'spatial', 'domain']:
            vec1 = getattr(self, attr)
            vec2 = getattr(other, attr)
            if vec1.size > 0 and vec2.size > 0:
                sim = 1 - cosine(vec1, vec2)
                similarities.append(sim)
                
        return np.mean(similarities) if similarities else 0.0

# ===================== MEMORY COMPONENTS =====================

class WorkingMemory:
    """Working memory with Miller's Law constraints"""
    
    def __init__(self, capacity: int = Config.WORKING_MEMORY_CAPACITY):
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)
        self.attention_weights: Dict[str, float] = {}
        
    def add(self, memory: MemoryItem) -> Optional[MemoryItem]:
        """Add item to working memory, returns overflow"""
        overflow = None
        if len(self.items) >= self.capacity:
            overflow = self.items.popleft()
            if overflow and overflow.id in self.attention_weights:
                del self.attention_weights[overflow.id]
                
        self.items.append(memory)
        self.attention_weights[memory.id] = 1.0
        self._update_attention()
        
        return overflow
        
    def _update_attention(self):
        """Update attention weights based on recency and access"""
        total_weight = 0.0
        for i, item in enumerate(self.items):
            # Recency weight (more recent = higher weight)
            recency_weight = (i + 1) / len(self.items)
            # Access weight
            access_weight = min(item.access_count / 10, 1.0)
            # Combined weight
            weight = recency_weight * 0.7 + access_weight * 0.3
            self.attention_weights[item.id] = weight
            total_weight += weight
            
        # Normalize weights
        if total_weight > 0:
            for item_id in self.attention_weights:
                self.attention_weights[item_id] /= total_weight
                
    def get_all(self) -> List[MemoryItem]:
        """Get all items sorted by attention weight"""
        items_with_weights = [
            (item, self.attention_weights.get(item.id, 0.0))
            for item in self.items
        ]
        items_with_weights.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in items_with_weights]
        
    def search(self, query: str) -> List[MemoryItem]:
        """Search working memory"""
        results = []
        query_lower = query.lower()
        
        for item in self.items:
            if query_lower in item.content.lower():
                item.access_count += 1
                item.last_access = time.time()
                results.append(item)
                
        self._update_attention()
        return results
        
    def clear(self):
        """Clear working memory"""
        self.items.clear()
        self.attention_weights.clear()

class ShortTermMemory:
    """Short-term memory with consolidation mechanisms"""
    
    def __init__(self, capacity: int = Config.SHORT_TERM_MEMORY_CAPACITY):
        self.capacity = capacity
        self.memories: Dict[str, MemoryItem] = {}
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)
        self.decay_rate = Config.DECAY_RATE
        self.last_consolidation = time.time()
        
    def add(self, memory: MemoryItem):
        """Add memory to short-term storage"""
        if len(self.memories) >= self.capacity:
            self._evict_weakest()
            
        self.memories[memory.id] = memory
        
        # Update temporal index
        hour_key = datetime.fromtimestamp(memory.timestamp).strftime("%Y-%m-%d-%H")
        self.temporal_index[hour_key].append(memory.id)
        
    def _evict_weakest(self):
        """Remove weakest memory"""
        if not self.memories:
            return
            
        weakest_id = min(
            self.memories.keys(),
            key=lambda k: self._calculate_strength(self.memories[k])
        )
        
        memory = self.memories[weakest_id]
        del self.memories[weakest_id]
        
        # Clean up temporal index
        hour_key = datetime.fromtimestamp(memory.timestamp).strftime("%Y-%m-%d-%H")
        if hour_key in self.temporal_index:
            self.temporal_index[hour_key].remove(weakest_id)
            if not self.temporal_index[hour_key]:
                del self.temporal_index[hour_key]
                
    def _calculate_strength(self, memory: MemoryItem) -> float:
        """Calculate memory strength for consolidation"""
        # Time decay
        age_hours = (time.time() - memory.timestamp) / 3600
        recency_score = math.exp(-self.decay_rate * age_hours)
        
        # Access frequency (logarithmic)
        access_score = math.log(1 + memory.access_count) / 5.0
        
        # Importance score
        importance_score = memory.importance
        
        # Feature richness (if available)
        feature_score = 0.5
        if memory.features:
            # Check how many modalities have non-zero features
            modality_count = sum(
                1 for attr in ['textual', 'temporal', 'emotional', 'semantic', 'spatial', 'domain']
                if np.any(getattr(memory.features, attr))
            )
            feature_score = modality_count / 6.0
            
        # Combined score
        strength = (
            recency_score * 0.3 +
            access_score * 0.2 +
            importance_score * 0.35 +
            feature_score * 0.15
        )
        
        return min(strength, 1.0)
        
    def get_consolidation_candidates(self) -> List[MemoryItem]:
        """Get memories ready for long-term storage"""
        candidates = []
        
        for memory in self.memories.values():
            strength = self._calculate_strength(memory)
            memory.consolidation_score = strength
            
            if strength >= Config.CONSOLIDATION_THRESHOLD:
                candidates.append(memory)
                
        return candidates
        
    def remove(self, memory_id: str):
        """Remove a specific memory"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            del self.memories[memory_id]
            
            # Clean up temporal index
            hour_key = datetime.fromtimestamp(memory.timestamp).strftime("%Y-%m-%d-%H")
            if hour_key in self.temporal_index and memory_id in self.temporal_index[hour_key]:
                self.temporal_index[hour_key].remove(memory_id)
                if not self.temporal_index[hour_key]:
                    del self.temporal_index[hour_key]
                    
    def search(self, query: str, limit: int = 20) -> List[MemoryItem]:
        """Search short-term memory"""
        results = []
        query_lower = query.lower()
        
        for memory in self.memories.values():
            score = 0.0
            
            # Content matching
            if query_lower in memory.content.lower():
                score += 1.0
                
            # Calculate final score with strength
            strength = self._calculate_strength(memory)
            final_score = score * strength
            
            if final_score > 0:
                memory.access_count += 1
                memory.last_access = time.time()
                results.append((memory, final_score))
                
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]

class LongTermMemory:
    """Long-term memory with semantic organization"""
    
    def __init__(self):
        self.memories: Dict[str, MemoryItem] = {}
        self.semantic_graph = nx.Graph()
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self.domain_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: Dict[str, Set[str]] = defaultdict(set)
        self.total_memories = 0
        
    def add(self, memory: MemoryItem):
        """Add memory to long-term storage"""
        self.memories[memory.id] = memory
        self.total_memories += 1
        
        # Add to semantic graph
        self.semantic_graph.add_node(memory.id, memory=memory)
        
        # Update indices
        self._update_keyword_index(memory)
        self._update_domain_index(memory)
        self._update_temporal_index(memory)
        
        # Create semantic connections
        self._create_semantic_connections(memory)
        
        logger.info(f"Added to long-term memory: {memory.id} (total: {self.total_memories})")
        
    def _update_keyword_index(self, memory: MemoryItem):
        """Update keyword index"""
        keywords = self._extract_keywords(memory.content)
        for keyword in keywords:
            self.keyword_index[keyword].add(memory.id)
            
    def _update_domain_index(self, memory: MemoryItem):
        """Update domain index"""
        domain = memory.experience.domain
        self.domain_index[domain].add(memory.id)
        
    def _update_temporal_index(self, memory: MemoryItem):
        """Update temporal index"""
        date_key = datetime.fromtimestamp(memory.timestamp).strftime("%Y-%m-%d")
        self.temporal_index[date_key].add(memory.id)
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'this', 'that', 'these', 'those'
        }
        
        words = text.lower().split()
        keywords = []
        
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            
            if word and len(word) > 2 and word not in stopwords:
                keywords.append(word)
                
        return list(set(keywords))[:10]  # Limit to 10 keywords
        
    def _create_semantic_connections(self, new_memory: MemoryItem):
        """Create connections based on semantic similarity"""
        # Find similar memories
        similar_memories = self._find_similar_memories(new_memory, limit=5)
        
        for similar_memory, similarity in similar_memories:
            if similarity > 0.3:  # Threshold for connection
                self.semantic_graph.add_edge(
                    new_memory.id,
                    similar_memory.id,
                    weight=similarity
                )
                
                # Update associations
                if similar_memory.id not in new_memory.associations:
                    new_memory.associations.append(similar_memory.id)
                if new_memory.id not in similar_memory.associations:
                    similar_memory.associations.append(new_memory.id)
                    
    def _find_similar_memories(self, memory: MemoryItem, limit: int = 10) -> List[Tuple[MemoryItem, float]]:
        """Find similar memories based on multiple factors"""
        similarities = []
        
        # Get candidate memories from indices
        candidates = set()
        
        # From keyword index
        keywords = self._extract_keywords(memory.content)
        for keyword in keywords:
            candidates.update(self.keyword_index.get(keyword, set()))
            
        # From domain index
        candidates.update(self.domain_index.get(memory.experience.domain, set()))
        
        # Calculate similarities
        for candidate_id in candidates:
            if candidate_id == memory.id or candidate_id not in self.memories:
                continue
                
            candidate = self.memories[candidate_id]
            similarity = self._calculate_similarity(memory, candidate)
            
            if similarity > 0:
                similarities.append((candidate, similarity))
                
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
        
    def _calculate_similarity(self, mem1: MemoryItem, mem2: MemoryItem) -> float:
        """Calculate similarity between two memories"""
        # Keyword overlap
        keywords1 = set(self._extract_keywords(mem1.content))
        keywords2 = set(self._extract_keywords(mem2.content))
        
        if keywords1 and keywords2:
            keyword_similarity = len(keywords1 & keywords2) / len(keywords1 | keywords2)
        else:
            keyword_similarity = 0.0
            
        # Feature similarity (if available)
        feature_similarity = 0.0
        if mem1.features and mem2.features:
            feature_similarity = mem1.features.calculate_similarity(mem2.features)
            
        # Temporal proximity (memories close in time are more similar)
        time_diff = abs(mem1.timestamp - mem2.timestamp)
        temporal_similarity = math.exp(-time_diff / (7 * 24 * 3600))  # Decay over a week
        
        # Domain similarity
        domain_similarity = 1.0 if mem1.experience.domain == mem2.experience.domain else 0.0
        
        # Weighted combination
        similarity = (
            keyword_similarity * 0.4 +
            feature_similarity * 0.3 +
            temporal_similarity * 0.2 +
            domain_similarity * 0.1
        )
        
        return similarity
        
    def search(self, query: str, limit: int = 20, domain: Optional[str] = None) -> List[MemoryItem]:
        """Advanced search across long-term memory"""
        results = {}
        query_lower = query.lower()
        
        # Keyword search
        keywords = self._extract_keywords(query)
        for keyword in keywords:
            for memory_id in self.keyword_index.get(keyword, set()):
                if memory_id in self.memories:
                    results[memory_id] = results.get(memory_id, 0) + 1
                    
        # Content search (slower but more thorough)
        for memory in self.memories.values():
            if query_lower in memory.content.lower():
                results[memory.id] = results.get(memory.id, 0) + 2
                
        # Domain filtering
        if domain:
            domain_memories = self.domain_index.get(domain, set())
            results = {k: v for k, v in results.items() if k in domain_memories}
            
        # Get memories and calculate final scores
        scored_results = []
        for memory_id, base_score in results.items():
            memory = self.memories[memory_id]
            
            # Access boost
            access_boost = min(memory.access_count / 10, 1.0)
            
            # Importance boost
            importance_boost = memory.importance
            
            # Graph centrality (well-connected memories are more important)
            centrality_boost = 0.5
            if self.semantic_graph.has_node(memory_id):
                degree = self.semantic_graph.degree(memory_id)
                centrality_boost = min(degree / 10, 1.0)
                
            # Final score
            final_score = base_score * (1 + access_boost * 0.2 + importance_boost * 0.3 + centrality_boost * 0.1)
            
            memory.access_count += 1
            memory.last_access = time.time()
            
            scored_results.append((memory, final_score))
            
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in scored_results[:limit]]
        
    def get_memory_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic graph"""
        if not self.semantic_graph:
            return {}
            
        stats = {
            'total_memories': self.total_memories,
            'graph_nodes': self.semantic_graph.number_of_nodes(),
            'graph_edges': self.semantic_graph.number_of_edges(),
            'average_connections': 0.0,
            'most_connected_memories': []
        }
        
        if self.semantic_graph.number_of_nodes() > 0:
            degrees = dict(self.semantic_graph.degree())
            stats['average_connections'] = sum(degrees.values()) / len(degrees)
            
            # Find most connected memories
            sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            for memory_id, degree in sorted_degrees[:5]:
                if memory_id in self.memories:
                    stats['most_connected_memories'].append({
                        'id': memory_id,
                        'connections': degree,
                        'content': self.memories[memory_id].content[:100] + '...'
                    })
                    
        return stats

# ===================== FEATURE EXTRACTION =====================

class FeatureExtractor:
    """Extract multi-modal features from experiences"""
    
    def __init__(self):
        self.modalities = ['textual', 'temporal', 'emotional', 'semantic', 'spatial', 'domain']
        
    def extract_features(self, experience: Experience) -> MultiModalFeatures:
        """Extract all features from an experience"""
        features = MultiModalFeatures()
        
        # Extract features for each modality
        features.textual = self._extract_textual_features(experience.content)
        features.temporal = self._extract_temporal_features(experience.timestamp)
        features.emotional = self._extract_emotional_features(experience.content)
        features.semantic = self._extract_semantic_features(experience.content)
        features.spatial = self._extract_spatial_features(experience)
        features.domain = self._extract_domain_features(experience.domain)
        
        return features
        
    def _extract_textual_features(self, text: str) -> np.ndarray:
        """Extract textual features"""
        features = np.zeros(16)
        
        # Basic text statistics
        words = text.split()
        features[0] = len(text) / 1000  # Length (normalized)
        features[1] = len(words) / 100  # Word count (normalized)
        features[2] = np.mean([len(w) for w in words]) / 10 if words else 0  # Avg word length
        
        # Punctuation features
        features[3] = text.count('?') / max(len(words), 1)
        features[4] = text.count('!') / max(len(words), 1)
        features[5] = text.count(',') / max(len(words), 1)
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        features[6] = unique_chars / 26  # Normalized by alphabet size
        
        # Capitalization
        features[7] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Digit presence
        features[8] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        
        return features
        
    def _extract_temporal_features(self, timestamp: float) -> np.ndarray:
        """Extract temporal features"""
        features = np.zeros(16)
        dt = datetime.fromtimestamp(timestamp)
        
        # Time of day features
        features[0] = dt.hour / 24  # Hour of day
        features[1] = dt.minute / 60  # Minute of hour
        features[2] = (dt.hour * 60 + dt.minute) / 1440  # Minute of day
        
        # Day features
        features[3] = dt.weekday() / 7  # Day of week
        features[4] = dt.day / 31  # Day of month
        features[5] = dt.month / 12  # Month of year
        
        # Cyclic encoding for hour
        hour_angle = 2 * np.pi * dt.hour / 24
        features[6] = np.sin(hour_angle)
        features[7] = np.cos(hour_angle)
        
        # Cyclic encoding for day of week
        day_angle = 2 * np.pi * dt.weekday() / 7
        features[8] = np.sin(day_angle)
        features[9] = np.cos(day_angle)
        
        # Weekend indicator
        features[10] = 1.0 if dt.weekday() >= 5 else 0.0
        
        return features
        
    def _extract_emotional_features(self, text: str) -> np.ndarray:
        """Extract emotional features using simple sentiment analysis"""
        features = np.zeros(16)
        
        text_lower = text.lower()
        
        # Emotion keywords (simplified - use proper sentiment analysis in production)
        emotions = {
            'positive': ['happy', 'good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'like', 'best', 'awesome'],
            'negative': ['sad', 'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'angry', 'disappointed', 'frustrating'],
            'surprise': ['wow', 'amazing', 'incredible', 'unexpected', 'surprise', 'shocked', 'astonished'],
            'fear': ['afraid', 'scared', 'fear', 'worried', 'anxious', 'nervous', 'terrified'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated'],
            'joy': ['happy', 'joy', 'excited', 'thrilled', 'delighted', 'cheerful']
        }
        
        # Count emotion indicators
        words = text_lower.split()
        word_count = max(len(words), 1)
        
        idx = 0
        for emotion, keywords in emotions.items():
            count = sum(1 for word in words if word in keywords)
            features[idx] = count / word_count
            idx += 1
            
        # Exclamation marks (arousal indicator)
        features[6] = text.count('!') / word_count
        
        # Question marks (uncertainty indicator)
        features[7] = text.count('?') / word_count
        
        # Capitalization (intensity indicator)
        features[8] = sum(1 for word in text.split() if word.isupper()) / word_count
        
        return features
        
    def _extract_semantic_features(self, text: str) -> np.ndarray:
        """Extract semantic features"""
        features = np.zeros(16)
        
        # Topic indicators (simplified)
        topics = {
            'personal': ['i', 'me', 'my', 'myself', 'mine'],
            'social': ['you', 'we', 'us', 'they', 'them'],
            'technical': ['code', 'program', 'system', 'data', 'algorithm'],
            'financial': ['price', 'money', 'cost', 'market', 'bitcoin'],
            'temporal': ['time', 'when', 'today', 'yesterday', 'tomorrow'],
            'spatial': ['where', 'here', 'there', 'location', 'place']
        }
        
        words = text.lower().split()
        word_count = max(len(words), 1)
        
        idx = 0
        for topic, keywords in topics.items():
            count = sum(1 for word in words if word in keywords)
            features[idx] = count / word_count
            idx += 1
            
        return features
        
    def _extract_spatial_features(self, experience: Experience) -> np.ndarray:
        """Extract spatial features (conceptual space)"""
        features = np.zeros(16)
        
        # Placeholder for spatial features
        # In a real implementation, this could include:
        # - Conceptual distance measures
        # - Topic clustering positions
        # - Knowledge graph embeddings
        
        return features
        
    def _extract_domain_features(self, domain: str) -> np.ndarray:
        """Extract domain-specific features"""
        features = np.zeros(16)
        
        # One-hot encoding for known domains
        domains = [
            'conversation', 'financial', 'technical', 'personal',
            'news', 'creative', 'educational', 'system'
        ]
        
        if domain in domains:
            features[domains.index(domain)] = 1.0
            
        return features

# ===================== EPISODIC BOUNDARY DETECTION =====================

class BoundaryDetector:
    """Detect episodic boundaries in experience stream"""
    
    def __init__(self, memory_system: 'MemorySystem'):
        self.memory_system = memory_system
        self.experience_buffer = deque(maxlen=50)
        self.boundaries = []
        self.current_episode_start = 0
        
    def process_experience(self, experience: Experience, memory: MemoryItem) -> bool:
        """Process experience and detect boundaries"""
        self.experience_buffer.append((experience, memory))
        
        if len(self.experience_buffer) < 3:
            return False
            
        # Calculate surprise score
        surprise = self._calculate_surprise(experience, memory)
        
        # Check for boundary
        is_boundary = surprise > 0.7  # Threshold
        
        if is_boundary:
            self.boundaries.append({
                'timestamp': experience.timestamp,
                'surprise_score': surprise,
                'episode_length': len(self.experience_buffer) - self.current_episode_start
            })
            self.current_episode_start = len(self.experience_buffer)
            
        return is_boundary
        
    def _calculate_surprise(self, experience: Experience, memory: MemoryItem) -> float:
        """Calculate surprise score for boundary detection"""
        if len(self.experience_buffer) < 2:
            return 0.0
            
        # Get recent experiences
        recent = [item[1] for item in list(self.experience_buffer)[-5:] if item[1].features]
        
        if not recent or not memory.features:
            return 0.0
            
        # Calculate average features of recent experiences
        avg_features = MultiModalFeatures()
        for modality in ['textual', 'temporal', 'emotional', 'semantic', 'spatial', 'domain']:
            modality_features = [getattr(m.features, modality) for m in recent]
            avg_feature = np.mean(modality_features, axis=0)
            setattr(avg_features, modality, avg_feature)
            
        # Calculate surprise as deviation from average
        surprise = 1.0 - memory.features.calculate_similarity(avg_features)
        
        # Domain change bonus
        recent_domains = [m.experience.domain for m in recent]
        if experience.domain not in recent_domains:
            surprise += 0.2
            
        # Temporal gap bonus
        if recent:
            time_gap = experience.timestamp - recent[-1].timestamp
            if time_gap > 3600:  # 1 hour gap
                surprise += 0.1
                
        return min(surprise, 1.0)

# ===================== IDENTITY MANAGEMENT =====================

class IdentityManager:
    """Manage persistent identity and relationships"""
    
    def __init__(self):
        self.relationships: Dict[str, Dict[str, Any]] = {}
        self.self_narrative = []
        self.interaction_patterns = defaultdict(int)
        self.preferences = {}
        self.personality_traits = {
            'helpful': 0.8,
            'analytical': 0.7,
            'curious': 0.6,
            'friendly': 0.9
        }
        
    def process_interaction(self, user_input: str, ai_response: str, timestamp: float):
        """Process an interaction and update identity"""
        # Extract relationship information
        self._extract_relationship_info(user_input)
        
        # Update interaction patterns
        self._update_interaction_patterns(user_input, ai_response)
        
        # Update self-narrative
        self._update_self_narrative(user_input, ai_response, timestamp)
        
    def _extract_relationship_info(self, text: str):
        """Extract relationship information from text"""
        text_lower = text.lower()
        
        # Name extraction
        name_patterns = [
            "my name is", "i'm", "i am", "call me", "this is"
        ]
        
        for pattern in name_patterns:
            if pattern in text_lower:
                # Simple extraction (can be improved with NLP)
                parts = text_lower.split(pattern)
                if len(parts) > 1:
                    potential_name = parts[1].strip().split()[0]
                    if potential_name and len(potential_name) > 1:
                        self.relationships['user_name'] = potential_name.capitalize()
                        logger.info(f"Extracted user name: {potential_name}")
                        
    def _update_interaction_patterns(self, user_input: str, ai_response: str):
        """Track interaction patterns"""
        # Question types
        if '?' in user_input:
            if 'what' in user_input.lower():
                self.interaction_patterns['what_questions'] += 1
            elif 'how' in user_input.lower():
                self.interaction_patterns['how_questions'] += 1
            elif 'why' in user_input.lower():
                self.interaction_patterns['why_questions'] += 1
                
        # Topic tracking
        topics = ['memory', 'bitcoin', 'price', 'help', 'name', 'time']
        for topic in topics:
            if topic in user_input.lower():
                self.interaction_patterns[f'topic_{topic}'] += 1
                
    def _update_self_narrative(self, user_input: str, ai_response: str, timestamp: float):
        """Update the self-narrative"""
        narrative_entry = {
            'timestamp': timestamp,
            'type': 'interaction',
            'summary': f"User asked about {self._summarize_input(user_input)}"
        }
        
        self.self_narrative.append(narrative_entry)
        
        # Keep narrative manageable
        if len(self.self_narrative) > 100:
            self.self_narrative = self.self_narrative[-50:]
            
    def _summarize_input(self, text: str) -> str:
        """Create a brief summary of the input"""
        if len(text) < 50:
            return text
            
        # Simple keyword-based summary
        keywords = []
        important_words = ['what', 'how', 'why', 'when', 'where', 'bitcoin', 'price', 'memory', 'name']
        
        for word in important_words:
            if word in text.lower():
                keywords.append(word)
                
        if keywords:
            return ', '.join(keywords)
        else:
            return text[:50] + '...'
            
    def get_identity_context(self) -> str:
        """Get current identity context"""
        context_parts = []
        
        # User relationships
        if 'user_name' in self.relationships:
            context_parts.append(f"User's name is {self.relationships['user_name']}")
            
        # Recent narrative
        if self.self_narrative:
            recent = self.self_narrative[-3:]
            for entry in recent:
                context_parts.append(entry['summary'])
                
        # Interaction patterns
        top_patterns = sorted(
            self.interaction_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if top_patterns:
            pattern_str = "Common topics: " + ", ".join([p[0].replace('_', ' ') for p in top_patterns])
            context_parts.append(pattern_str)
            
        return "\n".join(context_parts)
        
    def save_state(self) -> Dict[str, Any]:
        """Save identity state"""
        return {
            'relationships': self.relationships,
            'self_narrative': self.self_narrative[-50:],  # Keep last 50
            'interaction_patterns': dict(self.interaction_patterns),
            'preferences': self.preferences,
            'personality_traits': self.personality_traits
        }
        
    def load_state(self, state: Dict[str, Any]):
        """Load identity state"""
        self.relationships = state.get('relationships', {})
        self.self_narrative = state.get('self_narrative', [])
        self.interaction_patterns = defaultdict(int, state.get('interaction_patterns', {}))
        self.preferences = state.get('preferences', {})
        self.personality_traits = state.get('personality_traits', self.personality_traits)

# ===================== MEMORY SYSTEM =====================

class MemorySystem:
    """Integrated hierarchical memory system"""
    
    def __init__(self):
        # Memory components
        self.working_memory = WorkingMemory()
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        
        # Supporting components
        self.feature_extractor = FeatureExtractor()
        self.boundary_detector = BoundaryDetector(self)
        self.identity_manager = IdentityManager()
        
        # Statistics
        self.stats = {
            'total_experiences': 0,
            'consolidations': 0,
            'boundary_detections': 0,
            'search_queries': 0
        }
        
        # Configuration
        Config.ensure_directories()
        
    def process_experience(self, content: str, importance: float = 0.5,
                          domain: str = "conversation",
                          metadata: Optional[Dict] = None) -> MemoryItem:
        """Process new experience through the memory system"""
        
        # Create experience
        experience = Experience(
            content=content,
            importance=importance,
            domain=domain,
            metadata=metadata or {}
        )
        
        # Extract features
        features = self.feature_extractor.extract_features(experience)
        
        # Create memory item
        memory = MemoryItem(
            experience=experience,
            features=features
        )
        
        # Add to working memory
        overflow = self.working_memory.add(memory)
        
        # Handle overflow
        if overflow:
            self.short_term_memory.add(overflow)
            
        # Check for consolidation
        self._consolidate_memories()
        
        # Detect boundaries
        is_boundary = self.boundary_detector.process_experience(experience, memory)
        if is_boundary:
            self.stats['boundary_detections'] += 1
            
        self.stats['total_experiences'] += 1
        
        return memory
        
    def _consolidate_memories(self):
        """Consolidate memories from short-term to long-term"""
        candidates = self.short_term_memory.get_consolidation_candidates()
        
        for memory in candidates:
            self.long_term_memory.add(memory)
            self.short_term_memory.remove(memory.id)
            self.stats['consolidations'] += 1
            
            logger.info(
                f"Consolidated memory {memory.id} "
                f"(strength: {memory.consolidation_score:.2f})"
            )
            
    def search(self, query: str, limit: int = 20, search_working: bool = True) -> List[MemoryItem]:
        """Search across all memory stores"""
        self.stats['search_queries'] += 1
        
        results = []
        seen_ids = set()
        
        # Search working memory
        if search_working:
            wm_results = self.working_memory.search(query)
            for memory in wm_results:
                if memory.id not in seen_ids:
                    results.append(memory)
                    seen_ids.add(memory.id)
                    
        # Search short-term memory
        stm_results = self.short_term_memory.search(query, limit=limit)
        for memory in stm_results:
            if memory.id not in seen_ids:
                results.append(memory)
                seen_ids.add(memory.id)
                
        # Search long-term memory
        ltm_results = self.long_term_memory.search(query, limit=limit)
        for memory in ltm_results:
            if memory.id not in seen_ids:
                results.append(memory)
                seen_ids.add(memory.id)
                
        # Limit total results
        return results[:limit]
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = self.stats.copy()
        
        # Memory counts
        stats['working_memory_count'] = len(self.working_memory.items)
        stats['short_term_memory_count'] = len(self.short_term_memory.memories)
        stats['long_term_memory_count'] = self.long_term_memory.total_memories
        
        # Graph statistics
        stats['semantic_graph'] = self.long_term_memory.get_memory_graph_stats()
        
        # Identity info
        stats['relationships'] = len(self.identity_manager.relationships)
        stats['narrative_entries'] = len(self.identity_manager.self_narrative)
        
        return stats
        
    def save_to_disk(self, session_id: str):
        """Save complete memory system to disk"""
        save_path = os.path.join(Config.MEMORY_SAVE_DIR, f"session_{session_id}")
        os.makedirs(save_path, exist_ok=True)
        
        # Save each component
        components = {
            'working_memory': [vars(m.experience) for m in self.working_memory.get_all()],
            'short_term_memory': {
                k: vars(v.experience) for k, v in self.short_term_memory.memories.items()
            },
            'long_term_memory': {
                k: vars(v.experience) for k, v in self.long_term_memory.memories.items()
            },
            'identity': self.identity_manager.save_state(),
            'statistics': self.stats,
            'metadata': {
                'save_time': time.time(),
                'version': '2.0'
            }
        }
        
        # Save as pickle for efficiency
        with open(os.path.join(save_path, 'memory_state.pkl'), 'wb') as f:
            pickle.dump(components, f)
            
        # Also save summary as JSON for readability
        summary = {
            'session_id': session_id,
            'save_time': datetime.now().isoformat(),
            'statistics': self.stats,
            'memory_counts': {
                'working': len(self.working_memory.items),
                'short_term': len(self.short_term_memory.memories),
                'long_term': self.long_term_memory.total_memories
            }
        }
        
        with open(os.path.join(save_path, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved memory system to {save_path}")
        
    def load_from_disk(self, session_id: str) -> bool:
        """Load memory system from disk"""
        save_path = os.path.join(Config.MEMORY_SAVE_DIR, f"session_{session_id}")
        state_file = os.path.join(save_path, 'memory_state.pkl')
        
        if not os.path.exists(state_file):
            logger.warning(f"No saved state found for session {session_id}")
            return False
            
        try:
            with open(state_file, 'rb') as f:
                components = pickle.load(f)
                
            # Clear current state
            self.working_memory.clear()
            self.short_term_memory.memories.clear()
            self.long_term_memory = LongTermMemory()
            
            # Restore working memory
            for exp_data in components.get('working_memory', []):
                experience = Experience(**exp_data)
                features = self.feature_extractor.extract_features(experience)
                memory = MemoryItem(experience=experience, features=features)
                self.working_memory.add(memory)
                
            # Restore short-term memory
            for memory_id, exp_data in components.get('short_term_memory', {}).items():
                experience = Experience(**exp_data)
                features = self.feature_extractor.extract_features(experience)
                memory = MemoryItem(experience=experience, features=features)
                self.short_term_memory.memories[memory_id] = memory
                
            # Restore long-term memory
            for memory_id, exp_data in components.get('long_term_memory', {}).items():
                experience = Experience(**exp_data)
                features = self.feature_extractor.extract_features(experience)
                memory = MemoryItem(experience=experience, features=features)
                self.long_term_memory.add(memory)
                
            # Restore identity
            self.identity_manager.load_state(components.get('identity', {}))
            
            # Restore statistics
            self.stats = components.get('statistics', self.stats)
            
            logger.info(
                f"Loaded memory system from session {session_id} "
                f"({self.long_term_memory.total_memories} long-term memories)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading memory state: {e}")
            return False

# ===================== REAL-TIME DATA INTEGRATION =====================

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    async def fetch(self) -> Optional[Dict[str, Any]]:
        """Fetch data from source"""
        pass
        
    @abstractmethod
    def parse(self, data: Dict[str, Any]) -> Experience:
        """Parse data into experience"""
        pass

class BitcoinDataSource(DataSource):
    """Bitcoin price data source"""
    
    def __init__(self):
        self.url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        self.last_price = None
        
    async def fetch(self) -> Optional[Dict[str, Any]]:
        """Fetch Bitcoin price data"""
        try:
            response = requests.get(self.url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching Bitcoin data: {e}")
        return None
        
    def parse(self, data: Dict[str, Any]) -> Experience:
        """Parse Bitcoin data into experience"""
        price = float(data.get('price', 0))
        
        # Calculate change if we have previous price
        change_text = ""
        importance = 0.5
        
        if self.last_price:
            change = price - self.last_price
            change_pct = (change / self.last_price) * 100
            
            if abs(change_pct) > 1:
                importance = 0.8
                change_text = f" ({'+' if change > 0 else ''}{change_pct:.2f}%)"
            elif abs(change_pct) > 0.5:
                importance = 0.6
                
        self.last_price = price
        
        content = f"Bitcoin price: ${price:,.2f}{change_text}"
        
        return Experience(
            content=content,
            importance=importance,
            domain="financial",
            metadata={
                'source': 'binance',
                'symbol': 'BTCUSDT',
                'price': price
            }
        )

class NewsDataSource(DataSource):
    """RSS news feed data source"""
    
    def __init__(self, feed_url: str):
        self.feed_url = feed_url
        self.seen_urls = set()
        
    async def fetch(self) -> Optional[Dict[str, Any]]:
        """Fetch news from RSS feed"""
        try:
            feed = feedparser.parse(self.feed_url)
            if feed.entries:
                return {'entries': feed.entries[:5]}  # Top 5 stories
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
        return None
        
    def parse(self, data: Dict[str, Any]) -> Optional[Experience]:
        """Parse news data into experience"""
        entries = data.get('entries', [])
        if not entries:
            return None
            
        # Find new entry
        for entry in entries:
            if entry.link not in self.seen_urls:
                self.seen_urls.add(entry.link)
                
                content = f"News: {entry.title}"
                if hasattr(entry, 'summary'):
                    content += f" - {entry.summary[:200]}..."
                    
                return Experience(
                    content=content,
                    importance=0.6,
                    domain="news",
                    metadata={
                        'source': 'rss',
                        'url': entry.link,
                        'title': entry.title
                    }
                )
                
        return None

class RealTimeIntegrator:
    """Integrate real-time data into memory system"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system
        self.data_sources: Dict[str, DataSource] = {}
        self.running = False
        self.update_interval = Config.UPDATE_INTERVAL
        
    def add_source(self, name: str, source: DataSource):
        """Add a data source"""
        self.data_sources[name] = source
        logger.info(f"Added data source: {name}")
        
    async def update_sources(self):
        """Update all data sources"""
        for name, source in self.data_sources.items():
            try:
                # Fetch data
                data = await source.fetch()
                if data:
                    # Parse into experience
                    experience = source.parse(data)
                    
                    if isinstance(experience, Experience):
                        # Add to memory system
                        memory = self.memory_system.process_experience(
                            content=experience.content,
                            importance=experience.importance,
                            domain=experience.domain,
                            metadata=experience.metadata
                        )
                        
                        logger.info(f"Integrated data from {name}: {experience.content[:100]}")
                        
            except Exception as e:
                logger.error(f"Error updating source {name}: {e}")
                
    async def start(self):
        """Start real-time integration"""
        self.running = True
        logger.info("Started real-time data integration")
        
        while self.running:
            await self.update_sources()
            await asyncio.sleep(self.update_interval)
            
    def stop(self):
        """Stop real-time integration"""
        self.running = False
        logger.info("Stopped real-time data integration")

# ===================== LLM INTEGRATION =====================

class LLMInterface:
    """Interface for language model integration"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system
        self.model_type = "ollama"  # or "claude"
        self.context_window = Config.CONTEXT_WINDOW
        
    def prepare_context(self, user_input: str, relevant_memories: List[MemoryItem]) -> str:
        """Prepare context for LLM"""
        context_parts = []
        
        # System message
        context_parts.append(
            "You are a helpful AI assistant with an advanced memory system. "
            "You can remember past conversations and learn from interactions."
        )
        
        # Identity context
        identity_context = self.memory_system.identity_manager.get_identity_context()
        if identity_context:
            context_parts.append(f"\nContext about the user:\n{identity_context}")
            
        # Working memory
        working_memories = self.memory_system.working_memory.get_all()
        if working_memories:
            context_parts.append("\nCurrent conversation context:")
            for memory in working_memories[-5:]:  # Last 5
                context_parts.append(f"- {memory.content}")
                
        # Relevant memories
        if relevant_memories:
            context_parts.append("\nRelevant past memories:")
            for memory in relevant_memories[:10]:  # Top 10
                time_ago = self._format_time_ago(memory.timestamp)
                context_parts.append(f"- {memory.content} ({time_ago})")
                
        # Current input
        context_parts.append(f"\nUser: {user_input}")
        
        return "\n".join(context_parts)
        
    def _format_time_ago(self, timestamp: float) -> str:
        """Format timestamp as human-readable time ago"""
        seconds = time.time() - timestamp
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days > 1 else ''} ago"
            
    def generate_response(self, context: str) -> str:
        """Generate response using LLM"""
        if self.model_type == "ollama":
            return self._generate_ollama_response(context)
        else:
            # Placeholder for other LLM types
            return "Response generation not implemented for this model type."
            
    def _generate_ollama_response(self, context: str) -> str:
        """Generate response using Ollama"""
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'deepseek-r1:1.5b',
                    'prompt': context,
                    'temperature': Config.TEMPERATURE,
                    'stream': False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'Error generating response')
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return "Error generating response. Please ensure Ollama is running."

# ===================== MAIN CHAT INTERFACE =====================

class ChatInterface:
    """Interactive chat interface"""
    
    def __init__(self):
        self.memory_system = MemorySystem()
        self.llm_interface = LLMInterface(self.memory_system)
        self.real_time_integrator = RealTimeIntegrator(self.memory_system)
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.running = True
        
    def setup_data_sources(self):
        """Set up real-time data sources"""
        # Bitcoin price
        self.real_time_integrator.add_source(
            'bitcoin',
            BitcoinDataSource()
        )
        
        # News feed (example)
        self.real_time_integrator.add_source(
            'tech_news',
            NewsDataSource('https://feeds.arstechnica.com/arstechnica/index')
        )
        
    async def start_real_time_integration(self):
        """Start real-time data integration in background"""
        asyncio.create_task(self.real_time_integrator.start())
        
    def display_banner(self):
        """Display welcome banner"""
        print("\n" + "="*60)
        print("🧠 Advanced Memory Management System (AMMS)")
        print("="*60)
        print("A hierarchical memory system for conversational AI")
        print(f"Session ID: {self.session_id}")
        print("\nCommands:")
        print("  /stats    - Show memory statistics")
        print("  /search   - Search memories")
        print("  /save     - Save memory state")
        print("  /load     - Load previous session")
        print("  /clear    - Clear working memory")
        print("  /help     - Show this help")
        print("  /quit     - Exit")
        print("="*60 + "\n")
        
    def handle_command(self, command: str) -> str:
        """Handle special commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/stats':
            stats = self.memory_system.get_statistics()
            return self._format_statistics(stats)
            
        elif cmd == '/search':
            if len(parts) < 2:
                return "Usage: /search <query>"
            query = ' '.join(parts[1:])
            results = self.memory_system.search(query, limit=5)
            return self._format_search_results(results)
            
        elif cmd == '/save':
            self.memory_system.save_to_disk(self.session_id)
            return f"Memory state saved to session {self.session_id}"
            
        elif cmd == '/load':
            if len(parts) > 1:
                session_id = parts[1]
            else:
                # List available sessions
                sessions = self._list_sessions()
                if not sessions:
                    return "No saved sessions found."
                return "Available sessions:\n" + "\n".join(f"  - {s}" for s in sessions)
                
            if self.memory_system.load_from_disk(session_id):
                self.session_id = session_id
                return f"Loaded session {session_id}"
            else:
                return f"Failed to load session {session_id}"
                
        elif cmd == '/clear':
            self.memory_system.working_memory.clear()
            return "Working memory cleared"
            
        elif cmd == '/help':
            return self._get_help_text()
            
        elif cmd == '/quit':
            self.running = False
            return "Goodbye! Your memories have been preserved."
            
        else:
            return f"Unknown command: {cmd}"
            
    def _format_statistics(self, stats: Dict[str, Any]) -> str:
        """Format statistics for display"""
        lines = ["Memory System Statistics:"]
        lines.append(f"  Total experiences: {stats['total_experiences']}")
        lines.append(f"  Consolidations: {stats['consolidations']}")
        lines.append(f"  Boundary detections: {stats['boundary_detections']}")
        lines.append(f"  Search queries: {stats['search_queries']}")
        lines.append("\nMemory Distribution:")
        lines.append(f"  Working memory: {stats['working_memory_count']}")
        lines.append(f"  Short-term memory: {stats['short_term_memory_count']}")
        lines.append(f"  Long-term memory: {stats['long_term_memory_count']}")
        
        if 'semantic_graph' in stats and stats['semantic_graph']:
            graph = stats['semantic_graph']
            lines.append("\nSemantic Graph:")
            lines.append(f"  Total nodes: {graph.get('graph_nodes', 0)}")
            lines.append(f"  Total edges: {graph.get('graph_edges', 0)}")
            lines.append(f"  Average connections: {graph.get('average_connections', 0):.2f}")
            
        return "\n".join(lines)
        
    def _format_search_results(self, results: List[MemoryItem]) -> str:
        """Format search results for display"""
        if not results:
            return "No memories found."
            
        lines = [f"Found {len(results)} memories:"]
        for i, memory in enumerate(results, 1):
            time_ago = self.llm_interface._format_time_ago(memory.timestamp)
            lines.append(f"\n{i}. {memory.content}")
            lines.append(f"   [{time_ago}, importance: {memory.importance:.2f}, accessed: {memory.access_count} times]")
            
        return "\n".join(lines)
        
    def _list_sessions(self) -> List[str]:
        """List available saved sessions"""
        sessions = []
        if os.path.exists(Config.MEMORY_SAVE_DIR):
            for item in os.listdir(Config.MEMORY_SAVE_DIR):
                if item.startswith('session_'):
                    session_id = item.replace('session_', '')
                    sessions.append(session_id)
        return sorted(sessions)
        
    def _get_help_text(self) -> str:
        """Get help text"""
        return """Available Commands:
  /stats    - Display memory system statistics
  /search <query> - Search through all memories
  /save     - Save current memory state
  /load [session_id] - Load a previous session
  /clear    - Clear working memory
  /help     - Show this help message
  /quit     - Exit the system

The system automatically:
- Processes your inputs through hierarchical memory
- Consolidates important memories to long-term storage
- Integrates real-time data (Bitcoin prices, news)
- Maintains context across conversations
- Learns from interaction patterns"""
        
    def process_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        # Check for commands
        if user_input.startswith('/'):
            return self.handle_command(user_input)
            
        # Store user input in memory
        user_memory = self.memory_system.process_experience(
            content=f"User: {user_input}",
            importance=0.7,
            domain="conversation"
        )
        
        # Search for relevant memories
        relevant_memories = self.memory_system.search(user_input, limit=10)
        
        # Prepare context
        context = self.llm_interface.prepare_context(user_input, relevant_memories)
        
        # Generate response
        response = self.llm_interface.generate_response(context)
        
        # Store AI response in memory
        ai_memory = self.memory_system.process_experience(
            content=f"AI: {response}",
            importance=0.6,
            domain="conversation"
        )
        
        # Update identity
        self.memory_system.identity_manager.process_interaction(
            user_input, response, time.time()
        )
        
        # Format response with memory info
        memory_info = f"\n[{len(relevant_memories)} relevant memories found]"
        
        return response + memory_info
        
    async def run(self):
        """Run the chat interface"""
        self.display_banner()
        
        # Set up data sources
        self.setup_data_sources()
        
        # Start real-time integration
        await self.start_real_time_integration()
        
        # Check for previous sessions
        sessions = self._list_sessions()
        if sessions:
            print(f"Found {len(sessions)} previous session(s).")
            load_previous = input("Load previous session? (y/n): ").lower()
            if load_previous == 'y':
                if len(sessions) == 1:
                    self.memory_system.load_from_disk(sessions[0])
                    self.session_id = sessions[0]
                    print(f"Loaded session {sessions[0]}")
                else:
                    print("Available sessions:")
                    for s in sessions:
                        print(f"  - {s}")
                    session_id = input("Enter session ID: ")
                    if self.memory_system.load_from_disk(session_id):
                        self.session_id = session_id
                        print(f"Loaded session {session_id}")
                        
        print("\nReady! Type your message or /help for commands.\n")
        
        # Main chat loop
        while self.running:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                    
                response = self.process_input(user_input)
                print(f"\nAI: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Saving memory state...")
                self.memory_system.save_to_disk(self.session_id)
                break
                
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"Error: {e}")
                
        # Clean shutdown
        self.real_time_integrator.stop()
        self.memory_system.save_to_disk(self.session_id)
        print(f"\nMemory state saved to session {self.session_id}")
        print("Goodbye!")

# ===================== MAIN ENTRY POINT =====================

def main():
    """Main entry point"""
    # Run the chat interface
    chat = ChatInterface()
    
    # Create event loop and run
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(chat.run())
    finally:
        loop.close()

if __name__ == "__main__":
    main()