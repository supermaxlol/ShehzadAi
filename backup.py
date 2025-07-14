"""
Complete Enhanced Persistent Identity AI with Advanced Memory Management
Neurobiologically-Inspired Architecture with EM-LLM Integration + ALL Original Components + Memory Enhancements

This implements the COMPLETE system with ALL original components PLUS memory enhancements:
1. COMPLETE original persistent identity system with personality, cortical columns, identity formation
2. ALL original EM-LLM episodic memory with advanced memory management improvements  
3. ALL original 6-layer cortical architecture with episodic integration
4. ALL original identity formation mechanisms (ContinuousNarrator, IdentityComparer, etc.)
5. ALL original LLM integration and real-time data fetching
6. ALL original validation metrics and experimental framework
7. PLUS NEW: Token-level context management, graph-theoretic refinement, hierarchical memory
8. PLUS NEW: Advanced multi-strategy retrieval, compression, and cross-modal integration
9. PLUS NEW: Complete integration coordination and enhanced performance tracking

Author: Enhanced AI Research Team  
Version: 3.0 - Complete System with ALL Original + Memory Enhancements
"""

import json
import time
import random
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Tuple, Optional, Set, Union
import sqlite3
import hashlib
import uuid
from collections import defaultdict, deque, Counter
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import networkx as nx
import requests
import feedparser
from urllib.parse import urlencode
import pickle
import gzip
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE DATA STRUCTURES (COMPLETE ORIGINAL + ENHANCED)
# ============================================================================

@dataclass
class SensorimotorExperience:
    """Enhanced experience representation with episodic context + memory features"""
    experience_id: str
    content: str
    domain: str
    sensory_features: Dict[str, Any]
    motor_actions: List[str]
    contextual_embedding: np.ndarray
    temporal_markers: List[float]
    attention_weights: Dict[str, float]
    prediction_targets: Dict[str, float]
    novelty_score: float
    timestamp: str
    
    # Episodic enhancements (ORIGINAL)
    episodic_context: Optional[Dict[str, Any]] = None
    episode_boundary_score: float = 0.0
    cross_episode_similarity: float = 0.0
    
    # NEW: Enhanced memory features
    emotional_features: Dict[str, float] = field(default_factory=dict)
    causal_indicators: List[str] = field(default_factory=list)
    goal_relevance: Dict[str, float] = field(default_factory=dict)
    modality_features: Dict[str, np.ndarray] = field(default_factory=dict)
    importance_weight: float = 0.5
    access_frequency: int = 0
    last_access: float = field(default_factory=time.time)
    memory_strength: float = 1.0

@dataclass
class AdvancedPersonalityState:
    """Enhanced personality with episodic integration (COMPLETE ORIGINAL)"""
    traits_big5: Dict[str, float]
    cognitive_style: Dict[str, float]
    core_value_system: Dict[str, float]
    narrative_themes: List[str]
    identity_anchors: List[str]
    goal_hierarchy: Dict[str, Dict[str, float]]
    emotional_patterns: Dict[str, float]
    social_preferences: Dict[str, float]
    narrative_coherence: float
    identity_stability: float
    development_stage: str
    last_updated: str
    
    # Episodic enhancements (ORIGINAL)
    episodic_narrative_depth: float = 0.0
    episodic_identity_milestones: List[str] = field(default_factory=list)
    cross_episodic_coherence: float = 0.0

@dataclass
class ReferenceFrame:
    """Enhanced reference frame with episodic integration (COMPLETE ORIGINAL)"""
    frame_id: str
    domain: str
    spatial_map: Dict[str, np.ndarray]
    conceptual_hierarchy: Dict[str, List[str]]
    temporal_sequence: List[Tuple[str, float]]
    prediction_matrix: np.ndarray
    confidence_scores: Dict[str, float]
    last_updated: str
    
    # Episodic enhancements (ORIGINAL)
    episodic_spatial_context: Dict[str, Any] = field(default_factory=dict)
    cross_episodic_predictions: Dict[str, float] = field(default_factory=dict)

@dataclass
class Enhanced6LayerCorticalColumn:
    """Enhanced 6-layer cortical column with episodic integration (COMPLETE ORIGINAL)"""
    column_id: str
    specialization: str
    
    # 6-layer architecture with episodic integration
    layer1_sensory: Dict[str, Any]      # Sensory input processing
    layer2_pattern: Dict[str, Any]      # Pattern recognition & binding
    layer3_spatial: Dict[str, Any]      # Spatial location encoding
    layer4_temporal: Dict[str, Any]     # Temporal sequence learning
    layer5_prediction: Dict[str, Any]   # Prediction generation
    layer6_motor: Dict[str, Any]        # Motor output planning
    
    # Episodic integration
    episodic_context: Dict[str, Any]    # Current episodic context
    episodic_predictions: Dict[str, Any] # Predictions based on episodes
    
    # Learning metrics
    prediction_accuracy: float
    learning_rate: float
    episodic_influence: float           # How much episodes influence processing
    
    # Reference frame
    reference_frame: Dict[str, Any]
    last_updated: str

@dataclass
class CompressedMemoryBlock:
    """Compressed memory representation (NEW)"""
    block_id: str
    original_episode_ids: List[str]
    compressed_content: bytes
    compression_metadata: Dict[str, Any]
    abstraction_levels: Dict[str, Any]
    reconstruction_fidelity: float
    compression_ratio: float
    created_timestamp: str

@dataclass
class MemoryHierarchyLevel:
    """Represents a level in the memory hierarchy (NEW)"""
    level_name: str
    capacity: int
    retention_policy: str
    consolidation_threshold: float
    access_frequency_weight: float
    importance_weight: float

class MemoryRetrievalStrategy(ABC):
    """Abstract base class for memory retrieval strategies (NEW)"""
    
    @abstractmethod
    def retrieve(self, query_experience: SensorimotorExperience, 
                memory_store: Dict, max_results: int = 20) -> List[Dict]:
        pass
    
    @abstractmethod
    def calculate_relevance_score(self, query: SensorimotorExperience, 
                                candidate: Dict) -> float:
        pass

# ============================================================================
# ENHANCED TOKEN-LEVEL CONTEXT MANAGEMENT (NEW)
# ============================================================================

class TokenLevelContextManager:
    """Advanced token-level context management integrated with episodic memory"""
    
    def __init__(self, context_window: int = 32000, eviction_ratio: float = 0.3):
        self.context_window = context_window
        self.eviction_ratio = eviction_ratio
        
        # Three-tier context system (EM-LLM style)
        self.initial_tokens = []      # Attention sinks (first 4 tokens)
        self.evicted_tokens = []      # Tokens moved to episodic storage
        self.local_context = []       # Current active context window
        
        # Token metadata
        self.token_importance_scores = {}
        self.token_access_frequency = defaultdict(int)
        self.token_embedding_cache = {}
        
        # Eviction statistics
        self.eviction_history = deque(maxlen=1000)
        self.retrieval_statistics = defaultdict(int)
        
    def process_tokens_with_memory(self, input_tokens: List[str], 
                                 experience_context: Dict = None) -> List[str]:
        """Process tokens with intelligent memory management"""
        
        # Check if we need to evict tokens
        total_tokens_needed = len(self.local_context) + len(input_tokens)
        
        if total_tokens_needed > self.context_window:
            tokens_to_evict = total_tokens_needed - self.context_window
            evicted = self.intelligent_token_eviction(tokens_to_evict)
            self.evicted_tokens.extend(evicted)
        
        # Add new tokens to local context
        self.local_context.extend(input_tokens)
        
        # Update token metadata
        self._update_token_metadata(input_tokens)
        
        # Retrieve relevant evicted tokens if needed
        if experience_context:
            relevant_tokens = self.retrieve_relevant_tokens(experience_context)
            self.local_context = self._merge_tokens(self.local_context, relevant_tokens)
        
        return self.get_full_context()
    def process_tokens(self, content: str) -> Dict[str, Any]:
        """Process tokens (simplified version of process_tokens_with_memory)"""
        try:
            tokens = content.split() if isinstance(content, str) else content
            result = self.process_tokens_with_memory(tokens, experience_context={'content': str(content)})
            return {'tokens_processed': len(tokens), 'context_tokens': len(result) if result else 0}
        except Exception as e:
            return {'error': str(e), 'tokens_processed': 0}
    def intelligent_token_eviction(self, num_tokens_to_evict: int) -> List[str]:
        """Intelligent token eviction based on importance, frequency, and recency"""
        
        if not self.local_context or num_tokens_to_evict <= 0:
            return []
        
        # Calculate eviction scores for each token
        eviction_candidates = []
        
        for i, token in enumerate(self.local_context):
            # Factors influencing eviction decision
            recency_score = (len(self.local_context) - i) / len(self.local_context)
            importance_score = self.token_importance_scores.get(token, 0.5)
            frequency_score = min(1.0, self.token_access_frequency[token] / 10.0)
            
            # Combined eviction score (lower = more likely to evict)
            eviction_score = (
                recency_score * 0.4 +
                importance_score * 0.4 +
                frequency_score * 0.2
            )
            
            eviction_candidates.append((eviction_score, i, token))
        
        # Sort by eviction score (ascending - lowest first)
        eviction_candidates.sort(key=lambda x: x[0])
        
        # Evict tokens with lowest scores
        evicted_tokens = []
        evicted_indices = []
        
        for score, idx, token in eviction_candidates[:num_tokens_to_evict]:
            evicted_tokens.append(token)
            evicted_indices.append(idx)
            
            # Record eviction
            self.eviction_history.append({
                'token': token,
                'eviction_score': score,
                'timestamp': time.time(),
                'reason': 'context_overflow'
            })
        
        # Remove evicted tokens from local context
        for idx in sorted(evicted_indices, reverse=True):
            del self.local_context[idx]
        
        logger.info(f"Evicted {len(evicted_tokens)} tokens to episodic storage")
        return evicted_tokens
    
    def retrieve_relevant_tokens(self, experience_context: Dict, 
                               max_tokens: int = 1000) -> List[str]:
        """Retrieve relevant tokens from evicted storage"""
        
        if not self.evicted_tokens:
            return []
        
        # Extract context keywords
        context_keywords = self._extract_context_keywords(experience_context)
        
        # Score evicted tokens for relevance
        relevant_tokens = []
        
        for token in self.evicted_tokens:
            relevance_score = self._calculate_token_relevance(token, context_keywords)
            
            if relevance_score > 0.6:
                relevant_tokens.append((relevance_score, token))
                self.retrieval_statistics[token] += 1
        
        # Sort by relevance and return top tokens
        relevant_tokens.sort(reverse=True)
        return [token for score, token in relevant_tokens[:max_tokens]]
    
    def _calculate_token_relevance(self, token: str, context_keywords: Set[str]) -> float:
        """Calculate relevance of a token to current context"""
        
        # Simple keyword matching
        if token.lower() in context_keywords:
            return 1.0
        
        # Semantic similarity (simplified)
        similarity_scores = []
        for keyword in context_keywords:
            if len(keyword) > 3 and len(token) > 3:
                # Simple character overlap similarity
                overlap = len(set(token.lower()) & set(keyword.lower()))
                max_len = max(len(token), len(keyword))
                similarity_scores.append(overlap / max_len)
        
        return max(similarity_scores) if similarity_scores else 0.0
    
    def _extract_context_keywords(self, experience_context: Dict) -> Set[str]:
        """Extract keywords from experience context"""
        keywords = set()
        
        if 'content' in experience_context:
            content_words = experience_context['content'].lower().split()
            keywords.update(word for word in content_words if len(word) > 3)
        
        if 'domain' in experience_context:
            keywords.add(experience_context['domain'].lower())
        
        return keywords
    
    def _merge_tokens(self, local_tokens: List[str], retrieved_tokens: List[str]) -> List[str]:
        """Merge local and retrieved tokens efficiently"""
        
        # Insert retrieved tokens at optimal positions
        merged = local_tokens.copy()
        
        # Insert most relevant tokens near the beginning (after initial tokens)
        insert_position = min(10, len(merged))
        
        for token in retrieved_tokens:
            merged.insert(insert_position, token)
            insert_position += 1
        
        # Trim if over context window
        if len(merged) > self.context_window:
            merged = merged[:self.context_window]
        
        return merged
    
    def _update_token_metadata(self, tokens: List[str]):
        """Update token importance and frequency metadata"""
        
        for token in tokens:
            # Update access frequency
            self.token_access_frequency[token] += 1
            
            # Update importance score based on frequency and characteristics
            importance = self._calculate_token_importance(token)
            self.token_importance_scores[token] = importance
    
    def _calculate_token_importance(self, token: str) -> float:
        """Calculate importance score for a token"""
        
        # Base importance factors
        length_factor = min(1.0, len(token) / 10.0)  # Longer tokens more important
        frequency_factor = min(1.0, self.token_access_frequency[token] / 5.0)
        
        # Special token types
        if token.isdigit():
            type_factor = 0.8  # Numbers moderately important
        elif token.isupper():
            type_factor = 0.9  # Acronyms/proper nouns important
        elif len(token) > 8:
            type_factor = 0.9  # Long words often important
        else:
            type_factor = 0.5  # Default importance
        
        return (length_factor + frequency_factor + type_factor) / 3.0
    
    def get_full_context(self) -> List[str]:
        """Get complete context for LLM processing"""
        return self.initial_tokens + self.local_context
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get detailed memory usage statistics"""
        return {
            'context_window_size': self.context_window,
            'local_context_tokens': len(self.local_context),
            'evicted_tokens': len(self.evicted_tokens),
            'initial_tokens': len(self.initial_tokens),
            'utilization_ratio': len(self.local_context) / self.context_window,
            'total_evictions': len(self.eviction_history),
            'unique_evicted_tokens': len(set(self.evicted_tokens)),
            'retrieval_statistics': dict(self.retrieval_statistics)
        }

# ============================================================================
# ADVANCED GRAPH-THEORETIC BOUNDARY REFINEMENT (NEW)
# ============================================================================

class AdvancedBoundaryRefiner:
    """Advanced graph-theoretic boundary refinement system"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.graph_cache = {}
        self.refinement_history = deque(maxlen=100)
        
    def refine_boundaries_with_graph_metrics(self, episodes: List[Dict]) -> Tuple[List[Dict], Dict[str, float]]:
        """Comprehensive graph-theoretic boundary refinement"""
        
        if len(episodes) < 3:
            return episodes, {'status': 'insufficient_data'}
        
        # Build similarity graph
        G = self.build_episode_similarity_graph(episodes)
        
        # Calculate current boundary quality
        current_metrics = self.calculate_boundary_metrics(G, episodes)
        
        # Optimize boundaries using multiple algorithms
        optimization_results = self.multi_algorithm_optimization(G, episodes)
        
        # Select best optimization result
        best_result = self.select_best_optimization(optimization_results, current_metrics)
        
        # Validate and apply optimization
        if best_result['improvement'] > 0.05:  # 5% improvement threshold
            optimized_episodes = best_result['episodes']
            validation_scores = best_result['metrics']
            
            # Record refinement
            self.refinement_history.append({
                'timestamp': time.time(),
                'improvement': best_result['improvement'],
                'algorithm': best_result['algorithm'],
                'episodes_count': len(episodes)
            })
            
            logger.info(f"Boundary refinement improved quality by {best_result['improvement']:.3f}")
            return optimized_episodes, validation_scores
        
        return episodes, current_metrics
    
    def build_episode_similarity_graph(self, episodes: List[Dict]) -> nx.Graph:
        """Build weighted similarity graph between episodes"""
        
        # Check cache first
        episode_ids = tuple(ep.get('episode_id', str(i)) for i, ep in enumerate(episodes))
        cache_key = hashlib.md5(str(episode_ids).encode()).hexdigest()
        
        if cache_key in self.graph_cache:
            return self.graph_cache[cache_key]
        
        G = nx.Graph()
        
        # Add nodes with episode metadata
        for i, episode in enumerate(episodes):
            G.add_node(i, episode=episode, timestamp=episode.get('timestamp', ''))
        
        # Add edges based on multi-modal similarity
        for i in range(len(episodes)):
            for j in range(i + 1, len(episodes)):
                similarity = self.calculate_multi_modal_similarity(episodes[i], episodes[j])
                
                if similarity > self.similarity_threshold:
                    G.add_edge(i, j, weight=similarity)
        
        # Cache the graph
        self.graph_cache[cache_key] = G
        return G
    
    def calculate_multi_modal_similarity(self, ep1: Dict, ep2: Dict) -> float:
        """Calculate comprehensive multi-modal similarity"""
        
        similarities = {}
        
        # Content similarity
        if 'content' in ep1 and 'content' in ep2:
            similarities['content'] = self.calculate_content_similarity(
                ep1['content'], ep2['content']
            )
        
        # Temporal proximity
        if 'timestamp' in ep1 and 'timestamp' in ep2:
            similarities['temporal'] = self.calculate_temporal_similarity(
                ep1['timestamp'], ep2['timestamp']
            )
        
        # Domain similarity
        if 'domain' in ep1 and 'domain' in ep2:
            similarities['domain'] = 1.0 if ep1['domain'] == ep2['domain'] else 0.3
        
        # Semantic similarity
        if 'representative_tokens' in ep1 and 'representative_tokens' in ep2:
            similarities['semantic'] = self.calculate_semantic_similarity(
                ep1['representative_tokens'], ep2['representative_tokens']
            )
        
        # Embedding similarity
        if 'embedding_vector' in ep1 and 'embedding_vector' in ep2:
            try:
                similarities['embedding'] = 1 - cosine(ep1['embedding_vector'], ep2['embedding_vector'])
            except:
                similarities['embedding'] = 0.5
        
        # Novelty similarity
        if 'novelty_score' in ep1 and 'novelty_score' in ep2:
            novelty_diff = abs(ep1['novelty_score'] - ep2['novelty_score'])
            similarities['novelty'] = 1.0 - novelty_diff
        
        # Weighted combination
        weights = {
            'content': 0.25,
            'temporal': 0.15,
            'domain': 0.15,
            'semantic': 0.20,
            'embedding': 0.20,
            'novelty': 0.05
        }
        
        weighted_sum = sum(similarities.get(key, 0.5) * weight 
                          for key, weight in weights.items())
        
        return weighted_sum
    
    def calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity using multiple methods"""
        
        # Tokenize
        tokens1 = set(content1.lower().split())
        tokens2 = set(content2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Length similarity
        len1, len2 = len(content1), len(content2)
        length_sim = 1.0 - abs(len1 - len2) / max(len1, len2)
        
        # Combined similarity
        return (jaccard * 0.7 + length_sim * 0.3)
    
    def calculate_temporal_similarity(self, timestamp1: str, timestamp2: str) -> float:
        """Calculate temporal proximity similarity"""
        
        try:
            dt1 = datetime.fromisoformat(timestamp1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(timestamp2.replace('Z', '+00:00'))
            
            time_diff = abs((dt1 - dt2).total_seconds())
            
            # Similarity decreases exponentially with time difference
            # Peak similarity within 1 hour, decreases to 0.1 after 24 hours
            similarity = np.exp(-time_diff / 7200)  # 2-hour half-life
            return max(0.1, similarity)
            
        except:
            return 0.5  # Default similarity if parsing fails
    
    def calculate_semantic_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate semantic similarity between token lists"""
        
        if not tokens1 or not tokens2:
            return 0.0
        
        set1, set2 = set(tokens1), set(tokens2)
        
        # Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def multi_algorithm_optimization(self, G: nx.Graph, episodes: List[Dict]) -> List[Dict[str, Any]]:
        """Apply multiple optimization algorithms and compare results"""
        
        optimization_results = []
        
        # 1. Modularity-based optimization
        try:
            modularity_result = self.modularity_optimization(G, episodes)
            optimization_results.append(modularity_result)
        except Exception as e:
            logger.warning(f"Modularity optimization failed: {e}")
        
        # 2. Spectral clustering optimization
        try:
            spectral_result = self.spectral_clustering_optimization(G, episodes)
            optimization_results.append(spectral_result)
        except Exception as e:
            logger.warning(f"Spectral clustering optimization failed: {e}")
        
        # 3. Conductance-based optimization
        try:
            conductance_result = self.conductance_optimization(G, episodes)
            optimization_results.append(conductance_result)
        except Exception as e:
            logger.warning(f"Conductance optimization failed: {e}")
        
        # 4. Custom boundary detection
        try:
            custom_result = self.custom_boundary_optimization(G, episodes)
            optimization_results.append(custom_result)
        except Exception as e:
            logger.warning(f"Custom optimization failed: {e}")
        
        return optimization_results
    
    def modularity_optimization(self, G: nx.Graph, episodes: List[Dict]) -> Dict[str, Any]:
        """Optimize boundaries using modularity maximization"""
        
        # Use Louvain algorithm for community detection
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        
        # Convert communities to boundary-marked episodes
        optimized_episodes = episodes.copy()
        
        # Mark boundary episodes (first episode of each community)
        for i, episode in enumerate(optimized_episodes):
            episode['is_boundary'] = False
        
        for community in communities:
            if community:  # Non-empty community
                first_node = min(community)
                optimized_episodes[first_node]['is_boundary'] = True
        
        # Calculate metrics
        metrics = self.calculate_boundary_metrics(G, optimized_episodes)
        
        return {
            'algorithm': 'modularity',
            'episodes': optimized_episodes,
            'metrics': metrics,
            'communities': len(communities),
            'improvement': metrics.get('modularity', 0) - self.calculate_baseline_modularity(G)
        }
    
    def spectral_clustering_optimization(self, G: nx.Graph, episodes: List[Dict]) -> Dict[str, Any]:
        """Optimize boundaries using spectral clustering"""
        
        if len(episodes) < 4:
            return {'algorithm': 'spectral', 'improvement': 0, 'episodes': episodes}
        
        # Create adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(G).todense()
        
        # Determine optimal number of clusters
        n_clusters = max(2, min(len(episodes) // 3, 8))
        
        # Apply spectral clustering
        clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering.fit_predict(adjacency_matrix)
        
        # Convert clusters to boundary-marked episodes
        optimized_episodes = episodes.copy()
        
        # Mark boundaries at cluster transitions
        for i, episode in enumerate(optimized_episodes):
            episode['is_boundary'] = False
            episode['cluster_label'] = int(cluster_labels[i])
        
        # Mark first episode of each cluster as boundary
        seen_clusters = set()
        for i, episode in enumerate(optimized_episodes):
            cluster = episode['cluster_label']
            if cluster not in seen_clusters:
                episode['is_boundary'] = True
                seen_clusters.add(cluster)
        
        # Calculate metrics
        metrics = self.calculate_boundary_metrics(G, optimized_episodes)
        silhouette = silhouette_score(adjacency_matrix, cluster_labels) if len(set(cluster_labels)) > 1 else 0
        metrics['silhouette_score'] = silhouette
        
        return {
            'algorithm': 'spectral',
            'episodes': optimized_episodes,
            'metrics': metrics,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'improvement': silhouette  # Use silhouette as improvement metric
        }
    
    def conductance_optimization(self, G: nx.Graph, episodes: List[Dict]) -> Dict[str, Any]:
        """Optimize boundaries using conductance minimization"""
        
        # Find minimum conductance cuts
        best_conductance = float('inf')
        best_partition = None
        
        # Try different partition sizes
        n_nodes = len(G.nodes())
        for partition_size in range(max(1, n_nodes // 4), min(n_nodes, 3 * n_nodes // 4)):
            try:
                # Generate random partitions and evaluate conductance
                for _ in range(10):  # Try multiple random partitions
                    nodes = list(G.nodes())
                    random.shuffle(nodes)
                    partition_a = set(nodes[:partition_size])
                    partition_b = set(nodes[partition_size:])
                    
                    conductance = self.calculate_conductance(G, partition_a, partition_b)
                    
                    if conductance < best_conductance:
                        best_conductance = conductance
                        best_partition = (partition_a, partition_b)
            except:
                continue
        
        if best_partition is None:
            return {'algorithm': 'conductance', 'improvement': 0, 'episodes': episodes}
        
        # Apply best partition to episodes
        optimized_episodes = episodes.copy()
        partition_a, partition_b = best_partition
        
        for i, episode in enumerate(optimized_episodes):
            episode['is_boundary'] = i in partition_a and (i + 1) in partition_b
        
        # Calculate metrics
        metrics = self.calculate_boundary_metrics(G, optimized_episodes)
        metrics['conductance'] = best_conductance
        
        return {
            'algorithm': 'conductance',
            'episodes': optimized_episodes,
            'metrics': metrics,
            'conductance': best_conductance,
            'improvement': 1.0 - best_conductance  # Lower conductance is better
        }
    
    def custom_boundary_optimization(self, G: nx.Graph, episodes: List[Dict]) -> Dict[str, Any]:
        """Custom boundary optimization using domain-specific heuristics"""
        
        optimized_episodes = episodes.copy()
        
        # Apply domain-specific boundary detection
        for i, episode in enumerate(optimized_episodes):
            episode['is_boundary'] = False
            
            # Check for significant topic shifts
            if i > 0:
                topic_shift = self.detect_topic_shift(episodes[i-1], episode)
                time_gap = self.detect_temporal_gap(episodes[i-1], episode)
                novelty_spike = self.detect_novelty_spike(episodes[i-1], episode)
                
                # Combine indicators for boundary decision
                boundary_score = topic_shift * 0.4 + time_gap * 0.3 + novelty_spike * 0.3
                
                if boundary_score > 0.6:
                    episode['is_boundary'] = True
                    episode['boundary_score'] = boundary_score
        
        # Calculate metrics
        metrics = self.calculate_boundary_metrics(G, optimized_episodes)
        
        # Calculate improvement based on boundary quality
        avg_boundary_score = np.mean([ep.get('boundary_score', 0.5) 
                                    for ep in optimized_episodes if ep.get('is_boundary')])
        
        return {
            'algorithm': 'custom',
            'episodes': optimized_episodes,
            'metrics': metrics,
            'avg_boundary_score': avg_boundary_score,
            'improvement': avg_boundary_score - 0.5  # Improvement over random
        }
    
    def detect_topic_shift(self, ep1: Dict, ep2: Dict) -> float:
        """Detect topic shift between consecutive episodes"""
        
        if 'content' not in ep1 or 'content' not in ep2:
            return 0.5
        
        # Simple topic shift detection based on content overlap
        tokens1 = set(ep1['content'].lower().split())
        tokens2 = set(ep2['content'].lower().split())
        
        if not tokens1 or not tokens2:
            return 1.0
        
        overlap = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        # High topic shift = low overlap
        similarity = overlap / union if union > 0 else 0
        return 1.0 - similarity
    
    def detect_temporal_gap(self, ep1: Dict, ep2: Dict) -> float:
        """Detect temporal gap between episodes"""
        
        try:
            dt1 = datetime.fromisoformat(ep1['timestamp'].replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(ep2['timestamp'].replace('Z', '+00:00'))
            
            gap_seconds = abs((dt2 - dt1).total_seconds())
            
            # Normalize gap to 0-1 scale (1 hour = 0.5, 6 hours = 1.0)
            normalized_gap = min(1.0, gap_seconds / 21600)  # 6 hours
            return normalized_gap
            
        except:
            return 0.0
    
    def detect_novelty_spike(self, ep1: Dict, ep2: Dict) -> float:
        """Detect novelty spike between episodes"""
        
        novelty1 = ep1.get('novelty_score', 0.5)
        novelty2 = ep2.get('novelty_score', 0.5)
        
        # Spike if second episode is significantly more novel
        novelty_increase = max(0, novelty2 - novelty1)
        return min(1.0, novelty_increase * 2)  # Amplify spikes
    
    def calculate_boundary_metrics(self, G: nx.Graph, episodes: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive boundary quality metrics - FIXED"""
        
        metrics = {}
        
        try:
            # Safe handling of empty graphs/episodes
            if not episodes or len(episodes) == 0:
                return {
                    'modularity': 0.0,
                    'clustering_coefficient': 0.0,
                    'boundary_count': 0,
                    'boundary_density': 0.0,
                    'diameter': float('inf'),
                    'average_shortest_path': float('inf'),
                    'edge_density': 0.0,
                    'node_count': 0,
                    'edge_count': 0
                }
            
            # Modularity (safe calculation)
            try:
                communities = self.episodes_to_communities(episodes)
                if len(communities) > 1 and G.number_of_nodes() > 0:
                    metrics['modularity'] = nx.algorithms.community.modularity(G, communities)
                else:
                    metrics['modularity'] = 0.0
            except Exception as e:
                logger.warning(f"Modularity calculation failed: {e}")
                metrics['modularity'] = 0.0
                
            # Clustering coefficient (safe calculation)
            try:
                if G.number_of_nodes() > 0:
                    metrics['clustering_coefficient'] = nx.average_clustering(G)
                else:
                    metrics['clustering_coefficient'] = 0.0
            except Exception as e:
                logger.warning(f"Clustering coefficient calculation failed: {e}")
                metrics['clustering_coefficient'] = 0.0
            
            # Boundary counts
            boundary_count = sum(1 for ep in episodes if ep.get('is_boundary', False))
            metrics['boundary_count'] = boundary_count
            metrics['boundary_density'] = boundary_count / len(episodes)
            
            # Graph connectivity metrics (safe calculation)
            try:
                if G.number_of_nodes() > 1 and nx.is_connected(G):
                    metrics['diameter'] = nx.diameter(G)
                    metrics['average_shortest_path'] = nx.average_shortest_path_length(G)
                else:
                    metrics['diameter'] = float('inf')
                    metrics['average_shortest_path'] = float('inf')
            except Exception as e:
                logger.warning(f"Connectivity metrics calculation failed: {e}")
                metrics['diameter'] = float('inf')
                metrics['average_shortest_path'] = float('inf')
            
            # Edge density (safe calculation)
            try:
                metrics['edge_density'] = nx.density(G)
            except Exception as e:
                logger.warning(f"Edge density calculation failed: {e}")
                metrics['edge_density'] = 0.0
            
            # Additional metrics
            metrics['node_count'] = G.number_of_nodes()
            metrics['edge_count'] = G.number_of_edges()
            
        except Exception as e:
            logger.warning(f"Error calculating boundary metrics: {e}")
            metrics = {
                'error': str(e),
                'modularity': 0.0,
                'clustering_coefficient': 0.0,
                'boundary_count': 0,
                'boundary_density': 0.0,
                'edge_density': 0.0
            }
        
        return metrics
    
    def calculate_conductance(self, G: nx.Graph, partition_a: Set, partition_b: Set) -> float:
        """Calculate conductance between two partitions"""
        
        if not partition_a or not partition_b:
            return 1.0
        
        # Count edges between partitions
        cut_edges = 0
        for node_a in partition_a:
            for neighbor in G.neighbors(node_a):
                if neighbor in partition_b:
                    cut_edges += 1
        
        # Count total edges in smaller partition
        smaller_partition = partition_a if len(partition_a) <= len(partition_b) else partition_b
        partition_edges = 0
        
        for node in smaller_partition:
            partition_edges += G.degree(node)
        
        # Calculate conductance
        if partition_edges == 0:
            return 1.0
        
        conductance = cut_edges / partition_edges
        return conductance
    
    def episodes_to_communities(self, episodes: List[Dict]) -> List[Set[int]]:
        """Convert episode boundary markers to community structure"""
        
        communities = []
        current_community = set()
        
        for i, episode in enumerate(episodes):
            current_community.add(i)
            
            # End community at boundary
            if episode.get('is_boundary', False) and i > 0:
                if current_community:
                    communities.append(current_community)
                current_community = set()
        
        # Add final community
        if current_community:
            communities.append(current_community)
        
        return communities if communities else [set(range(len(episodes)))]
    
    def calculate_baseline_modularity(self, G: nx.Graph) -> float:
        """Calculate baseline modularity for comparison"""
        
        # Baseline: each node as its own community
        baseline_communities = [{node} for node in G.nodes()]
        
        try:
            return nx.algorithms.community.modularity(G, baseline_communities)
        except:
            return 0.0
    
    def select_best_optimization(self, optimization_results: List[Dict], 
                               current_metrics: Dict) -> Dict[str, Any]:
        """Select the best optimization result"""
        
        if not optimization_results:
            return {'improvement': 0, 'episodes': [], 'metrics': current_metrics}
        
        # Score each optimization result
        scored_results = []
        
        for result in optimization_results:
            score = self.calculate_optimization_score(result, current_metrics)
            result['optimization_score'] = score
            scored_results.append(result)
        
        # Return best result
        best_result = max(scored_results, key=lambda x: x['optimization_score'])
        return best_result
    
    def calculate_optimization_score(self, result: Dict, baseline_metrics: Dict) -> float:
        """Calculate optimization quality score"""
        
        score = 0.0
        
        # Weight different improvements
        improvement_weights = {
            'modularity': 0.3,
            'silhouette_score': 0.3,
            'conductance': 0.2,  # Lower is better, so we'll invert
            'boundary_score': 0.2
        }
        
        result_metrics = result.get('metrics', {})
        
        for metric, weight in improvement_weights.items():
            if metric in result_metrics:
                if metric == 'conductance':
                    # Lower conductance is better
                    improvement = max(0, 0.5 - result_metrics[metric])
                else:
                    improvement = result_metrics.get(metric, 0)
                
                score += improvement * weight
        
        # Add algorithm-specific bonuses
        if result.get('algorithm') == 'modularity' and result_metrics.get('modularity', 0) > 0.3:
            score += 0.1
        
        if result.get('algorithm') == 'spectral' and result_metrics.get('silhouette_score', 0) > 0.5:
            score += 0.1
        
        return score

# ============================================================================
# HIERARCHICAL MEMORY SYSTEM (NEW)
# ============================================================================

class HierarchicalMemorySystem:
    """Multi-level memory hierarchy with consolidation"""
    
    def __init__(self):
        # Memory hierarchy levels
        self.memory_levels = {
            'working': MemoryHierarchyLevel(
                level_name='working',
                capacity=7,  # Miller's law: 7±2 items
                retention_policy='fifo',
                consolidation_threshold=0.8,
                access_frequency_weight=0.5,
                importance_weight=0.5
            ),
            'short_term': MemoryHierarchyLevel(
                level_name='short_term',
                capacity=50,
                retention_policy='importance_lru',
                consolidation_threshold=0.7,
                access_frequency_weight=0.3,
                importance_weight=0.7
            ),
            'long_term': MemoryHierarchyLevel(
                level_name='long_term',
                capacity=10000,
                retention_policy='importance_aging',
                consolidation_threshold=0.6,
                access_frequency_weight=0.2,
                importance_weight=0.8
            ),
            'semantic': MemoryHierarchyLevel(
                level_name='semantic',
                capacity=-1,  # Unlimited
                retention_policy='none',
                consolidation_threshold=0.9,
                access_frequency_weight=0.1,
                importance_weight=0.9
            )
        }
        
        # Memory stores
        self.working_memory = deque(maxlen=self.memory_levels['working'].capacity)
        self.short_term_memory = deque(maxlen=self.memory_levels['short_term'].capacity)
        self.long_term_memory = {}
        self.semantic_memory = {}
        
        # Consolidation tracking
        self.consolidation_history = deque(maxlen=1000)
        self.forgetting_history = deque(maxlen=1000)
        
        # Background consolidation
        self.consolidation_thread = None
        self.consolidation_active = True
        
        self._start_background_consolidation()
    
    def store_experience(self, experience: SensorimotorExperience) -> Dict[str, Any]:
        """Store experience in appropriate memory level"""
        
        # Always start in working memory
        memory_item = self._create_memory_item(experience)
        self.working_memory.append(memory_item)
        
        # Check for immediate consolidation needs
        consolidation_results = self._check_immediate_consolidation()
        
        return {
            'storage_level': 'working',
            'memory_item_id': memory_item['item_id'],
            'consolidation_triggered': len(consolidation_results) > 0,
            'consolidation_results': consolidation_results
        }
    
    def retrieve_memories(self, query_experience: SensorimotorExperience, 
                         max_memories: int = 20) -> List[Dict[str, Any]]:
        """Retrieve memories from all hierarchy levels"""
        
        retrieved_memories = []
        
        # Search working memory first (highest priority)
        working_results = self._search_memory_level(
            query_experience, list(self.working_memory), 'working'
        )
        retrieved_memories.extend(working_results)
        
        # Search short-term memory
        if len(retrieved_memories) < max_memories:
            short_term_results = self._search_memory_level(
                query_experience, list(self.short_term_memory), 'short_term'
            )
            retrieved_memories.extend(short_term_results)
        
        # Search long-term memory
        if len(retrieved_memories) < max_memories:
            long_term_results = self._search_memory_level(
                query_experience, list(self.long_term_memory.values()), 'long_term'
            )
            retrieved_memories.extend(long_term_results)
        
        # Search semantic memory
        if len(retrieved_memories) < max_memories:
            semantic_results = self._search_semantic_memory(query_experience)
            retrieved_memories.extend(semantic_results)
        
        # Sort by relevance and return top results
        retrieved_memories.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Update access statistics
        for memory in retrieved_memories[:max_memories]:
            self._update_access_statistics(memory)
        
        return retrieved_memories[:max_memories]
    
    def _create_memory_item(self, experience: SensorimotorExperience) -> Dict[str, Any]:
        """Create memory item from experience"""
        
        return {
            'item_id': f"mem_{uuid.uuid4().hex[:8]}",
            'experience': experience,
            'storage_timestamp': time.time(),
            'access_count': 0,
            'last_access': time.time(),
            'importance_score': experience.importance_weight,
            'memory_strength': experience.memory_strength,
            'consolidation_score': 0.0,
            'level': 'working'
        }
    
    def _search_memory_level(self, query_experience: SensorimotorExperience, 
                           memory_items: List[Dict], level: str) -> List[Dict[str, Any]]:
        """Search specific memory level"""
        
        results = []
        
        for item in memory_items:
            relevance = self._calculate_memory_relevance(query_experience, item)
            
            if relevance > 0.3:  # Relevance threshold
                results.append({
                    'memory_item': item,
                    'relevance_score': relevance,
                    'memory_level': level,
                    'access_priority': self._calculate_access_priority(item, relevance)
                })
        
        return results
    
    def _search_semantic_memory(self, query_experience: SensorimotorExperience) -> List[Dict[str, Any]]:
        """Search semantic memory using concept matching"""
        
        results = []
        query_concepts = self._extract_concepts(query_experience)
        
        for concept, concept_data in self.semantic_memory.items():
            concept_relevance = self._calculate_concept_relevance(query_concepts, concept_data)
            
            if concept_relevance > 0.4:
                results.append({
                    'memory_item': concept_data,
                    'relevance_score': concept_relevance,
                    'memory_level': 'semantic',
                    'concept': concept,
                    'access_priority': concept_relevance
                })
        
        return results
    
    def _calculate_memory_relevance(self, query: SensorimotorExperience, 
                                  memory_item: Dict) -> float:
        """Calculate relevance between query and memory item"""
        
        stored_experience = memory_item['experience']
        
        # Multi-factor relevance calculation
        relevance_factors = {}
        
        # Content similarity
        if hasattr(stored_experience, 'content') and hasattr(query, 'content'):
            content_sim = self._calculate_content_similarity(query.content, stored_experience.content)
            relevance_factors['content'] = content_sim
        
        # Domain similarity
        if hasattr(stored_experience, 'domain') and hasattr(query, 'domain'):
            domain_sim = 1.0 if query.domain == stored_experience.domain else 0.3
            relevance_factors['domain'] = domain_sim
        
        # Temporal relevance (more recent = more relevant)
        storage_time = memory_item['storage_timestamp']
        time_diff = time.time() - storage_time
        temporal_relevance = np.exp(-time_diff / 86400)  # 1-day half-life
        relevance_factors['temporal'] = temporal_relevance
        
        # Importance boost
        importance_boost = memory_item['importance_score']
        relevance_factors['importance'] = importance_boost
        
        # Access frequency boost
        access_boost = min(1.0, memory_item['access_count'] / 10.0)
        relevance_factors['access'] = access_boost
        
        # Weighted combination
        weights = {
            'content': 0.4,
            'domain': 0.2,
            'temporal': 0.2,
            'importance': 0.1,
            'access': 0.1
        }
        
        relevance = sum(relevance_factors.get(factor, 0.5) * weight 
                       for factor, weight in weights.items())
        
        return relevance
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity (simplified)"""
        
        tokens1 = set(content1.lower().split())
        tokens2 = set(content2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_access_priority(self, memory_item: Dict, relevance: float) -> float:
        """Calculate access priority for memory item"""
        
        # Combine relevance with memory-specific factors
        memory_strength = memory_item['memory_strength']
        importance = memory_item['importance_score']
        recency = 1.0 / (1.0 + (time.time() - memory_item['last_access']) / 3600)  # Hour-based recency
        
        priority = (relevance * 0.5 + memory_strength * 0.2 + 
                   importance * 0.2 + recency * 0.1)
        
        return priority
    
    def _update_access_statistics(self, memory_result: Dict[str, Any]):
        """Update access statistics for retrieved memory"""
        
        memory_item = memory_result['memory_item']
        memory_item['access_count'] += 1
        memory_item['last_access'] = time.time()
        
        # Boost memory strength based on access
        memory_item['memory_strength'] = min(1.0, memory_item['memory_strength'] * 1.1)
    
    def _check_immediate_consolidation(self) -> List[Dict[str, Any]]:
        """Check if immediate consolidation is needed"""
        
        consolidation_results = []
        
        # Check working memory capacity
        if len(self.working_memory) >= self.memory_levels['working'].capacity:
            working_consolidation = self._consolidate_working_to_short_term()
            if working_consolidation:
                consolidation_results.append(working_consolidation)
        
        # Check short-term memory capacity
        if len(self.short_term_memory) >= self.memory_levels['short_term'].capacity:
            short_term_consolidation = self._consolidate_short_term_to_long_term()
            if short_term_consolidation:
                consolidation_results.append(short_term_consolidation)
        
        return consolidation_results
    
    def _consolidate_working_to_short_term(self) -> Optional[Dict[str, Any]]:
        """Consolidate working memory to short-term memory"""
        
        if not self.working_memory:
            return None
        
        # Select items for consolidation
        consolidation_candidates = self._select_consolidation_candidates(
            list(self.working_memory), self.memory_levels['short_term']
        )
        
        consolidated_count = 0
        for candidate in consolidation_candidates:
            candidate['level'] = 'short_term'
            candidate['consolidation_score'] = self._calculate_consolidation_score(candidate)
            self.short_term_memory.append(candidate)
            consolidated_count += 1
        
        # Remove consolidated items from working memory
        for candidate in consolidation_candidates:
            try:
                self.working_memory.remove(candidate)
            except ValueError:
                pass
        
        return {
            'consolidation_type': 'working_to_short_term',
            'items_consolidated': consolidated_count,
            'timestamp': time.time()
        }
    
    def _consolidate_short_term_to_long_term(self) -> Optional[Dict[str, Any]]:
        """Consolidate short-term memory to long-term memory"""
        
        if not self.short_term_memory:
            return None
        
        # Select items for long-term consolidation
        consolidation_candidates = self._select_consolidation_candidates(
            list(self.short_term_memory), self.memory_levels['long_term']
        )
        
        consolidated_count = 0
        for candidate in consolidation_candidates:
            candidate['level'] = 'long_term'
            candidate['consolidation_score'] = self._calculate_consolidation_score(candidate)
            
            # Generate unique key for long-term storage
            memory_key = f"ltm_{candidate['item_id']}_{int(time.time())}"
            self.long_term_memory[memory_key] = candidate
            consolidated_count += 1
            
            # Extract semantic patterns for semantic memory
            self._extract_and_store_semantic_patterns(candidate)
        
        # Remove consolidated items from short-term memory
        for candidate in consolidation_candidates:
            try:
                self.short_term_memory.remove(candidate)
            except ValueError:
                pass
        
        return {
            'consolidation_type': 'short_term_to_long_term',
            'items_consolidated': consolidated_count,
            'timestamp': time.time()
        }
    
    def _select_consolidation_candidates(self, memory_items: List[Dict], 
                                       target_level: MemoryHierarchyLevel) -> List[Dict]:
        """Select items for consolidation to target level"""
        
        candidates = []
        
        for item in memory_items:
            consolidation_score = self._calculate_consolidation_score(item)
            
            if consolidation_score >= target_level.consolidation_threshold:
                candidates.append(item)
        
        # Sort by consolidation score and return top candidates
        candidates.sort(key=lambda x: x.get('consolidation_score', 0), reverse=True)
        
        # Limit based on target level capacity
        if target_level.capacity > 0:
            max_candidates = max(1, target_level.capacity // 4)  # Consolidate 25% at a time
            candidates = candidates[:max_candidates]
        
        return candidates
    
    def _calculate_consolidation_score(self, memory_item: Dict) -> float:
        """Calculate consolidation score for memory item"""
        
        # Factors affecting consolidation
        importance = memory_item['importance_score']
        access_frequency = min(1.0, memory_item['access_count'] / 5.0)
        memory_strength = memory_item['memory_strength']
        
        # Time-based factors
        storage_time = memory_item['storage_timestamp']
        age_factor = min(1.0, (time.time() - storage_time) / 3600)  # Age in hours
        
        # Novelty factor
        experience = memory_item['experience']
        novelty_factor = getattr(experience, 'novelty_score', 0.5)
        
        # Combined consolidation score
        consolidation_score = (
            importance * 0.3 +
            access_frequency * 0.2 +
            memory_strength * 0.2 +
            age_factor * 0.1 +
            novelty_factor * 0.2
        )
        
        return consolidation_score
    
    def _extract_and_store_semantic_patterns(self, memory_item: Dict):
        """Extract semantic patterns and store in semantic memory"""
        
        experience = memory_item['experience']
        
        # Extract concepts
        concepts = self._extract_concepts(experience)
        
        for concept in concepts:
            if concept not in self.semantic_memory:
                self.semantic_memory[concept] = {
                    'concept': concept,
                    'instances': [],
                    'strength': 0.0,
                    'last_updated': time.time(),
                    'access_count': 0
                }
            
            # Add this instance
            self.semantic_memory[concept]['instances'].append({
                'memory_item_id': memory_item['item_id'],
                'experience_content': experience.content,
                'timestamp': memory_item['storage_timestamp'],
                'importance': memory_item['importance_score']
            })
            
            # Update concept strength
            self.semantic_memory[concept]['strength'] = min(1.0, 
                self.semantic_memory[concept]['strength'] + 0.1
            )
            self.semantic_memory[concept]['last_updated'] = time.time()
    
    def _extract_concepts(self, experience: SensorimotorExperience) -> List[str]:
        """Extract concepts from experience"""
        
        concepts = []
        
        # Domain-based concepts
        if hasattr(experience, 'domain'):
            concepts.append(experience.domain)
        
        # Content-based concepts (simplified)
        if hasattr(experience, 'content'):
            content_tokens = experience.content.lower().split()
            
            # Extract significant words as concepts
            for token in content_tokens:
                if len(token) > 4 and token.isalpha():
                    concepts.append(token)
        
        # Emotional concepts
        if hasattr(experience, 'emotional_features'):
            for emotion, intensity in experience.emotional_features.items():
                if intensity > 0.6:
                    concepts.append(f"emotion_{emotion}")
        
        return list(set(concepts))  # Remove duplicates
    
    def _calculate_concept_relevance(self, query_concepts: List[str], 
                                   concept_data: Dict) -> float:
        """Calculate relevance of concept to query"""
        
        concept = concept_data['concept']
        
        # Direct concept match
        if concept in query_concepts:
            return concept_data['strength']
        
        # Partial concept match
        for query_concept in query_concepts:
            if concept in query_concept or query_concept in concept:
                return concept_data['strength'] * 0.7
        
        return 0.0
    
    def _start_background_consolidation(self):
        """Start background consolidation thread"""
        
        def consolidation_worker():
            while self.consolidation_active:
                try:
                    self._background_consolidation_cycle()
                    time.sleep(30)  # Run every 30 seconds
                except Exception as e:
                    logger.error(f"Background consolidation error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self.consolidation_thread = threading.Thread(target=consolidation_worker, daemon=True)
        self.consolidation_thread.start()
    
    def _background_consolidation_cycle(self):
        """Background consolidation and maintenance"""
        
        # Periodic memory consolidation
        consolidation_results = self._check_immediate_consolidation()
        
        if consolidation_results:
            self.consolidation_history.extend(consolidation_results)
        
        # Intelligent forgetting
        forgotten_items = self._intelligent_forgetting()
        
        if forgotten_items:
            self.forgetting_history.extend(forgotten_items)
        
        # Semantic memory cleanup
        self._cleanup_semantic_memory()
    
    def _intelligent_forgetting(self) -> List[Dict[str, Any]]:
        """Implement intelligent forgetting based on forgetting curves"""
        
        forgotten_items = []
        current_time = time.time()
        
        # Forget from long-term memory
        items_to_forget = []
        
        for memory_key, memory_item in self.long_term_memory.items():
            forgetting_probability = self._calculate_forgetting_probability(memory_item, current_time)
            
            if forgetting_probability > 0.8:  # 80% forgetting threshold
                items_to_forget.append(memory_key)
                
                forgotten_items.append({
                    'memory_key': memory_key,
                    'item_id': memory_item['item_id'],
                    'forgetting_probability': forgetting_probability,
                    'timestamp': current_time,
                    'reason': 'forgetting_curve'
                })
        
        # Remove forgotten items
        for memory_key in items_to_forget:
            del self.long_term_memory[memory_key]
        
        return forgotten_items
    
    def _calculate_forgetting_probability(self, memory_item: Dict, current_time: float) -> float:
        """Calculate forgetting probability using Ebbinghaus forgetting curve"""
        
        # Time since last access
        last_access = memory_item['last_access']
        time_since_access = current_time - last_access
        
        # Base forgetting rate (1 day half-life)
        base_decay_rate = 1.0 / 86400  # 1/day in seconds
        
        # Adjust decay rate based on factors
        importance = memory_item['importance_score']
        access_frequency = memory_item['access_count']
        memory_strength = memory_item['memory_strength']
        
        # Factors that slow forgetting
        importance_factor = 1.0 - (importance * 0.5)
        frequency_factor = 1.0 / (1.0 + access_frequency * 0.1)
        strength_factor = 1.0 - (memory_strength * 0.3)
        
        adjusted_decay_rate = base_decay_rate * importance_factor * frequency_factor * strength_factor
        
        # Calculate forgetting probability
        forgetting_probability = 1.0 - np.exp(-adjusted_decay_rate * time_since_access)
        
        return forgetting_probability
    
    def _cleanup_semantic_memory(self):
        """Clean up semantic memory by removing weak concepts"""
        
        concepts_to_remove = []
        current_time = time.time()
        
        for concept, concept_data in self.semantic_memory.items():
            # Remove concepts that haven't been accessed recently and are weak
            time_since_update = current_time - concept_data['last_updated']
            
            if (concept_data['strength'] < 0.3 and 
                time_since_update > 7 * 86400 and  # 7 days
                concept_data['access_count'] < 2):
                
                concepts_to_remove.append(concept)
        
        for concept in concepts_to_remove:
            del self.semantic_memory[concept]
        
        logger.info(f"Cleaned up {len(concepts_to_remove)} weak semantic concepts")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        
        return {
            'working_memory': {
                'count': len(self.working_memory),
                'capacity': self.memory_levels['working'].capacity,
                'utilization': len(self.working_memory) / self.memory_levels['working'].capacity
            },
            'short_term_memory': {
                'count': len(self.short_term_memory),
                'capacity': self.memory_levels['short_term'].capacity,
                'utilization': len(self.short_term_memory) / self.memory_levels['short_term'].capacity
            },
            'long_term_memory': {
                'count': len(self.long_term_memory),
                'capacity': self.memory_levels['long_term'].capacity,
                'utilization': len(self.long_term_memory) / self.memory_levels['long_term'].capacity if self.memory_levels['long_term'].capacity > 0 else 0
            },
            'semantic_memory': {
                'concepts': len(self.semantic_memory),
                'total_instances': sum(len(concept_data['instances']) for concept_data in self.semantic_memory.values())
            },
            'consolidation_history': len(self.consolidation_history),
            'forgetting_history': len(self.forgetting_history),
            'background_consolidation_active': self.consolidation_active
        }

# ============================================================================
# MEMORY COMPRESSION SYSTEM (NEW)
# ============================================================================

class MemoryCompressionSystem:
    """Advanced memory compression with hierarchical abstractions"""
    
    def __init__(self, target_compression_ratio: float = 0.1):
        self.target_compression_ratio = target_compression_ratio
        self.abstraction_levels = ['concrete', 'categorical', 'abstract', 'semantic']
        self.compression_cache = {}
        self.pattern_library = {}
        self.reconstruction_fidelity_threshold = 0.8
        
    def compress_episode_sequence(self, episodes: List[Dict]) -> Tuple[CompressedMemoryBlock, Dict[str, Any]]:
        """Compress episode sequence using multiple techniques"""
        
        if len(episodes) < 2:
            return None, {'error': 'insufficient_episodes'}
        
        # Generate compression key
        episode_ids = [ep.get('episode_id', str(i)) for i, ep in enumerate(episodes)]
        compression_key = hashlib.md5(str(episode_ids).encode()).hexdigest()
        
        # Check cache first
        if compression_key in self.compression_cache:
            return self.compression_cache[compression_key], {'cache_hit': True}
        
        # Step 1: Identify recurring patterns
        patterns = self.identify_recurring_patterns(episodes)
        
        # Step 2: Create hierarchical abstractions
        abstractions = self.create_hierarchical_abstractions(episodes)
        
        # Step 3: Apply pattern-based compression
        compressed_data = self.pattern_based_compression(episodes, patterns)
        
        # Step 4: Create compressed memory block
        compressed_block = CompressedMemoryBlock(
            block_id=f"cmb_{uuid.uuid4().hex[:8]}",
            original_episode_ids=episode_ids,
            compressed_content=self._serialize_compressed_data(compressed_data),
            compression_metadata={
                'patterns': patterns,
                'compression_algorithm': 'pattern_hierarchical',
                'original_size': len(str(episodes)),
                'compressed_size': len(compressed_data),
                'abstraction_levels': len(self.abstraction_levels)
            },
            abstraction_levels=abstractions,
            reconstruction_fidelity=self._calculate_reconstruction_fidelity(episodes, compressed_data),
            compression_ratio=len(compressed_data) / len(str(episodes)),
            created_timestamp=datetime.now().isoformat()
        )
        
        # Cache the result
        self.compression_cache[compression_key] = compressed_block
        
        # Update pattern library
        self._update_pattern_library(patterns)
        
        return compressed_block, {
            'compression_ratio': compressed_block.compression_ratio,
            'reconstruction_fidelity': compressed_block.reconstruction_fidelity,
            'patterns_found': len(patterns),
            'abstraction_levels': len(abstractions)
        }
    
    def identify_recurring_patterns(self, episodes: List[Dict]) -> Dict[str, Any]:
        """Identify recurring patterns in episode sequences"""
        
        patterns = {}
        
        # Temporal patterns (sequence patterns)
        for window_size in [2, 3, 4, 5]:
            sequence_patterns = self._find_sequence_patterns(episodes, window_size)
            patterns[f'sequence_{window_size}'] = sequence_patterns
        
        # Content patterns (repeated content structures)
        content_patterns = self._find_content_patterns(episodes)
        patterns['content'] = content_patterns
        
        # Domain patterns (domain transitions)
        domain_patterns = self._find_domain_patterns(episodes)
        patterns['domain'] = domain_patterns
        
        # Structural patterns (metadata patterns)
        structural_patterns = self._find_structural_patterns(episodes)
        patterns['structural'] = structural_patterns
        
        return patterns
    
    def _find_sequence_patterns(self, episodes: List[Dict], window_size: int) -> Dict[str, Any]:
        """Find recurring sequence patterns"""
        
        if len(episodes) < window_size:
            return {}
        
        sequence_patterns = {}
        
        for i in range(len(episodes) - window_size + 1):
            window = episodes[i:i + window_size]
            
            # Create pattern signature
            pattern_sig = self._create_sequence_signature(window)
            
            if pattern_sig not in sequence_patterns:
                sequence_patterns[pattern_sig] = {
                    'episodes': window,
                    'frequency': 0,
                    'positions': [],
                    'pattern_type': 'sequence'
                }
            
            sequence_patterns[pattern_sig]['frequency'] += 1
            sequence_patterns[pattern_sig]['positions'].append(i)
        
        # Filter frequent patterns
        frequent_patterns = {
            sig: data for sig, data in sequence_patterns.items() 
            if data['frequency'] >= 2
        }
        
        return frequent_patterns
    
    def _create_sequence_signature(self, episode_window: List[Dict]) -> str:
        """Create signature for episode sequence"""
        
        signature_parts = []
        
        for episode in episode_window:
            # Use domain and content length as signature components
            domain = episode.get('domain', 'unknown')
            content_length = len(episode.get('content', ''))
            novelty = episode.get('novelty_score', 0.5)
            
            # Discretize values
            length_bucket = content_length // 100  # 100-char buckets
            novelty_bucket = int(novelty * 10)     # 0.1 buckets
            
            sig_part = f"{domain}_{length_bucket}_{novelty_bucket}"
            signature_parts.append(sig_part)
        
        return '|'.join(signature_parts)
    
    def _find_content_patterns(self, episodes: List[Dict]) -> Dict[str, Any]:
        """Find recurring content patterns"""
        
        content_patterns = {}
        
        # Extract common phrases and structures
        all_content = [ep.get('content', '') for ep in episodes]
        
        # Find common phrases (simplified)
        phrase_counts = Counter()
        
        for content in all_content:
            words = content.lower().split()
            
            # Extract 2-3 word phrases
            for i in range(len(words) - 1):
                phrase = ' '.join(words[i:i+2])
                if len(phrase) > 8:  # Significant phrases only
                    phrase_counts[phrase] += 1
                
                if i < len(words) - 2:
                    phrase_3 = ' '.join(words[i:i+3])
                    if len(phrase_3) > 12:
                        phrase_counts[phrase_3] += 1
        
        # Keep frequent phrases
        frequent_phrases = {
            phrase: count for phrase, count in phrase_counts.items()
            if count >= 2
        }
        
        content_patterns['phrases'] = frequent_phrases
        
        return content_patterns
    
    def _find_domain_patterns(self, episodes: List[Dict]) -> Dict[str, Any]:
        """Find domain transition patterns"""
        
        domains = [ep.get('domain', 'unknown') for ep in episodes]
        
        # Find domain transition patterns
        transitions = {}
        
        for i in range(len(domains) - 1):
            transition = f"{domains[i]} -> {domains[i+1]}"
            
            if transition not in transitions:
                transitions[transition] = 0
            transitions[transition] += 1
        
        # Find domain clusters
        domain_counts = Counter(domains)
        
        return {
            'transitions': transitions,
            'domain_distribution': dict(domain_counts),
            'dominant_domain': domain_counts.most_common(1)[0] if domain_counts else None
        }
    
    def _find_structural_patterns(self, episodes: List[Dict]) -> Dict[str, Any]:
        """Find structural metadata patterns"""
        
        structural_patterns = {}
        
        # Novelty patterns
        novelty_scores = [ep.get('novelty_score', 0.5) for ep in episodes]
        structural_patterns['novelty_trend'] = {
            'mean': np.mean(novelty_scores),
            'std': np.std(novelty_scores),
            'trend': 'increasing' if novelty_scores[-1] > novelty_scores[0] else 'decreasing'
        }
        
        # Timestamp patterns
        timestamps = [ep.get('timestamp', '') for ep in episodes]
        if timestamps and timestamps[0]:
            structural_patterns['temporal_span'] = {
                'start': timestamps[0],
                'end': timestamps[-1],
                'count': len(timestamps)
            }
        
        return structural_patterns
    
    def create_hierarchical_abstractions(self, episodes: List[Dict]) -> Dict[str, Any]:
        """Create multi-level abstractions of episodes"""
        
        abstractions = {}
        
        for level in self.abstraction_levels:
            abstractions[level] = self._create_abstraction_level(episodes, level)
        
        return abstractions
    
    def _create_abstraction_level(self, episodes: List[Dict], level: str) -> Dict[str, Any]:
        """Create specific abstraction level"""
        
        if level == 'concrete':
            # Preserve original episodes with minimal compression
            return {
                'type': 'concrete',
                'data': [self._create_episode_summary(ep) for ep in episodes]
            }
        
        elif level == 'categorical':
            # Group by categories
            categories = {}
            for ep in episodes:
                domain = ep.get('domain', 'unknown')
                if domain not in categories:
                    categories[domain] = []
                categories[domain].append(self._create_episode_summary(ep))
            
            return {
                'type': 'categorical',
                'data': categories
            }
        
        elif level == 'abstract':
            # Extract high-level themes
            themes = self._extract_abstract_themes(episodes)
            return {
                'type': 'abstract',
                'data': themes
            }
        
        elif level == 'semantic':
            # Create semantic knowledge representation
            semantic_graph = self._create_semantic_graph(episodes)
            return {
                'type': 'semantic',
                'data': semantic_graph
            }
        
        return {}
    
    def _create_episode_summary(self, episode: Dict) -> Dict[str, Any]:
        """Create compact episode summary"""
        
        return {
            'id': episode.get('episode_id', ''),
            'domain': episode.get('domain', ''),
            'content_length': len(episode.get('content', '')),
            'novelty': episode.get('novelty_score', 0.5),
            'timestamp': episode.get('timestamp', ''),
            'key_tokens': episode.get('representative_tokens', [])[:3]
        }
    
    def _extract_abstract_themes(self, episodes: List[Dict]) -> Dict[str, Any]:
        """Extract abstract themes from episodes"""
        
        themes = {}
        
        # Content themes
        all_content = ' '.join([ep.get('content', '') for ep in episodes])
        content_words = all_content.lower().split()
        
        # Simple keyword extraction
        word_counts = Counter(word for word in content_words if len(word) > 4)
        top_keywords = word_counts.most_common(10)
        
        themes['content_themes'] = [word for word, count in top_keywords]
        
        # Domain themes
        domains = [ep.get('domain', '') for ep in episodes]
        domain_counts = Counter(domains)
        themes['domain_themes'] = list(domain_counts.keys())
        
        # Temporal themes
        timestamps = [ep.get('timestamp', '') for ep in episodes if ep.get('timestamp')]
        if timestamps:
            themes['temporal_theme'] = {
                'span': f"{timestamps[0][:10]} to {timestamps[-1][:10]}",
                'episode_count': len(episodes)
            }
        
        return themes
    
    def _create_semantic_graph(self, episodes: List[Dict]) -> Dict[str, Any]:
        """Create semantic knowledge graph"""
        
        # Create simple semantic graph
        graph = {
            'nodes': [],
            'edges': [],
            'concepts': {}
        }
        
        # Extract concepts and relationships
        for i, episode in enumerate(episodes):
            content = episode.get('content', '')
            domain = episode.get('domain', '')
            
            # Add episode node
            episode_node = {
                'id': f"ep_{i}",
                'type': 'episode',
                'domain': domain,
                'content_length': len(content)
            }
            graph['nodes'].append(episode_node)
            
            # Extract key concepts
            key_words = [word for word in content.lower().split() if len(word) > 5][:5]
            
            for word in key_words:
                if word not in graph['concepts']:
                    graph['concepts'][word] = {
                        'frequency': 0,
                        'episodes': []
                    }
                
                graph['concepts'][word]['frequency'] += 1
                graph['concepts'][word]['episodes'].append(f"ep_{i}")
        
        # Create edges between related episodes
        for i in range(len(episodes) - 1):
            if episodes[i].get('domain') == episodes[i+1].get('domain'):
                graph['edges'].append({
                    'source': f"ep_{i}",
                    'target': f"ep_{i+1}",
                    'type': 'temporal_sequence'
                })
        
        return graph
    
    def pattern_based_compression(self, episodes: List[Dict], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Apply pattern-based compression"""
        
        compressed_data = {
            'pattern_substitutions': {},
            'original_indices': [],
            'compressed_episodes': []
        }
        
        # Apply pattern substitutions
        substitution_map = self._create_pattern_substitution_map(patterns)
        
        # Compress episodes
        for i, episode in enumerate(episodes):
            compressed_episode = self._compress_single_episode(episode, substitution_map)
            compressed_data['compressed_episodes'].append(compressed_episode)
            compressed_data['original_indices'].append(i)
        
        compressed_data['pattern_substitutions'] = substitution_map
        
        return compressed_data
    
    def _create_pattern_substitution_map(self, patterns: Dict[str, Any]) -> Dict[str, str]:
        """Create pattern to symbol substitution map"""
        
        substitution_map = {}
        symbol_counter = 0
        
        for pattern_type, pattern_data in patterns.items():
            if pattern_type == 'content' and 'phrases' in pattern_data:
                for phrase, frequency in pattern_data['phrases'].items():
                    if frequency >= 3:  # Substitute frequent phrases
                        symbol = f"#P{symbol_counter}#"
                        substitution_map[phrase] = symbol
                        symbol_counter += 1
        
        return substitution_map
    
    def _compress_single_episode(self, episode: Dict, substitution_map: Dict[str, str]) -> Dict[str, Any]:
        """Compress a single episode using patterns"""
        
        compressed_episode = {}
        
        # Compress content using pattern substitution
        content = episode.get('content', '')
        for pattern, symbol in substitution_map.items():
            content = content.replace(pattern, symbol)
        
        compressed_episode['content'] = content
        
        # Keep essential metadata
        compressed_episode['domain'] = episode.get('domain', '')
        compressed_episode['novelty_score'] = episode.get('novelty_score', 0.5)
        compressed_episode['timestamp'] = episode.get('timestamp', '')
        
        # Compress other fields
        if 'representative_tokens' in episode:
            compressed_episode['tokens'] = episode['representative_tokens'][:3]  # Keep top 3
        
        return compressed_episode
    
    def _serialize_compressed_data(self, compressed_data: Dict[str, Any]) -> bytes:
        """Serialize compressed data to bytes"""
        
        try:
            # Use gzip compression on top of pickle
            pickled_data = pickle.dumps(compressed_data)
            compressed_bytes = gzip.compress(pickled_data)
            return compressed_bytes
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            return b''
    
    def _calculate_reconstruction_fidelity(self, original_episodes: List[Dict], 
                                         compressed_data: Dict[str, Any]) -> float:
        """Calculate reconstruction fidelity"""
        
        try:
            # Simple fidelity calculation based on content preservation
            original_content = ' '.join([ep.get('content', '') for ep in original_episodes])
            
            # Estimate compressed content size
            compressed_episodes = compressed_data.get('compressed_episodes', [])
            compressed_content = ' '.join([ep.get('content', '') for ep in compressed_episodes])
            
            # Calculate content preservation ratio
            if len(original_content) == 0:
                return 1.0
            
            content_preservation = len(compressed_content) / len(original_content)
            
            # Factor in pattern substitution quality
            pattern_count = len(compressed_data.get('pattern_substitutions', {}))
            pattern_factor = min(1.0, pattern_count / 10.0)  # Bonus for pattern detection
            
            fidelity = min(1.0, content_preservation + pattern_factor * 0.1)
            return fidelity
            
        except Exception as e:
            logger.error(f"Fidelity calculation failed: {e}")
            return 0.5
    
    def _update_pattern_library(self, patterns: Dict[str, Any]):
        """Update global pattern library"""
        
        for pattern_type, pattern_data in patterns.items():
            if pattern_type not in self.pattern_library:
                self.pattern_library[pattern_type] = {}
            
            # Update pattern statistics
            if pattern_type == 'content' and 'phrases' in pattern_data:
                for phrase, frequency in pattern_data['phrases'].items():
                    if phrase not in self.pattern_library[pattern_type]:
                        self.pattern_library[pattern_type][phrase] = 0
                    self.pattern_library[pattern_type][phrase] += frequency
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get compression system statistics"""
        
        return {
            'cache_size': len(self.compression_cache),
            'pattern_library_size': sum(len(patterns) for patterns in self.pattern_library.values()),
            'target_compression_ratio': self.target_compression_ratio,
            'reconstruction_fidelity_threshold': self.reconstruction_fidelity_threshold,
            'abstraction_levels': self.abstraction_levels
        }

# ============================================================================
# CROSS-MODAL MEMORY INTEGRATION (NEW)
# ============================================================================

class CrossModalMemorySystem:
    """Cross-modal memory integration system"""
    
    def __init__(self):
        self.modalities = ['text', 'visual', 'audio', 'temporal', 'spatial', 'emotional']
        self.cross_modal_graph = nx.MultiGraph()
        self.modal_indices = {modality: {} for modality in self.modalities}
        self.association_strength_threshold = 0.6
        self.cross_modal_lock = threading.Lock()
        
    def store_cross_modal_experience(self, experience: SensorimotorExperience) -> Dict[str, Any]:
        """Store experience with cross-modal associations"""
        
        experience_id = experience.experience_id
        
        # Extract features for each modality
        modal_features = {}
        for modality in self.modalities:
            features = self._extract_modal_features(experience, modality)
            if features is not None:
                modal_features[modality] = features
        
        # Store in modality-specific indices
        storage_results = {}
        for modality, features in modal_features.items():
            storage_result = self._store_in_modal_index(experience_id, modality, features)
            storage_results[modality] = storage_result
        
        # Create cross-modal associations
        associations = self._create_cross_modal_associations(experience_id, modal_features)
        
        # Add to cross-modal graph
        self._add_to_cross_modal_graph(experience_id, modal_features, associations)
        
        return {
            'experience_id': experience_id,
            'modalities_stored': list(modal_features.keys()),
            'cross_modal_associations': len(associations),
            'storage_results': storage_results
        }
    def process_cross_modal_experience(self, experience: SensorimotorExperience) -> Dict[str, Any]:
        """Process cross-modal experience (alias for store_cross_modal_experience)"""
        try:
            return self.store_cross_modal_experience(experience)
        except Exception as e:
            return {'error': str(e), 'modalities_stored': [], 'cross_modal_associations': 0}
    def retrieve_cross_modal(self, query_experience: SensorimotorExperience, 
                           target_modalities: List[str] = None,
                           max_results: int = 20) -> List[Dict[str, Any]]:
        """Retrieve experiences using cross-modal associations"""
        
        if target_modalities is None:
            target_modalities = self.modalities
        
        # Extract query features for each target modality
        query_features = {}
        for modality in target_modalities:
            features = self._extract_modal_features(query_experience, modality)
            if features is not None:
                query_features[modality] = features
        
        # Find matches in each modality
        modal_matches = {}
        for modality, features in query_features.items():
            matches = self._find_modal_matches(modality, features, max_results * 2)
            modal_matches[modality] = matches
        
        # Combine cross-modal matches
        cross_modal_results = self._combine_cross_modal_matches(modal_matches, query_features)
        
        # Rank by cross-modal relevance
        ranked_results = self._rank_cross_modal_results(cross_modal_results, query_features)
        
        return ranked_results[:max_results]
    def _extract_text_features_fixed(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract text features with consistent dimension"""
        
        content = getattr(experience, 'content', '')
        words = content.lower().split()
        
        features = []
        
        # Basic text statistics
        features.append(len(words) / 100.0)  # Word count
        features.append(len(content) / 1000.0)  # Character count
        features.append(len(set(words)) / max(len(words), 1))  # Lexical diversity
        features.append(np.mean([len(word) for word in words]) if words else 0)  # Avg word length
        
        # Punctuation features
        features.append(content.count('.') / max(len(words), 1))
        features.append(content.count('!') / max(len(words), 1))
        features.append(content.count('?') / max(len(words), 1))
        features.append(content.count(',') / max(len(words), 1))
        
        # Content complexity
        features.append(getattr(experience, 'novelty_score', 0.5))
        features.append(min(1.0, len(content.split('.')) / 10.0))  # Sentence complexity
        
        # Semantic categories
        categories = {
            'analytical': ['analysis', 'data', 'study', 'research'],
            'emotional': ['feel', 'emotion', 'sentiment', 'mood'],
            'action': ['action', 'do', 'perform', 'execute']
        }
        
        for category, keywords in categories.items():
            score = sum(1 for word in keywords if word in words) / max(len(keywords), 1)
            features.append(score)
        
        # Domain-specific features
        domain = getattr(experience, 'domain', 'general')
        if domain == 'financial_analysis':
            features.append(1.0)
            features.append(0.0)
        elif domain == 'research':
            features.append(0.0)
            features.append(1.0)
        else:
            features.append(0.5)
            features.append(0.5)
        
        # Random feature for uniqueness
        features.append(random.uniform(0, 0.1))
        
        return np.array(features[:16])  # Ensure exactly 16 features
    def _create_spectral_clusters(self, similarity_matrix: np.ndarray, 
                                n_clusters: int = 3) -> np.ndarray:
        """Create spectral clusters with error handling - FIXED"""
        
        try:
            # Validate input
            if similarity_matrix.shape[0] < n_clusters:
                # Not enough data points for clustering
                return np.zeros(similarity_matrix.shape[0], dtype=int)
            
            # Ensure matrix is properly formatted
            similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Add small diagonal to avoid singular matrix
            similarity_matrix += np.eye(similarity_matrix.shape[0]) * 1e-6
            
            from sklearn.cluster import SpectralClustering
            
            # Use precomputed affinity matrix
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42,
                assign_labels='discretize'  # More stable than 'kmeans'
            )
            
            cluster_labels = spectral.fit_predict(similarity_matrix)
            return cluster_labels
            
        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}")
            # Return simple sequential clustering as fallback
            n_points = similarity_matrix.shape[0]
            return np.array([i % n_clusters for i in range(n_points)])

    def _extract_temporal_features_fixed(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract temporal features with consistent dimension"""
        
        features = []
        
        current_time = time.time()
        
        # Time-based features
        features.append((current_time % 86400) / 86400)  # Time of day
        features.append(datetime.now().weekday() / 6)  # Day of week
        features.append((current_time % 3600) / 3600)  # Hour fraction
        
        # Content temporal markers
        content = getattr(experience, 'content', '')
        temporal_words = ['now', 'today', 'yesterday', 'tomorrow', 'soon', 'later', 'before', 'after']
        features.append(sum(1 for word in temporal_words if word in content.lower()) / max(len(content.split()), 1))
        
        # Sequence features
        features.append(getattr(experience, 'sequence_position', 0.5))
        features.append(min(1.0, len(content) / 500.0))  # Content duration proxy
        
        # Temporal patterns
        features.extend([
            random.uniform(0, 0.1),  # Temporal novelty
            random.uniform(0, 0.1),  # Temporal consistency
            random.uniform(0, 0.1),  # Temporal surprise
            random.uniform(0, 0.1),  # Temporal importance
            random.uniform(0, 0.1),  # Temporal context
            random.uniform(0, 0.1),  # Temporal relevance
            random.uniform(0, 0.1),  # Temporal persistence
            random.uniform(0, 0.1),  # Temporal dynamics
            random.uniform(0, 0.1),  # Temporal coherence
            random.uniform(0, 0.1)   # Temporal stability
        ])
        
        return np.array(features[:16])
    def _extract_spatial_features_fixed(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract spatial features with consistent dimension"""
        
        features = []
        
        # Domain-based spatial encoding
        domain = getattr(experience, 'domain', 'unknown')
        domain_map = {
            'financial_analysis': [1.0, 0.0, 0.0],
            'research': [0.0, 1.0, 0.0],
            'general': [0.0, 0.0, 1.0]
        }
        
        domain_features = domain_map.get(domain, [0.33, 0.33, 0.33])
        features.extend(domain_features)
        
        # Content-based spatial features
        content = getattr(experience, 'content', '')
        content_words = content.lower().split()
        
        # Conceptual dimensions
        abstract_words = ['concept', 'idea', 'theory', 'principle', 'approach']
        concrete_words = ['data', 'number', 'result', 'measurement', 'fact']
        
        abstract_score = sum(1 for word in content_words if word in abstract_words) / max(len(content_words), 1)
        concrete_score = sum(1 for word in content_words if word in concrete_words) / max(len(content_words), 1)
        
        features.extend([abstract_score, concrete_score])
        
        # Additional spatial features
        features.append(getattr(experience, 'novelty_score', 0.5))
        features.append(len(content) / 1000.0)  # Content size
        features.append(len(set(content_words)) / max(len(content_words), 1))  # Diversity
        
        # Fill to 16 features
        while len(features) < 16:
            features.append(random.uniform(0, 0.05))
        
        return np.array(features[:16])

    def _extract_emotional_features_fixed(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract emotional features with consistent dimension"""
        
        content = getattr(experience, 'content', '').lower()
        features = []
        
        # Emotion categories
        emotion_keywords = {
            'positive': ['good', 'great', 'excellent', 'success', 'growth', 'improve'],
            'negative': ['bad', 'poor', 'decline', 'loss', 'fail', 'decrease'],
            'neutral': ['stable', 'steady', 'maintain', 'continue', 'same'],
            'excitement': ['surge', 'boom', 'spike', 'jump', 'rocket'],
            'concern': ['risk', 'danger', 'worry', 'caution', 'warning']
        }
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content) / max(len(content.split()), 1)
            features.append(score)
        
        # Emotional intensity markers
        features.append(content.count('!') / max(len(content.split()), 1))  # Excitement
        features.append(content.count('?') / max(len(content.split()), 1))  # Uncertainty
        features.append(content.count('.') / max(len(content.split()), 1))  # Neutrality
        
        # Sentiment polarity (simple)
        positive_words = ['good', 'great', 'excellent', 'success', 'up', 'rise']
        negative_words = ['bad', 'poor', 'terrible', 'failure', 'down', 'fall']
        
        pos_count = sum(1 for word in positive_words if word in content)
        neg_count = sum(1 for word in negative_words if word in content)
        
        features.append(pos_count / max(len(content.split()), 1))
        features.append(neg_count / max(len(content.split()), 1))
        features.append((pos_count - neg_count) / max(len(content.split()), 1))  # Net sentiment
        
        # Fill to 16 features
        while len(features) < 16:
            features.append(random.uniform(0, 0.05))
        
        return np.array(features[:16])

    def _extract_visual_features_fixed(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract visual-like features with consistent dimension"""
        
        content = getattr(experience, 'content', '')
        features = []
        
        # Visual metaphor features
        visual_words = ['show', 'see', 'view', 'display', 'chart', 'graph', 'image', 'picture']
        color_words = ['red', 'green', 'blue', 'black', 'white', 'bright', 'dark', 'color']
        size_words = ['large', 'small', 'big', 'tiny', 'huge', 'massive', 'micro', 'giant']
        
        features.append(sum(1 for word in visual_words if word in content.lower()) / max(len(content.split()), 1))
        features.append(sum(1 for word in color_words if word in content.lower()) / max(len(content.split()), 1))
        features.append(sum(1 for word in size_words if word in content.lower()) / max(len(content.split()), 1))
        
        # Text complexity as visual complexity
        features.append(len(content) / 1000.0)
        features.append(getattr(experience, 'novelty_score', 0.5))
        features.append(len(set(content.lower().split())) / max(len(content.split()), 1))  # Diversity
        
        # Pattern features (visual patterns in text)
        features.append(content.count('.') / max(len(content), 1))  # Structure
        features.append(content.count(',') / max(len(content), 1))  # Complexity
        features.append(content.count(' ') / max(len(content), 1))  # Spacing
        
        # Shape-like features
        avg_word_length = np.mean([len(word) for word in content.split()]) if content.split() else 0
        features.append(avg_word_length / 10.0)
        features.append(len(content.split()) / 100.0)  # "Height"
        features.append(max([len(word) for word in content.split()]) / 20.0 if content.split() else 0)  # "Width"
        
        # Fill to 16 features
        while len(features) < 16:
            features.append(random.uniform(0, 0.05))
        
        return np.array(features[:16])

    def _extract_audio_features_fixed(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract audio-like features with consistent dimension"""
        
        content = getattr(experience, 'content', '')
        words = content.split()
        features = []
        
        if words:
            # Rhythm features (word length patterns)
            word_lengths = [len(word) for word in words]
            features.append(np.mean(word_lengths) / 10.0)  # Average "note" length
            features.append(np.std(word_lengths) / 5.0)    # Rhythm variation
            
            # "Tone" features (punctuation and emphasis)
            features.append(content.count('!') / max(len(words), 1))  # High pitch
            features.append(content.count('?') / max(len(words), 1))  # Rising tone
            features.append(content.count('.') / max(len(words), 1))  # Falling tone
            features.append(content.count(',') / max(len(words), 1))  # Pause
            
            # Volume features (word frequency and importance)
            common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to']
            uncommon_ratio = 1.0 - (sum(1 for word in words if word.lower() in common_words) / len(words))
            features.append(uncommon_ratio)  # "Loudness"
            
            # Tempo features (sentence structure)
            sentences = content.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            features.append(avg_sentence_length / 20.0)  # Tempo
            
            # Harmony features (word repetition)
            word_freq = Counter(word.lower() for word in words)
            repetition_score = sum(1 for freq in word_freq.values() if freq > 1) / len(word_freq)
            features.append(repetition_score)  # Harmony
            
        else:
            features.extend([0.0] * 9)  # Fill with zeros if no content
        
        # Additional audio-like features
        features.append(getattr(experience, 'novelty_score', 0.5))  # "Novelty" in music
        features.append(len(content) / 500.0)  # Duration
        
        # Fill to 16 features
        while len(features) < 16:
            features.append(random.uniform(0, 0.05))
        
        return np.array(features[:16])
    def _extract_modal_features(self, experience: SensorimotorExperience, 
                            modality: str) -> np.ndarray:
        """Extract modal features with consistent dimensions - FINAL FIX"""
        
        STANDARD_FEATURE_DIM = 16  # Consistent dimension for all modalities
        
        try:
            if modality == 'text':
                features = self._extract_text_features_fixed(experience)
            elif modality == 'temporal':
                features = self._extract_temporal_features_fixed(experience)
            elif modality == 'spatial':
                features = self._extract_spatial_features_fixed(experience)
            elif modality == 'emotional':
                features = self._extract_emotional_features_fixed(experience)
            elif modality == 'visual':
                features = self._extract_visual_features_fixed(experience)
            elif modality == 'audio':
                features = self._extract_audio_features_fixed(experience)
            else:
                # Default features
                features = np.random.uniform(0, 0.1, STANDARD_FEATURE_DIM)
            
            # Ensure consistent dimension
            return self._normalize_feature_dimension(features, STANDARD_FEATURE_DIM)
            
        except Exception as e:
            logger.error(f"Modal feature extraction failed for {modality}: {e}")
            return np.zeros(STANDARD_FEATURE_DIM)

    
    def _extract_text_features(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract text-based features"""
        
        content = getattr(experience, 'content', '')
        
        # Simple bag-of-words features
        words = content.lower().split()
        
        # Feature categories
        financial_words = ['market', 'stock', 'price', 'bitcoin', 'trading', 'investment']
        tech_words = ['technology', 'ai', 'algorithm', 'data', 'system', 'model']
        action_words = ['increase', 'decrease', 'surge', 'decline', 'improve', 'develop']
        
        features = []
        
        # Word category frequencies
        features.append(sum(1 for word in words if word in financial_words) / max(len(words), 1))
        features.append(sum(1 for word in words if word in tech_words) / max(len(words), 1))
        features.append(sum(1 for word in words if word in action_words) / max(len(words), 1))
        
        # Text statistics
        features.append(len(words) / 100.0)  # Normalized length
        features.append(len(set(words)) / max(len(words), 1))  # Lexical diversity
        features.append(np.mean([len(word) for word in words]) if words else 0)  # Avg word length
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])
    
    def _extract_temporal_features(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract temporal features"""
        
        features = []
        
        # Time-based features
        if hasattr(experience, 'timestamp'):
            try:
                dt = datetime.fromisoformat(experience.timestamp.replace('Z', '+00:00'))
                
                # Time of day features
                features.append(dt.hour / 24.0)  # Hour of day
                features.append(dt.weekday() / 6.0)  # Day of week
                features.append(dt.day / 31.0)  # Day of month
                features.append(dt.month / 12.0)  # Month of year
                
            except:
                features.extend([0.5, 0.5, 0.5, 0.5])
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])
        
        # Temporal markers
        if hasattr(experience, 'temporal_markers'):
            markers = experience.temporal_markers
            if markers:
                features.append(np.mean(markers))
                features.append(np.std(markers))
                features.append(len(markers) / 10.0)
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10])
    
    def _extract_spatial_features(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract spatial features"""
        
        features = []
        
        # Domain-based spatial encoding
        domain = getattr(experience, 'domain', 'unknown')
        domain_map = {
            'financial_analysis': [1.0, 0.0, 0.0],
            'research': [0.0, 1.0, 0.0],
            'general': [0.0, 0.0, 1.0]
        }
        
        domain_features = domain_map.get(domain, [0.33, 0.33, 0.33])
        features.extend(domain_features)
        
        # Content-based spatial features (conceptual space)
        content = getattr(experience, 'content', '')
        content_words = content.lower().split()
        
        # Conceptual dimensions
        abstract_words = ['concept', 'idea', 'theory', 'principle', 'approach']
        concrete_words = ['data', 'number', 'result', 'measurement', 'fact']
        
        abstract_score = sum(1 for word in content_words if word in abstract_words) / max(len(content_words), 1)
        concrete_score = sum(1 for word in content_words if word in concrete_words) / max(len(content_words), 1)
        
        features.extend([abstract_score, concrete_score])
        
        # Novelty as spatial dimension
        novelty = getattr(experience, 'novelty_score', 0.5)
        features.append(novelty)
        
        # Pad to fixed size
        while len(features) < 15:
            features.append(0.0)
        
        return np.array(features[:15])
    
    def _extract_emotional_features_array(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract emotional features as array"""
        
        content = getattr(experience, 'content', '').lower()
        
        # Emotion categories
        emotion_keywords = {
            'positive': ['good', 'great', 'excellent', 'success', 'growth', 'improve'],
            'negative': ['bad', 'poor', 'decline', 'loss', 'fail', 'decrease'],
            'neutral': ['stable', 'steady', 'maintain', 'continue', 'same'],
            'excitement': ['surge', 'boom', 'spike', 'jump', 'rocket'],
            'concern': ['risk', 'danger', 'worry', 'caution', 'warning']
        }
        
        features = []
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            features.append(score / max(len(content.split()), 1))
        
        # Add emotional intensity
        if hasattr(experience, 'emotional_features'):
            for emotion in ['valence', 'arousal', 'dominance']:
                features.append(experience.emotional_features.get(emotion, 0.5))
        else:
            features.extend([0.5, 0.5, 0.5])
        
        # Pad to fixed size
        while len(features) < 12:
            features.append(0.0)
        
        return np.array(features[:12])
    
    def _extract_visual_features(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract visual-like features (conceptual visualization)"""
        
        # Since we don't have actual visual data, create conceptual visual features
        features = []
        
        content = getattr(experience, 'content', '')
        
        # Visual metaphor features
        visual_words = ['show', 'see', 'view', 'display', 'chart', 'graph', 'image']
        color_words = ['red', 'green', 'blue', 'black', 'white', 'bright', 'dark']
        size_words = ['large', 'small', 'big', 'tiny', 'huge', 'massive', 'micro']
        
        features.append(sum(1 for word in visual_words if word in content.lower()) / max(len(content.split()), 1))
        features.append(sum(1 for word in color_words if word in content.lower()) / max(len(content.split()), 1))
        features.append(sum(1 for word in size_words if word in content.lower()) / max(len(content.split()), 1))
        
        # Complexity as visual feature
        features.append(len(content) / 1000.0)  # Text complexity
        features.append(getattr(experience, 'novelty_score', 0.5))  # Novelty as visual surprise
        
        # Pad to fixed size
        while len(features) < 8:
            features.append(0.0)
        
        return np.array(features[:8])
    
    def _extract_audio_features(self, experience: SensorimotorExperience) -> np.ndarray:
        """Extract audio-like features (rhythm and tone)"""
        
        # Since we don't have actual audio, create rhythm/tone features from text
        features = []
        
        content = getattr(experience, 'content', '')
        words = content.split()
        
        if words:
            # Rhythm features (word length patterns)
            word_lengths = [len(word) for word in words]
            features.append(np.mean(word_lengths) / 10.0)  # Average word length
            features.append(np.std(word_lengths) / 5.0)    # Word length variation
            
            # "Tone" features (punctuation and emphasis)
            exclamation_count = content.count('!')
            question_count = content.count('?')
            comma_count = content.count(',')
            
            features.append(exclamation_count / max(len(words), 1))  # Excitement
            features.append(question_count / max(len(words), 1))     # Questioning
            features.append(comma_count / max(len(words), 1))        # Complexity
            
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Pad to fixed size
        while len(features) < 8:
            features.append(0.0)
        
        return np.array(features[:8])
    
    def _store_in_modal_index(self, experience_id: str, modality: str, 
                            features: np.ndarray) -> Dict[str, Any]:
        """Store features in modality-specific index"""
        
        if modality not in self.modal_indices:
            self.modal_indices[modality] = {}
        
        self.modal_indices[modality][experience_id] = {
            'features': features,
            'storage_time': time.time(),
            'access_count': 0
        }
        
        return {
            'modality': modality,
            'feature_dimensions': len(features),
            'storage_success': True
        }
    
    def _create_cross_modal_associations(self, experience_id: str, 
                                       modal_features: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Create associations between modalities"""
        
        associations = []
        
        # Create pairwise associations between modalities
        modality_list = list(modal_features.keys())
        
        for i in range(len(modality_list)):
            for j in range(i + 1, len(modality_list)):
                mod1, mod2 = modality_list[i], modality_list[j]
                
                # Calculate association strength
                association_strength = self._calculate_association_strength(
                    modal_features[mod1], modal_features[mod2]
                )
                
                if association_strength > self.association_strength_threshold:
                    associations.append({
                        'modality_1': mod1,
                        'modality_2': mod2,
                        'strength': association_strength,
                        'experience_id': experience_id
                    })
        
        return associations
    def _normalize_feature_dimension(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Normalize feature vector to target dimension"""
        
        if len(features) == target_dim:
            return features
        elif len(features) > target_dim:
            # Truncate to target dimension
            return features[:target_dim]
        else:
            # Pad with interpolated values to avoid zeros
            padding_size = target_dim - len(features)
            
            if len(features) > 0:
                # Use mean and small random variation for padding
                mean_val = np.mean(features)
                std_val = np.std(features) if np.std(features) > 1e-6 else 0.1
                padding = np.random.normal(mean_val, std_val * 0.1, padding_size)
            else:
                # Fallback for empty features
                padding = np.random.uniform(0, 0.1, padding_size)
            
            return np.concatenate([features, padding])
    def _calculate_association_strength(self, features1: np.ndarray, 
                                    features2: np.ndarray) -> float:
        """Calculate association strength between modality features - FIXED"""
        
        try:
            # Handle dimension mismatch by normalizing to same size
            target_dim = 16  # Standardized dimension for all comparisons
            
            # Normalize both feature vectors to same dimension
            features1_norm = self._normalize_feature_dimension(features1, target_dim)
            features2_norm = self._normalize_feature_dimension(features2, target_dim)
            
            # Normalize vectors
            norm1 = features1_norm / (np.linalg.norm(features1_norm) + 1e-10)
            norm2 = features2_norm / (np.linalg.norm(features2_norm) + 1e-10)
            
            # Calculate multiple similarity metrics and combine
            # 1. Cosine similarity (robust)
            cosine_sim = np.dot(norm1, norm2)
            
            # 2. Correlation (if vectors are not constant)
            if np.std(norm1) > 1e-6 and np.std(norm2) > 1e-6:
                correlation = np.corrcoef(norm1, norm2)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = cosine_sim  # Fallback to cosine similarity
            
            # 3. Euclidean distance similarity
            euclidean_dist = np.linalg.norm(norm1 - norm2)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            
            # Combine metrics with weights
            combined_similarity = (
                cosine_sim * 0.5 + 
                abs(correlation) * 0.3 + 
                euclidean_sim * 0.2
            )
            
            # Ensure positive association strength
            association_strength = abs(combined_similarity)
            
            return min(1.0, max(0.0, association_strength))
            
        except Exception as e:
            logger.error(f"Association strength calculation failed: {e}")
            return 0.0
    
    def _add_to_cross_modal_graph(self, experience_id: str, 
                                modal_features: Dict[str, np.ndarray],
                                associations: List[Dict[str, Any]]):
        """Add experience to cross-modal graph"""
        
        # Add experience node
        self.cross_modal_graph.add_node(experience_id, 
                                       modalities=list(modal_features.keys()),
                                       timestamp=time.time())
        
        # Add modality nodes if they don't exist
        for modality in modal_features.keys():
            modality_node = f"mod_{modality}"
            if not self.cross_modal_graph.has_node(modality_node):
                self.cross_modal_graph.add_node(modality_node, type='modality')
            
            # Connect experience to modality
            self.cross_modal_graph.add_edge(experience_id, modality_node, 
                                          type='contains_modality')
        
        # Add cross-modal association edges
        for association in associations:
            mod1_node = f"mod_{association['modality_1']}"
            mod2_node = f"mod_{association['modality_2']}"
            
            self.cross_modal_graph.add_edge(mod1_node, mod2_node,
                                          type='cross_modal_association',
                                          strength=association['strength'],
                                          experience=experience_id)
    
    def _find_modal_matches(self, modality: str, query_features: np.ndarray, 
                          max_matches: int) -> List[Dict[str, Any]]:
        """Find matches within a specific modality"""
        
        if modality not in self.modal_indices:
            return []
        
        matches = []
        
        for experience_id, stored_data in list(self.modal_indices[modality].items()):
            stored_features = stored_data['features']
            
            # Calculate similarity
            similarity = self._calculate_feature_similarity(query_features, stored_features)
            
            if similarity > 0.3:  # Similarity threshold
                matches.append({
                    'experience_id': experience_id,
                    'modality': modality,
                    'similarity': similarity,
                    'stored_data': stored_data
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches[:max_matches]
    
    def _calculate_feature_similarity(self, features1: np.ndarray, 
                                    features2: np.ndarray) -> float:
        """Calculate similarity between feature vectors"""
        
        try:
            # Use cosine similarity
            similarity = 1 - cosine(features1, features2)
            
            # Handle NaN
            if np.isnan(similarity):
                similarity = 0.0
            
            return max(0.0, similarity)
            
        except Exception as e:
            return 0.0
    
    def _combine_cross_modal_matches(self, modal_matches: Dict[str, List[Dict]], 
                                   query_features: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Combine matches from different modalities"""
        
        # Collect all unique experiences
        all_experiences = {}
        
        for modality, matches in modal_matches.items():
            for match in matches:
                exp_id = match['experience_id']
                
                if exp_id not in all_experiences:
                    all_experiences[exp_id] = {
                        'experience_id': exp_id,
                        'modality_matches': {},
                        'total_similarity': 0.0,
                        'modality_count': 0
                    }
                
                all_experiences[exp_id]['modality_matches'][modality] = match
                all_experiences[exp_id]['total_similarity'] += match['similarity']
                all_experiences[exp_id]['modality_count'] += 1
        
        # Calculate cross-modal scores
        cross_modal_results = []
        
        for exp_id, exp_data in all_experiences.items():
            # Average similarity across modalities
            avg_similarity = exp_data['total_similarity'] / exp_data['modality_count']
            
            # Bonus for multiple modality matches
            modality_bonus = (exp_data['modality_count'] - 1) * 0.1
            
            # Cross-modal association bonus
            association_bonus = self._calculate_cross_modal_association_bonus(
                exp_id, list(exp_data['modality_matches'].keys())
            )
            
            final_score = avg_similarity + modality_bonus + association_bonus
            
            cross_modal_results.append({
                'experience_id': exp_id,
                'modality_matches': exp_data['modality_matches'],
                'avg_similarity': avg_similarity,
                'modality_count': exp_data['modality_count'],
                'cross_modal_score': final_score
            })
        
        return cross_modal_results
    
    def _calculate_cross_modal_association_bonus(self, experience_id: str, 
                                             matched_modalities: List[str]) -> float:
        """Calculate bonus for cross-modal associations"""

        if not self.cross_modal_graph.has_node(experience_id):
            return 0.0

        bonus = 0.0

        # Thread-safe access to the graph
        with self.cross_modal_lock:
            try:
                edges_snapshot = list(self.cross_modal_graph.edges(data=True))
            except RuntimeError:
                return 0.0

        for u, v, edge_data in edges_snapshot:
            if (
                edge_data.get('type') == 'cross_modal_association' and
                edge_data.get('experience') == experience_id
            ):
                strength = edge_data.get('strength', 0.0)
                bonus += strength * 0.1

        return min(0.3, bonus)  # Cap the bonus
    
    def _rank_cross_modal_results(self, cross_modal_results: List[Dict], 
                                query_features: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Rank cross-modal results by relevance"""
        
        # Sort by cross-modal score
        ranked_results = sorted(cross_modal_results, 
                              key=lambda x: x['cross_modal_score'], 
                              reverse=True)
        
        # Add additional ranking factors
        for result in ranked_results:
            exp_id = result['experience_id']
            
            # Recency factor (if available)
            recency_factor = 0.5
            if self.cross_modal_graph.has_node(exp_id):
                node_data = self.cross_modal_graph.nodes[exp_id]
                if 'timestamp' in node_data:
                    age = time.time() - node_data['timestamp']
                    recency_factor = np.exp(-age / 86400)  # 1-day half-life
            
            result['recency_factor'] = recency_factor
            
            # Final ranking score
            result['final_score'] = (result['cross_modal_score'] * 0.8 + 
                                   recency_factor * 0.2)
        
        # Re-sort by final score
        ranked_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return ranked_results
    
    def get_cross_modal_statistics(self) -> Dict[str, Any]:
        """Get cross-modal system statistics"""
        
        # Modal index statistics
        modal_stats = {}
        for modality, index in self.modal_indices.items():
            modal_stats[modality] = {
                'stored_experiences': len(index),
                'total_accesses': sum(data['access_count'] for data in index.values())
            }
        
        # Graph statistics
        graph_stats = {
            'total_nodes': self.cross_modal_graph.number_of_nodes(),
            'total_edges': self.cross_modal_graph.number_of_edges(),
            'modality_nodes': len([n for n in self.cross_modal_graph.nodes() if n.startswith('mod_')]),
            'experience_nodes': len([n for n in self.cross_modal_graph.nodes() if not n.startswith('mod_')])
        }
        
        return {
            'supported_modalities': self.modalities,
            'modal_index_stats': modal_stats,
            'cross_modal_graph_stats': graph_stats,
            'association_threshold': self.association_strength_threshold
        }

# ============================================================================
# ADVANCED RETRIEVAL STRATEGIES (NEW)
# ============================================================================

class SemanticSimilarityStrategy(MemoryRetrievalStrategy):
    """Semantic similarity-based retrieval"""
    
    def retrieve(self, query_experience: SensorimotorExperience, 
                memory_store: Dict, max_results: int = 20) -> List[Dict]:
        
        results = []
        
        for item in memory_store.values():
            relevance = self.calculate_relevance_score(query_experience, item)
            
            if relevance > 0.3:
                results.append({
                    'memory_item': item,
                    'relevance_score': relevance,
                    'strategy': 'semantic_similarity'
                })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]
    
    def calculate_relevance_score(self, query: SensorimotorExperience, 
                                candidate: Dict) -> float:
        
        # Extract content
        query_content = getattr(query, 'content', '')
        candidate_content = candidate.get('content', '')
        
        if not query_content or not candidate_content:
            return 0.0
        
        # Calculate semantic similarity (simplified)
        query_tokens = set(query_content.lower().split())
        candidate_tokens = set(candidate_content.lower().split())
        
        if not query_tokens or not candidate_tokens:
            return 0.0
        
        intersection = len(query_tokens & candidate_tokens)
        union = len(query_tokens | candidate_tokens)
        
        return intersection / union if union > 0 else 0.0

class CausalRelevanceStrategy(MemoryRetrievalStrategy):
    """Causal relationship-based retrieval"""
    
    def __init__(self):
        self.causal_indicators = [
            'because', 'since', 'due to', 'caused by', 'resulting in',
            'leads to', 'triggers', 'influences', 'affects', 'impacts'
        ]
    
    def retrieve(self, query_experience: SensorimotorExperience, 
                memory_store: Dict, max_results: int = 20) -> List[Dict]:
        
        results = []
        
        for item in memory_store.values():
            relevance = self.calculate_relevance_score(query_experience, item)
            
            if relevance > 0.4:
                results.append({
                    'memory_item': item,
                    'relevance_score': relevance,
                    'strategy': 'causal_relevance'
                })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]
    
    def calculate_relevance_score(self, query: SensorimotorExperience, 
                                candidate: Dict) -> float:
        
        query_content = getattr(query, 'content', '').lower()
        candidate_content = candidate.get('content', '').lower()
        
        # Look for causal indicators
        query_causal_score = sum(1 for indicator in self.causal_indicators if indicator in query_content)
        candidate_causal_score = sum(1 for indicator in self.causal_indicators if indicator in candidate_content)
        
        if query_causal_score == 0 and candidate_causal_score == 0:
            return 0.0
        
        # Calculate causal relevance
        causal_relevance = (query_causal_score + candidate_causal_score) / (2 * len(self.causal_indicators))
        
        # Boost if both have causal indicators
        if query_causal_score > 0 and candidate_causal_score > 0:
            causal_relevance *= 1.5
        
        return min(1.0, causal_relevance)

class EmotionalResonanceStrategy(MemoryRetrievalStrategy):
    """Emotional similarity-based retrieval"""
    
    def __init__(self):
        self.emotion_keywords = {
            'positive': ['happy', 'excited', 'joy', 'pleased', 'optimistic', 'confident'],
            'negative': ['sad', 'angry', 'frustrated', 'disappointed', 'worried', 'anxious'],
            'neutral': ['calm', 'steady', 'balanced', 'stable', 'measured', 'composed']
        }
    
    def retrieve(self, query_experience: SensorimotorExperience, 
                memory_store: Dict, max_results: int = 20) -> List[Dict]:
        
        results = []
        
        for item in memory_store.values():
            relevance = self.calculate_relevance_score(query_experience, item)
            
            if relevance > 0.5:
                results.append({
                    'memory_item': item,
                    'relevance_score': relevance,
                    'strategy': 'emotional_resonance'
                })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]
    
    def calculate_relevance_score(self, query: SensorimotorExperience, 
                                candidate: Dict) -> float:
        
        query_emotions = self._extract_emotional_features(query)
        candidate_emotions = self._extract_emotional_features_from_dict(candidate)
        
        if not query_emotions or not candidate_emotions:
            return 0.0
        
        # Calculate emotional similarity
        emotional_similarity = 0.0
        
        for emotion_type in ['positive', 'negative', 'neutral']:
            query_score = query_emotions.get(emotion_type, 0)
            candidate_score = candidate_emotions.get(emotion_type, 0)
            
            # Calculate similarity for this emotion type
            if query_score > 0 or candidate_score > 0:
                similarity = 1.0 - abs(query_score - candidate_score)
                emotional_similarity += similarity
        
        return emotional_similarity / 3.0  # Average across emotion types
    
    def _extract_emotional_features(self, experience: SensorimotorExperience) -> Dict[str, float]:
        """Extract emotional features from experience"""
        
        content = getattr(experience, 'content', '').lower()
        emotions = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        total_emotion_words = 0
        
        for emotion_type, keywords in self.emotion_keywords.items():
            emotion_count = sum(1 for keyword in keywords if keyword in content)
            emotions[emotion_type] = emotion_count
            total_emotion_words += emotion_count
        
        # Normalize
        if total_emotion_words > 0:
            for emotion_type in emotions:
                emotions[emotion_type] /= total_emotion_words
        
        return emotions
    
    def _extract_emotional_features_from_dict(self, item: Dict) -> Dict[str, float]:
        """Extract emotional features from memory item"""
        
        content = item.get('content', '').lower()
        emotions = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        total_emotion_words = 0
        
        for emotion_type, keywords in self.emotion_keywords.items():
            emotion_count = sum(1 for keyword in keywords if keyword in content)
            emotions[emotion_type] = emotion_count
            total_emotion_words += emotion_count
        
        # Normalize
        if total_emotion_words > 0:
            for emotion_type in emotions:
                emotions[emotion_type] /= total_emotion_words
        
        return emotions

class GoalRelevanceStrategy(MemoryRetrievalStrategy):
    """Goal and intention-based retrieval"""
    
    def __init__(self):
        self.goal_keywords = [
            'goal', 'objective', 'target', 'aim', 'purpose', 'intention',
            'plan', 'strategy', 'approach', 'method', 'way', 'how to'
        ]
    
    def retrieve(self, query_experience: SensorimotorExperience, 
                memory_store: Dict, max_results: int = 20) -> List[Dict]:
        
        results = []
        
        for item in memory_store.values():
            relevance = self.calculate_relevance_score(query_experience, item)
            
            if relevance > 0.4:
                results.append({
                    'memory_item': item,
                    'relevance_score': relevance,
                    'strategy': 'goal_relevance'
                })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]
    
    def calculate_relevance_score(self, query: SensorimotorExperience, 
                                candidate: Dict) -> float:
        
        query_content = getattr(query, 'content', '').lower()
        candidate_content = candidate.get('content', '').lower()
        
        # Count goal-related keywords
        query_goal_score = sum(1 for keyword in self.goal_keywords if keyword in query_content)
        candidate_goal_score = sum(1 for keyword in self.goal_keywords if keyword in candidate_content)
        
        if query_goal_score == 0 and candidate_goal_score == 0:
            return 0.0
        
        # Calculate goal relevance
        max_possible_score = len(self.goal_keywords) * 2
        total_score = query_goal_score + candidate_goal_score
        
        goal_relevance = total_score / max_possible_score
        
        # Boost if both mention goals
        if query_goal_score > 0 and candidate_goal_score > 0:
            goal_relevance *= 1.3
        
        return min(1.0, goal_relevance)

class AdvancedRetrievalSystem:
    """Multi-strategy retrieval system with ensemble methods"""
    
    def __init__(self):
        self.strategies = {
            'semantic_similarity': SemanticSimilarityStrategy(),
            'causal_relevance': CausalRelevanceStrategy(),
            'emotional_resonance': EmotionalResonanceStrategy(),
            'goal_relevance': GoalRelevanceStrategy()
        }
        
        self.strategy_weights = {
            'semantic_similarity': 0.4,
            'causal_relevance': 0.25,
            'emotional_resonance': 0.20,
            'goal_relevance': 0.15
        }
        
        self.retrieval_history = deque(maxlen=1000)
    
    def multi_strategy_retrieval(self, query_experience: SensorimotorExperience, 
                                memory_store: Dict, max_results: int = 20) -> List[Dict]:
        """Use multiple strategies and ensemble the results"""
        
        strategy_results = {}
        
        # Apply each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                results = strategy.retrieve(query_experience, memory_store, max_results * 2)
                strategy_results[strategy_name] = results
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed: {e}")
                strategy_results[strategy_name] = []
        
        # Ensemble the results
        ensembled_results = self._ensemble_strategy_results(strategy_results, query_experience)
        
        # Rank by combined relevance score
        ranked_results = self._rank_by_combined_relevance(ensembled_results, query_experience)
        
        # Record retrieval statistics
        self._record_retrieval_statistics(query_experience, strategy_results, ranked_results)
        
        return ranked_results[:max_results]
    
    def _ensemble_strategy_results(self, strategy_results: Dict[str, List[Dict]], 
                                 query_experience: SensorimotorExperience) -> List[Dict]:
        """Ensemble results from multiple strategies"""
        
        # Collect all unique memory items
        all_items = {}
        
        for strategy_name, results in strategy_results.items():
            strategy_weight = self.strategy_weights.get(strategy_name, 1.0)
            
            for result in results:
                memory_item = result['memory_item']
                item_id = memory_item.get('item_id', id(memory_item))
                
                if item_id not in all_items:
                    all_items[item_id] = {
                        'memory_item': memory_item,
                        'strategy_scores': {},
                        'combined_score': 0.0,
                        'supporting_strategies': []
                    }
                
                # Add strategy score
                weighted_score = result['relevance_score'] * strategy_weight
                all_items[item_id]['strategy_scores'][strategy_name] = weighted_score
                all_items[item_id]['supporting_strategies'].append(strategy_name)
        
        # Calculate combined scores
        for item_id, item_data in all_items.items():
            total_score = sum(item_data['strategy_scores'].values())
            strategy_count = len(item_data['strategy_scores'])
            
            # Bonus for multiple strategy agreement
            if strategy_count > 1:
                agreement_bonus = (strategy_count - 1) * 0.1
                total_score += agreement_bonus
            
            item_data['combined_score'] = total_score
        
        return list(all_items.values())
    
    def _rank_by_combined_relevance(self, ensembled_results: List[Dict], 
                                  query_experience: SensorimotorExperience) -> List[Dict]:
        """Rank results by combined relevance score"""
        
        # Sort by combined score
        ranked_results = sorted(ensembled_results, 
                              key=lambda x: x['combined_score'], 
                              reverse=True)
        
        # Add additional ranking factors
        for i, result in enumerate(ranked_results):
            memory_item = result['memory_item']
            
            # Recency factor
            if hasattr(memory_item, 'last_access'):
                recency = 1.0 / (1.0 + (time.time() - memory_item['last_access']) / 3600)
                result['recency_factor'] = recency
            else:
                result['recency_factor'] = 0.5
            
            # Importance factor
            importance = memory_item.get('importance_score', 0.5)
            result['importance_factor'] = importance
            
            # Final ranking score
            final_score = (result['combined_score'] * 0.7 + 
                          result['recency_factor'] * 0.2 + 
                          result['importance_factor'] * 0.1)
            
            result['final_ranking_score'] = final_score
        
        # Re-sort by final ranking score
        ranked_results.sort(key=lambda x: x['final_ranking_score'], reverse=True)
        
        return ranked_results
    
    def _record_retrieval_statistics(self, query_experience: SensorimotorExperience,
                                   strategy_results: Dict[str, List[Dict]], 
                                   final_results: List[Dict]):
        """Record retrieval performance statistics"""
        
        retrieval_record = {
            'timestamp': time.time(),
            'query_domain': getattr(query_experience, 'domain', 'unknown'),
            'strategy_performance': {},
            'total_results': len(final_results),
            'ensemble_effectiveness': 0.0
        }
        
        # Record strategy performance
        for strategy_name, results in strategy_results.items():
            retrieval_record['strategy_performance'][strategy_name] = {
                'result_count': len(results),
                'avg_relevance': np.mean([r['relevance_score'] for r in results]) if results else 0.0,
                'max_relevance': max([r['relevance_score'] for r in results]) if results else 0.0
            }
        
        # Calculate ensemble effectiveness
        if final_results:
            strategy_counts = [len(r['supporting_strategies']) for r in final_results]
            avg_strategy_support = np.mean(strategy_counts)
            retrieval_record['ensemble_effectiveness'] = avg_strategy_support / len(self.strategies)
        
        self.retrieval_history.append(retrieval_record)
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retrieval system statistics"""
        
        if not self.retrieval_history:
            return {'status': 'no_retrieval_history'}
        
        # Calculate overall statistics
        total_retrievals = len(self.retrieval_history)
        
        strategy_performance = {}
        for strategy_name in self.strategies.keys():
            performances = []
            for record in self.retrieval_history:
                if strategy_name in record['strategy_performance']:
                    performances.append(record['strategy_performance'][strategy_name]['avg_relevance'])
            
            strategy_performance[strategy_name] = {
                'avg_performance': np.mean(performances) if performances else 0.0,
                'usage_frequency': len(performances) / total_retrievals
            }
        
        ensemble_effectiveness = np.mean([r['ensemble_effectiveness'] for r in self.retrieval_history])
        
        return {
            'total_retrievals': total_retrievals,
            'strategy_performance': strategy_performance,
            'ensemble_effectiveness': ensemble_effectiveness,
            'strategy_weights': self.strategy_weights
        }

# ============================================================================
# EM-LLM EPISODIC MEMORY ENGINE (COMPLETE ORIGINAL + ENHANCED)
# ============================================================================

class EnhancedEpisodicMemoryEngine:
    """Enhanced EM-LLM inspired episodic memory system with ALL original features + advanced enhancements"""
    
    def __init__(self, model_name: str = "gemma3n:e4b", max_episodes: int = 10000):
        self.model_name = model_name
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
        self.episode_embeddings = []
        self.surprise_detector = BayesianSurpriseDetector()
        self.boundary_refiner = AdvancedBoundaryRefiner()
        
        # Enhanced buffers for retrieval (inspired by EM-LLM paper)
        self.similarity_buffer_size = 15  # k_s similar episodes
        self.contiguity_buffer_size = 8   # k_c contiguous episodes
        self.contiguity_buffer = deque(maxlen=self.contiguity_buffer_size)
        
        # NEW: Advanced memory management integration
        self.token_manager = TokenLevelContextManager(
            context_window=self._get_context_window(model_name)
        )
        self.compression_system = MemoryCompressionSystem()
        self.cross_modal_system = CrossModalMemorySystem()
        self.hierarchical_memory = HierarchicalMemorySystem()
        self.advanced_retrieval = AdvancedRetrievalSystem()
        
        # Episodic statistics (ORIGINAL)
        self.total_tokens_stored = 0
        self.episode_boundaries = []
        self.surprise_threshold = 0.6
        
        # Advanced tracking (NEW)
        self.retrieval_performance = deque(maxlen=1000)
        self.compression_stats = defaultdict(int)
        
    def _get_context_window(self, model_name: str) -> int:
        """Get context window size for model"""
        if "gemma3n" in model_name:
            return 32000  # 32K context window
        elif "deepseek" in model_name:
            return 4000   # 4K context window
        else:
            return 8000   # Default
        
    def detect_episode_boundary(self, experience: SensorimotorExperience, 
                               cortical_result: Dict[str, Any]) -> bool:
        """Enhanced episode boundary detection with ALL original features + advanced enhancements"""
        
        # ORIGINAL: Calculate surprise based on prediction accuracy and novelty
        prediction_accuracy = cortical_result.get('prediction_accuracy', 0.5)
        novelty_score = experience.novelty_score
        consensus_confidence = cortical_result.get('consensus', {}).get('overall_confidence', 0.5)
        
        # ORIGINAL: Enhanced Bayesian surprise with additional factors
        surprise_score = (1.0 - prediction_accuracy) * 0.4 + novelty_score * 0.4 + (1.0 - consensus_confidence) * 0.2
        
        # ORIGINAL: Additional surprise factors
        content_complexity = len(experience.content.split()) / 100.0  # Normalize
        domain_shift = self._detect_domain_shift(experience)
        
        # NEW: Token-level surprise
        token_surprise = self._calculate_token_level_surprise(experience)
        
        # NEW: Cross-modal surprise
        cross_modal_surprise = self._calculate_cross_modal_surprise(experience)
        
        # NEW: Hierarchical memory surprise
        hierarchical_surprise = self._calculate_hierarchical_memory_surprise(experience)
        
        total_surprise = (surprise_score + content_complexity * 0.1 + domain_shift * 0.3 + 
                         token_surprise * 0.1 + cross_modal_surprise * 0.1 + hierarchical_surprise * 0.1)
        
        is_boundary = total_surprise > self.surprise_threshold
        
        if is_boundary:
            print(f"🔥 Enhanced episode boundary detected! Surprise: {total_surprise:.3f}")
            self.episode_boundaries.append({
                'timestamp': experience.timestamp,
                'surprise_score': total_surprise,
                'content_preview': experience.content[:100] + "...",
                'factors': {
                    'prediction': 1.0 - prediction_accuracy,
                    'novelty': novelty_score,
                    'consensus': 1.0 - consensus_confidence,
                    'complexity': content_complexity,
                    'domain_shift': domain_shift,
                    'token_surprise': token_surprise,
                    'cross_modal_surprise': cross_modal_surprise,
                    'hierarchical_surprise': hierarchical_surprise
                }
            })
        
        return is_boundary
    
    def _calculate_token_level_surprise(self, experience: SensorimotorExperience) -> float:
        """Calculate surprise at token level (NEW)"""
        
        # Convert experience to tokens
        experience_tokens = experience.content.split()
        
        # Check token-level novelty
        token_surprise = 0.0
        
        for token in experience_tokens:
            # Check if token is in vocabulary
            if token not in self.token_manager.token_importance_scores:
                token_surprise += 0.1  # Novel token
            else:
                # Check if token is used in unexpected context
                importance = self.token_manager.token_importance_scores[token]
                frequency = self.token_manager.token_access_frequency[token]
                
                if importance > 0.8 and frequency < 2:  # Important but rare
                    token_surprise += 0.05
        
        return min(1.0, token_surprise / len(experience_tokens)) if experience_tokens else 0.0
    
    def _calculate_cross_modal_surprise(self, experience: SensorimotorExperience) -> float:
        """Calculate cross-modal surprise (NEW)"""
        
        # Extract modality features
        modal_features = {}
        modalities = ['text', 'temporal', 'spatial', 'emotional']
        
        for modality in modalities:
            features = self.cross_modal_system._extract_modal_features(experience, modality)
            if features is not None:
                modal_features[modality] = features
        
        if len(modal_features) < 2:
            return 0.0
        
        # Calculate cross-modal consistency
        modality_list = list(modal_features.keys())
        inconsistencies = []
        
        for i in range(len(modality_list)):
            for j in range(i + 1, len(modality_list)):
                mod1, mod2 = modality_list[i], modality_list[j]
                consistency = self._calculate_modal_consistency(
                    modal_features[mod1], modal_features[mod2]
                )
                inconsistencies.append(1.0 - consistency)
        
        # High inconsistency = high surprise
        return np.mean(inconsistencies) if inconsistencies else 0.0
    
    def _calculate_hierarchical_memory_surprise(self, experience: SensorimotorExperience) -> float:
        """Calculate surprise based on hierarchical memory predictions (NEW)"""
        
        # Get predictions from hierarchical memory
        recent_memories = self.hierarchical_memory.retrieve_memories(experience, max_memories=5)
        
        if not recent_memories:
            return 0.5  # Medium surprise if no memories
        
        # Calculate expectation based on recent memories
        expected_domain = Counter()
        expected_novelty = []
        
        for memory_result in recent_memories:
            memory_item = memory_result['memory_item']
            stored_experience = memory_item['experience']
            
            expected_domain[stored_experience.domain] += 1
            expected_novelty.append(stored_experience.novelty_score)
        
        # Domain surprise
        most_common_domain = expected_domain.most_common(1)[0][0] if expected_domain else experience.domain
        domain_surprise = 0.8 if experience.domain != most_common_domain else 0.2
        
        # Novelty surprise
        avg_expected_novelty = np.mean(expected_novelty) if expected_novelty else 0.5
        novelty_surprise = abs(experience.novelty_score - avg_expected_novelty)
        
        return (domain_surprise + novelty_surprise) / 2.0
    
    def _calculate_modal_consistency(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate consistency between modality features (NEW)"""
        
        try:
            # Normalize features
            norm1 = features1 / (np.linalg.norm(features1) + 1e-10)
            norm2 = features2 / (np.linalg.norm(features2) + 1e-10)
            
            # Calculate correlation
            correlation = np.corrcoef(norm1, norm2)[0, 1]
            
            # Handle NaN
            if np.isnan(correlation):
                correlation = 0.0
            
            return abs(correlation)
            
        except Exception as e:
            return 0.5
    
    def _detect_domain_shift(self, experience: SensorimotorExperience) -> float:
        """Enhanced domain shift detection (ORIGINAL enhanced)"""
        if not self.episodes:
            return 0.0
        
        # Simple domain shift detection based on content similarity
        recent_episode = self.episodes[-1] if self.episodes else None
        if recent_episode:
            # Enhanced keyword-based domain detection
            current_domain_words = set(experience.content.lower().split())
            recent_domain_words = set(recent_episode['content'].lower().split())
            
            intersection = len(current_domain_words & recent_domain_words)
            union = len(current_domain_words | recent_domain_words)
            
            if union > 0:
                similarity = intersection / union
                return 1.0 - similarity
        
        return 0.0
    
    def store_episode(self, experience: SensorimotorExperience, 
                     cortical_result: Dict[str, Any], 
                     identity_result: Dict[str, Any],
                     is_boundary: bool = False):
        """Enhanced episode storage with ALL original features + advanced memory management"""
        
        # ORIGINAL: Create enhanced episodic memory entry
        episode = {
            'episode_id': f"ep_{uuid.uuid4().hex[:8]}",
            'timestamp': experience.timestamp,
            'content': experience.content,
            'domain': experience.domain,
            'experience_id': experience.experience_id,
            'novelty_score': experience.novelty_score,
            'is_boundary': is_boundary,
            
            # ORIGINAL: Cortical processing results
            'cortical_patterns': cortical_result.get('consensus', {}).get('consensus_patterns', {}),
            'prediction_accuracy': cortical_result.get('prediction_accuracy', 0.5),
            'domain_expertise': cortical_result.get('domain_expertise_level', 0.5),
            
            # ORIGINAL: Identity formation results
            'personality_state': identity_result.get('personality_state', {}),
            'narrative_themes': identity_result.get('identity_analysis', {}).get('narrative_connection', ''),
            'identity_coherence': identity_result.get('coherence_assessment', {}).get('overall_coherence', 0.5),
            
            # ORIGINAL: Representative tokens (for retrieval)
            'representative_tokens': self._extract_representative_tokens(experience.content),
            'embedding_vector': self._generate_episode_embedding(experience, cortical_result, identity_result)
        }
        
        # NEW: Enhanced storage with advanced features
        
        # 1. Token-level processing
        experience_tokens = experience.content.split()
        token_context = self.token_manager.process_tokens_with_memory(
            experience_tokens,
            experience_context={'content': experience.content, 'domain': experience.domain}
        )
        episode['token_context_size'] = len(token_context)
        
        # 2. Cross-modal storage
        cross_modal_result = self.cross_modal_system.store_cross_modal_experience(experience)
        episode['cross_modal_associations'] = cross_modal_result.get('cross_modal_associations', 0)
        
        # 3. Hierarchical memory storage
        hierarchical_result = self.hierarchical_memory.store_experience(experience)
        episode['hierarchical_storage_level'] = hierarchical_result['storage_level']
        
        # 4. Store episode (ORIGINAL)
        self.episodes.append(episode)
        self.episode_embeddings.append(episode['embedding_vector'])
        self.total_tokens_stored += len(experience.content.split())
        
        # 5. Update contiguity buffer (ORIGINAL)
        self.contiguity_buffer.append(episode)
        
        # 6. Enhanced boundary refinement (NEW)
        if len(self.episodes) > 5:
            recent_episodes = list(self.episodes)[-10:]
            refined_episodes, refinement_metrics = self.boundary_refiner.refine_boundaries_with_graph_metrics(recent_episodes)
            
            # Update episodes with refined boundaries
            for i, refined_episode in enumerate(refined_episodes):
                if i < len(recent_episodes):
                    original_idx = len(self.episodes) - len(recent_episodes) + i
                    if 0 <= original_idx < len(self.episodes):
                        self.episodes[original_idx]['is_boundary'] = refined_episode.get('is_boundary', False)
        
        # 7. Periodic compression (NEW)
        if len(self.episodes) % 50 == 0:  # Every 50 episodes
            self._perform_periodic_compression()
    
    def _perform_periodic_compression(self):
        """Perform periodic compression of episode sequences (NEW)"""
        
        if len(self.episodes) < 20:
            return
        
        # Select episodes for compression (older episodes)
        episodes_to_compress = list(self.episodes)[-50:-20]  # Episodes 20-50 from end
        
        if len(episodes_to_compress) >= 10:
            try:
                compressed_block, compression_stats = self.compression_system.compress_episode_sequence(episodes_to_compress)
                
                if compressed_block:
                    # Store compression metadata
                    self.compression_stats['blocks_created'] += 1
                    self.compression_stats['episodes_compressed'] += len(episodes_to_compress)
                    self.compression_stats['compression_ratio'] += compressed_block.compression_ratio
                    
                    logger.info(f"Compressed {len(episodes_to_compress)} episodes into block {compressed_block.block_id}")
                    
            except Exception as e:
                logger.error(f"Episode compression failed: {e}")
    
    def retrieve_episodic_context(self, query_experience: SensorimotorExperience, 
                                max_context_tokens: int = 8000) -> Dict[str, Any]:
        """Enhanced episodic context retrieval with ALL original features + advanced enhancements"""
        
        retrieval_start_time = time.time()
        
        if not self.episodes:
            return {'episodes': [], 'context_summary': 'No episodic memory available'}
        
        # ORIGINAL: Generate enhanced query embedding
        query_embedding = self._generate_query_embedding(query_experience)
        
        # ORIGINAL: Enhanced Stage 1: Multi-strategy similarity retrieval
        similarity_episodes = self._enhanced_similarity_retrieval(query_embedding, query_experience, self.similarity_buffer_size)
        
        # ORIGINAL: Enhanced Stage 2: Temporally contiguous retrieval
        contiguity_episodes = list(self.contiguity_buffer)
        
        # NEW: Stage 3: Cross-modal retrieval
        cross_modal_episodes = self._cross_modal_episode_retrieval(query_experience)
        
        # NEW: Stage 4: Token-level guided retrieval
        token_guided_episodes = self._token_guided_retrieval(query_experience)
        
        # NEW: Stage 5: Hierarchical memory retrieval
        hierarchical_episodes = self._hierarchical_memory_retrieval(query_experience)
        
        # NEW: Stage 6: Advanced multi-strategy retrieval
        advanced_episodes = self._advanced_strategy_retrieval(query_experience)
        
        # Enhanced combination and ranking (ENHANCED)
        combined_episodes = self._enhanced_combine_and_rank_episodes(
            similarity_episodes, contiguity_episodes, cross_modal_episodes, 
            token_guided_episodes, hierarchical_episodes, advanced_episodes, query_experience
        )
        
        # ORIGINAL: Select episodes within token budget
        selected_episodes = self._select_episodes_within_budget(combined_episodes, max_context_tokens)
        
        # ORIGINAL: Generate enhanced context summary
        context_summary = self._generate_enhanced_context_summary(selected_episodes)
        
        # Record retrieval performance (ENHANCED)
        retrieval_time = time.time() - retrieval_start_time
        self.retrieval_performance.append({
            'timestamp': time.time(),
            'retrieval_time': retrieval_time,
            'episodes_retrieved': len(selected_episodes),
            'strategies_used': 6,  # Updated count
            'query_domain': query_experience.domain
        })
        
        return {
            'episodes': selected_episodes,
            'context_summary': context_summary,
            'total_episodes_retrieved': len(selected_episodes),
            'similarity_count': len(similarity_episodes),
            'contiguity_count': len(contiguity_episodes),
            'cross_modal_count': len(cross_modal_episodes),
            'token_guided_count': len(token_guided_episodes),
            'hierarchical_count': len(hierarchical_episodes),
            'advanced_count': len(advanced_episodes),
            'total_memory_episodes': len(self.episodes),
            'memory_span_tokens': self.total_tokens_stored,
            'retrieval_time': retrieval_time,
            'advanced_features': {
                'graph_refinement': True,
                'cross_modal_integration': True,
                'token_level_management': True,
                'hierarchical_memory': True,
                'advanced_strategies': True,
                'compression_active': self.compression_stats['blocks_created'] > 0
            }
        }
    
    def _enhanced_similarity_retrieval(self, query_embedding: np.ndarray, 
                                     query_experience: SensorimotorExperience,
                                     k: int) -> List[Dict]:
        """Enhanced similarity-based episode retrieval (ORIGINAL enhanced)"""
        
        if not self.episode_embeddings:
            return []
        
        similarities = []
        for i, episode_embedding in enumerate(self.episode_embeddings):
            # Basic similarity
            similarity = np.dot(query_embedding, episode_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(episode_embedding)
            )
            
            # Enhanced similarity with additional factors
            episode = self.episodes[i]
            
            # Domain boost
            domain_boost = 0.2 if episode['domain'] == query_experience.domain else 0.0
            
            # Novelty alignment
            novelty_alignment = 1.0 - abs(episode['novelty_score'] - query_experience.novelty_score)
            novelty_boost = novelty_alignment * 0.1
            
            # Recency boost
            episode_age = time.time() - time.mktime(time.strptime(episode['timestamp'][:19], '%Y-%m-%dT%H:%M:%S'))
            recency_boost = np.exp(-episode_age / 86400) * 0.1  # 1-day half-life
            
            enhanced_similarity = similarity + domain_boost + novelty_boost + recency_boost
            similarities.append((enhanced_similarity, i))
        
        # Get top-k most similar episodes
        similarities.sort(reverse=True)
        top_episodes = []
        
        for similarity_score, episode_idx in similarities[:k]:
            episode = dict(self.episodes[episode_idx])  # Create copy
            episode['similarity_score'] = similarity_score
            episode['retrieval_reason'] = 'enhanced_similarity'
            top_episodes.append(episode)
        
        return top_episodes
    
    def _cross_modal_episode_retrieval(self, query_experience: SensorimotorExperience) -> List[Dict]:
        """Retrieve episodes using cross-modal associations (NEW)"""
        
        # Get cross-modal matches
        cross_modal_results = self.cross_modal_system.retrieve_cross_modal(query_experience, max_results=10)
        
        episodes = []
        for result in cross_modal_results:
            exp_id = result['experience_id']
            
            # Find corresponding episode
            for episode in self.episodes:
                if episode.get('experience_id') == exp_id:
                    episode_copy = dict(episode)
                    episode_copy['similarity_score'] = result['final_score']
                    episode_copy['retrieval_reason'] = 'cross_modal'
                    episode_copy['modality_matches'] = result.get('modality_matches', {})
                    episodes.append(episode_copy)
                    break
        
        return episodes
    
    def _token_guided_retrieval(self, query_experience: SensorimotorExperience) -> List[Dict]:
        """Retrieve episodes guided by token-level analysis (NEW)"""
        
        # Get relevant tokens from token manager
        query_tokens = query_experience.content.split()
        relevant_tokens = self.token_manager.retrieve_relevant_tokens(
            {'content': query_experience.content, 'domain': query_experience.domain}
        )
        
        if not relevant_tokens:
            return []
        
        # Find episodes containing relevant tokens
        episodes = []
        relevant_token_set = set(relevant_tokens)
        
        for episode in list(self.episodes)[-20:]:  # Search recent episodes
            episode_tokens = set(episode['content'].split())
            token_overlap = len(episode_tokens & relevant_token_set)
            
            if token_overlap > 0:
                episode_copy = dict(episode)
                episode_copy['similarity_score'] = token_overlap / len(relevant_token_set)
                episode_copy['retrieval_reason'] = 'token_guided'
                episode_copy['token_overlap'] = token_overlap
                episodes.append(episode_copy)
        
        # Sort by token overlap
        episodes.sort(key=lambda x: x['similarity_score'], reverse=True)
        return episodes[:5]  # Top 5 token-guided episodes
    
    def _hierarchical_memory_retrieval(self, query_experience: SensorimotorExperience) -> List[Dict]:
        """Retrieve episodes using hierarchical memory system (NEW)"""
        
        # Get memories from hierarchical system
        hierarchical_memories = self.hierarchical_memory.retrieve_memories(query_experience, max_memories=10)
        
        episodes = []
        for memory_result in hierarchical_memories:
            memory_item = memory_result['memory_item']
            exp_id = memory_item['experience'].experience_id
            
            # Find corresponding episode
            for episode in self.episodes:
                if episode.get('experience_id') == exp_id:
                    episode_copy = dict(episode)
                    episode_copy['similarity_score'] = memory_result['relevance_score']
                    episode_copy['retrieval_reason'] = 'hierarchical'
                    episode_copy['memory_level'] = memory_result['memory_level']
                    episodes.append(episode_copy)
                    break
        
        return episodes
    
    def _advanced_strategy_retrieval(self, query_experience: SensorimotorExperience) -> List[Dict]:
        """Retrieve episodes using advanced multi-strategy system (NEW)"""
        
        # Create a memory store from episodes for advanced retrieval
        episode_store = {}
        for episode in list(self.episodes)[-50:]:  # Recent episodes
            episode_store[episode['episode_id']] = episode
        
        if not episode_store:
            return []
        
        # Use advanced retrieval system
        advanced_results = self.advanced_retrieval.multi_strategy_retrieval(
            query_experience, episode_store, max_results=8
        )
        
        episodes = []
        for result in advanced_results:
            episode = result['memory_item']
            episode_copy = dict(episode)
            episode_copy['similarity_score'] = result['final_ranking_score']
            episode_copy['retrieval_reason'] = 'advanced_strategies'
            episode_copy['supporting_strategies'] = result['supporting_strategies']
            episodes.append(episode_copy)
        
        return episodes
    
    def _enhanced_combine_and_rank_episodes(self, similarity_episodes: List[Dict], 
                                          contiguity_episodes: List[Dict],
                                          cross_modal_episodes: List[Dict],
                                          token_guided_episodes: List[Dict],
                                          hierarchical_episodes: List[Dict],
                                          advanced_episodes: List[Dict],
                                          query_experience: SensorimotorExperience) -> List[Dict]:
        """Enhanced episode combination and ranking with ALL strategies (ENHANCED)"""
        
        # Add contiguity episodes with metadata
        enhanced_contiguity = []
        for episode in contiguity_episodes:
            enhanced_episode = dict(episode)
            enhanced_episode['retrieval_reason'] = 'contiguity'
            enhanced_episode['similarity_score'] = 0.7  # Default relevance for contiguous
            enhanced_contiguity.append(enhanced_episode)
        
        # Combine all episodes
        all_episodes = (similarity_episodes + enhanced_contiguity + cross_modal_episodes + 
                       token_guided_episodes + hierarchical_episodes + advanced_episodes)
        
        # Remove duplicates by episode_id
        seen_ids = set()
        unique_episodes = []
        episode_sources = {}  # Track which sources contributed to each episode
        
        for episode in all_episodes:
            episode_id = episode['episode_id']
            
            if episode_id not in seen_ids:
                unique_episodes.append(episode)
                seen_ids.add(episode_id)
                episode_sources[episode_id] = [episode['retrieval_reason']]
            else:
                # Add retrieval reason to existing episode
                for existing_episode in unique_episodes:
                    if existing_episode['episode_id'] == episode_id:
                        episode_sources[episode_id].append(episode['retrieval_reason'])
                        # Boost similarity score for multi-source episodes
                        existing_episode['similarity_score'] = max(
                            existing_episode['similarity_score'], 
                            episode['similarity_score']
                        )
                        break
        
        # Enhanced ranking with multiple factors
        current_time = time.time()
        
        for episode in unique_episodes:
            episode_id = episode['episode_id']
            episode_time = time.mktime(time.strptime(episode['timestamp'][:19], '%Y-%m-%dT%H:%M:%S'))
            
            # Recency score
            recency_score = 1.0 / (1.0 + (current_time - episode_time) / 86400)  # Decay over days
            
            # Boundary importance
            boundary_bonus = 0.2 if episode.get('is_boundary', False) else 0.0
            
            # Multi-source bonus
            source_count = len(episode_sources.get(episode_id, []))
            multi_source_bonus = (source_count - 1) * 0.1
            
            # Identity coherence boost
            identity_coherence = episode.get('identity_coherence', 0.5)
            coherence_boost = identity_coherence * 0.1
            
            # Cross-modal association boost
            cross_modal_bonus = 0.15 if 'cross_modal' in episode_sources.get(episode_id, []) else 0.0
            
            # NEW: Advanced strategy bonus
            advanced_bonus = 0.12 if 'advanced_strategies' in episode_sources.get(episode_id, []) else 0.0
            
            # NEW: Hierarchical memory bonus
            hierarchical_bonus = 0.10 if 'hierarchical' in episode_sources.get(episode_id, []) else 0.0
            
            episode['combined_score'] = (
                episode['similarity_score'] * 0.35 +
                recency_score * 0.15 +
                identity_coherence * 0.15 +
                boundary_bonus +
                multi_source_bonus +
                coherence_boost +
                cross_modal_bonus +
                advanced_bonus +
                hierarchical_bonus
            )
            
            episode['ranking_factors'] = {
                'similarity': episode['similarity_score'],
                'recency': recency_score,
                'boundary': boundary_bonus,
                'multi_source': multi_source_bonus,
                'coherence': coherence_boost,
                'cross_modal': cross_modal_bonus,
                'advanced': advanced_bonus,
                'hierarchical': hierarchical_bonus,
                'sources': episode_sources.get(episode_id, [])
            }
        
        # Sort by combined score
        unique_episodes.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return unique_episodes
    
    def _select_episodes_within_budget(self, episodes: List[Dict], max_tokens: int) -> List[Dict]:
        """Enhanced episode selection within token budget (ORIGINAL enhanced)"""
        selected = []
        total_tokens = 0
        
        for episode in episodes:
            episode_tokens = len(episode['content'].split())
            
            # Add token context size if available
            token_context_size = episode.get('token_context_size', 0)
            total_episode_tokens = episode_tokens + token_context_size
            
            if total_tokens + total_episode_tokens <= max_tokens:
                selected.append(episode)
                total_tokens += total_episode_tokens
            else:
                # Try to fit partial episode if it's highly relevant
                if episode.get('combined_score', 0) > 0.8 and len(selected) < 3:
                    # Truncate episode content to fit budget
                    remaining_tokens = max_tokens - total_tokens
                    if remaining_tokens > 50:  # Minimum meaningful content
                        episode_copy = dict(episode)
                        words = episode['content'].split()
                        truncated_content = ' '.join(words[:remaining_tokens])
                        episode_copy['content'] = truncated_content + "... [truncated]"
                        episode_copy['truncated'] = True
                        selected.append(episode_copy)
                        total_tokens = max_tokens
                break
        
        return selected
    
    def _generate_enhanced_context_summary(self, episodes: List[Dict]) -> str:
        """Generate enhanced summary of episodic context (ORIGINAL enhanced)"""
        if not episodes:
            return "No relevant episodic memories found."
        
        total_episodes = len(episodes)
        
        # Count by retrieval reason
        retrieval_counts = defaultdict(int)
        for episode in episodes:
            reasons = episode.get('ranking_factors', {}).get('sources', [episode.get('retrieval_reason', 'unknown')])
            for reason in reasons:
                retrieval_counts[reason] += 1
        
        boundary_count = len([ep for ep in episodes if ep.get('is_boundary', False)])
        truncated_count = len([ep for ep in episodes if ep.get('truncated', False)])
        
        # Time span analysis
        timespan = "recent interactions"
        if episodes:
            timestamps = [ep['timestamp'] for ep in episodes if ep.get('timestamp')]
            if len(timestamps) > 1:
                first_time = min(timestamps)
                last_time = max(timestamps)
                timespan = f"spanning from {first_time[:10]} to {last_time[:10]}"
        
        # Advanced features summary
        advanced_features = []
        if any('cross_modal' in ep.get('ranking_factors', {}).get('sources', []) for ep in episodes):
            advanced_features.append("cross-modal associations")
        if any('token_guided' in ep.get('ranking_factors', {}).get('sources', []) for ep in episodes):
            advanced_features.append("token-level guidance")
        if any('hierarchical' in ep.get('ranking_factors', {}).get('sources', []) for ep in episodes):
            advanced_features.append("hierarchical memory")
        if any('advanced_strategies' in ep.get('ranking_factors', {}).get('sources', []) for ep in episodes):
            advanced_features.append("multi-strategy retrieval")
        if boundary_count > 0:
            advanced_features.append("episode boundary analysis")
        
        summary = f"Retrieved {total_episodes} relevant episodes {timespan}. "
        
        # Retrieval method breakdown
        method_parts = []
        for reason, count in retrieval_counts.items():
            method_parts.append(f"{count} by {reason.replace('_', ' ')}")
        
        if method_parts:
            summary += f"Sources: {', '.join(method_parts)}. "
        
        if boundary_count > 0:
            summary += f"{boundary_count} episodes mark significant boundaries. "
        
        if truncated_count > 0:
            summary += f"{truncated_count} episodes truncated to fit context. "
        
        if advanced_features:
            summary += f"Enhanced with: {', '.join(advanced_features)}."
        
        return summary
    
    def _extract_representative_tokens(self, content: str) -> List[str]:
        """Enhanced representative token extraction (ORIGINAL enhanced)"""
        words = content.lower().split()
        word_freq = {}
        
        # Enhanced common words list
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'
        }
        
        # Domain-specific important words
        domain_words = {
            'market', 'price', 'stock', 'bitcoin', 'trading', 'investment', 'analysis', 'research',
            'technology', 'ai', 'algorithm', 'data', 'model', 'system', 'neural', 'learning'
        }
        
        for word in words:
            if word not in common_words and len(word) > 2:
                # Boost domain-specific words
                weight = 2.0 if word in domain_words else 1.0
                word_freq[word] = word_freq.get(word, 0) + weight
        
        # Return top representative words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:7]]  # Increased to 7 tokens
    
    def _generate_episode_embedding(self, experience: SensorimotorExperience, 
                                  cortical_result: Dict, identity_result: Dict) -> np.ndarray:
        """Enhanced episode embedding generation (ORIGINAL enhanced)"""
        
        # Enhanced content features
        content_features = self._encode_enhanced_content_features(experience.content)
        
        # Enhanced cortical features
        cortical_features = self._encode_enhanced_cortical_features(cortical_result)
        
        # Enhanced identity features
        identity_features = self._encode_enhanced_identity_features(identity_result)
        
        # NEW: Cross-modal features
        cross_modal_features = self._encode_cross_modal_features(experience)
        
        # NEW: Token-level features
        token_features = self._encode_token_level_features(experience)
        
        # NEW: Hierarchical memory features
        hierarchical_features = self._encode_hierarchical_features(experience)
        
        # Concatenate and normalize
        combined = np.concatenate([
            content_features, cortical_features, identity_features, 
            cross_modal_features, token_features, hierarchical_features
        ])
        return combined / (np.linalg.norm(combined) + 1e-10)
    
    def _encode_enhanced_content_features(self, content: str) -> np.ndarray:
        """Enhanced content feature encoding (ORIGINAL enhanced)"""
        words = content.lower().split()
        
        # Enhanced domain categorization
        financial_words = ['market', 'stock', 'price', 'bitcoin', 'trading', 'investment', 'crypto', 'economy', 'fed', 'rate']
        tech_words = ['technology', 'ai', 'algorithm', 'data', 'system', 'model', 'neural', 'research', 'quantum', 'computing']
        action_words = ['increase', 'decrease', 'surge', 'decline', 'improve', 'develop', 'analyze', 'predict', 'announce', 'report']
        sentiment_words = ['positive', 'negative', 'bullish', 'bearish', 'optimistic', 'pessimistic', 'confident', 'uncertain']
        
        features = []
        word_count = len(words)
        
        # Domain feature calculation
        features.append(sum(1 for word in words if word in financial_words) / max(word_count, 1))
        features.append(sum(1 for word in words if word in tech_words) / max(word_count, 1))
        features.append(sum(1 for word in words if word in action_words) / max(word_count, 1))
        features.append(sum(1 for word in words if word in sentiment_words) / max(word_count, 1))
        
        # Enhanced text statistics
        features.append(word_count / 100.0)  # Content length
        features.append(len(set(words)) / max(word_count, 1))  # Lexical diversity
        features.append(np.mean([len(word) for word in words]) if words else 0)  # Avg word length
        features.append(len([word for word in words if word.isupper()]) / max(word_count, 1))  # Acronym density
        features.append(content.count('!') / max(word_count, 1))  # Excitement markers
        features.append(content.count('?') / max(word_count, 1))  # Question markers
        
        # Pad to fixed size
        while len(features) < 25:
            features.append(random.uniform(0, 0.05))
        
        return np.array(features[:25])
    
    def _encode_enhanced_cortical_features(self, cortical_result: Dict) -> np.ndarray:
        """Enhanced cortical processing feature encoding (ORIGINAL enhanced)"""
        features = []
        
        # Core cortical metrics
        features.append(cortical_result.get('prediction_accuracy', 0.5))
        features.append(cortical_result.get('domain_expertise_level', 0.5))
        
        consensus = cortical_result.get('consensus', {})
        features.append(consensus.get('overall_confidence', 0.5))
        features.append(len(consensus.get('consensus_patterns', {})) / 10.0)
        features.append(consensus.get('agreement_level', 0.5))
        
        # Enhanced cortical features
        features.append(cortical_result.get('episodic_integration_quality', 0.5))
        features.append(cortical_result.get('learning_quality', 0.7))
        
        # Reference frame complexity
        ref_frame_summary = cortical_result.get('reference_frame_updates', {})
        features.append(ref_frame_summary.get('spatial_locations', 0) / 100.0)
        features.append(ref_frame_summary.get('temporal_sequence_length', 0) / 100.0)
        features.append(ref_frame_summary.get('episodic_spatial_contexts', 0) / 50.0)
        
        # Pad to fixed size
        while len(features) < 15:
            features.append(random.uniform(0, 0.05))
        
        return np.array(features[:15])
    
    def _encode_enhanced_identity_features(self, identity_result: Dict) -> np.ndarray:
        """Enhanced identity processing feature encoding (ORIGINAL enhanced)"""
        features = []
        
        # Core identity metrics
        coherence = identity_result.get('coherence_assessment', {})
        features.append(coherence.get('overall_coherence', 0.5))
        features.append(coherence.get('trait_coherence', 0.5))
        features.append(coherence.get('narrative_coherence', 0.5))
        
        personality_state = identity_result.get('personality_state', {})
        features.append(personality_state.get('identity_stability', 0.5))
        features.append(personality_state.get('narrative_coherence', 0.5))
        features.append(personality_state.get('episodic_narrative_depth', 0.0))
        features.append(personality_state.get('cross_episodic_coherence', 0.0))
        
        # Enhanced identity features
        episodic_influence = identity_result.get('episodic_influence_metrics', {})
        features.append(episodic_influence.get('episodic_influence_score', 0.0))
        features.append(episodic_influence.get('memory_depth_factor', 0.0))
        features.append(episodic_influence.get('boundary_influence', 0.0))
        
        # Pad to fixed size
        while len(features) < 15:
            features.append(random.uniform(0, 0.05))
        
        return np.array(features[:15])
    
    def _encode_cross_modal_features(self, experience: SensorimotorExperience) -> np.ndarray:
        """Encode cross-modal features (NEW)"""
        features = []
        
        # Extract basic cross-modal features
        modalities = ['text', 'temporal', 'spatial', 'emotional']
        
        for modality in modalities:
            modal_features = self.cross_modal_system._extract_modal_features(experience, modality)
            if modal_features is not None:
                # Use first few features from each modality
                features.extend(modal_features[:2].tolist())
            else:
                features.extend([0.0, 0.0])
        
        # Cross-modal consistency
        if len(features) >= 4:
            consistency = np.std(features[:4]) if len(features) >= 4 else 0.0
            features.append(1.0 - min(1.0, consistency))
        else:
            features.append(0.5)
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(random.uniform(0, 0.05))
        
        return np.array(features[:10])
    
    def _encode_token_level_features(self, experience: SensorimotorExperience) -> np.ndarray:
        """Encode token-level features (NEW)"""
        features = []
        
        tokens = experience.content.split()
        
        # Token statistics
        features.append(len(tokens) / 100.0)  # Token count
        features.append(len(set(tokens)) / max(len(tokens), 1))  # Token diversity
        features.append(np.mean([len(token) for token in tokens]) if tokens else 0)  # Avg token length
        
        # Token importance (if available)
        if hasattr(self.token_manager, 'token_importance_scores'):
            importance_scores = [self.token_manager.token_importance_scores.get(token, 0.5) for token in tokens]
            features.append(np.mean(importance_scores) if importance_scores else 0.5)
            features.append(np.std(importance_scores) if len(importance_scores) > 1 else 0.0)
        else:
            features.extend([0.5, 0.0])
        
        # Pad to fixed size
        while len(features) < 8:
            features.append(random.uniform(0, 0.05))
        
        return np.array(features[:8])
    
    def _encode_hierarchical_features(self, experience: SensorimotorExperience) -> np.ndarray:
        """Encode hierarchical memory features (NEW)"""
        features = []
        
        # Memory importance and access patterns
        features.append(getattr(experience, 'importance_weight', 0.5))
        features.append(getattr(experience, 'memory_strength', 1.0))
        features.append(min(1.0, getattr(experience, 'access_frequency', 0) / 10.0))
        
        # Goal relevance
        goal_relevance = getattr(experience, 'goal_relevance', {})
        features.append(np.mean(list(goal_relevance.values())) if goal_relevance else 0.5)
        
        # Emotional features
        emotional_features = getattr(experience, 'emotional_features', {})
        features.append(np.mean(list(emotional_features.values())) if emotional_features else 0.5)
        
        # Pad to fixed size
        while len(features) < 8:
            features.append(random.uniform(0, 0.05))
        
        return np.array(features[:8])
    
    def _generate_query_embedding(self, experience: SensorimotorExperience) -> np.ndarray:
        """Generate enhanced query embedding for retrieval (ORIGINAL enhanced)"""
        # Use the same enhanced encoding for query
        content_features = self._encode_enhanced_content_features(experience.content)
        cross_modal_features = self._encode_cross_modal_features(experience)
        token_features = self._encode_token_level_features(experience)
        hierarchical_features = self._encode_hierarchical_features(experience)
        
        combined = np.concatenate([content_features, cross_modal_features, token_features, hierarchical_features])
        return combined / (np.linalg.norm(combined) + 1e-10)
    
    def get_enhanced_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enhanced memory statistics (ORIGINAL enhanced)"""
        
        base_stats = {
            'total_episodes': len(self.episodes),
            'total_tokens_stored': self.total_tokens_stored,
            'episode_boundaries': len(self.episode_boundaries),
            'memory_span_days': self._calculate_memory_span(),
            'average_episode_length': self.total_tokens_stored / max(len(self.episodes), 1),
            'surprise_threshold': self.surprise_threshold,
            'buffer_sizes': {
                'similarity_buffer': self.similarity_buffer_size,
                'contiguity_buffer': self.contiguity_buffer_size,
                'current_contiguity': len(self.contiguity_buffer)
            }
        }
        
        # Enhanced statistics (NEW)
        enhanced_stats = {
            'token_management': self.token_manager.get_memory_statistics(),
            'compression_stats': dict(self.compression_stats),
            'cross_modal_stats': self.cross_modal_system.get_cross_modal_statistics(),
            'hierarchical_memory_stats': self.hierarchical_memory.get_memory_statistics(),
            'advanced_retrieval_stats': self.advanced_retrieval.get_retrieval_statistics(),
            'boundary_refinement': {
                'refinements_performed': len(self.boundary_refiner.refinement_history),
                'average_improvement': np.mean([r['improvement'] for r in self.boundary_refiner.refinement_history]) if self.boundary_refiner.refinement_history else 0.0
            },
            'retrieval_performance': {
                'total_retrievals': len(self.retrieval_performance),
                'average_retrieval_time': np.mean([r['retrieval_time'] for r in self.retrieval_performance]) if self.retrieval_performance else 0.0,
                'average_episodes_per_retrieval': np.mean([r['episodes_retrieved'] for r in self.retrieval_performance]) if self.retrieval_performance else 0.0
            }
        }
        
        return {**base_stats, 'enhanced_features': enhanced_stats}
    
    def _calculate_memory_span(self) -> float:
        """Calculate memory span in days (ORIGINAL)"""
        if len(self.episodes) < 2:
            return 0.0
        
        earliest = self.episodes[0]['timestamp']
        latest = self.episodes[-1]['timestamp']
        
        try:
            earliest_time = datetime.fromisoformat(earliest.replace('Z', '+00:00'))
            latest_time = datetime.fromisoformat(latest.replace('Z', '+00:00'))
            return (latest_time - earliest_time).total_seconds() / 86400
        except:
            return 0.0

class BayesianSurpriseDetector:
    """Enhanced Bayesian surprise detection (ORIGINAL)"""
    
    def __init__(self):
        self.prediction_history = deque(maxlen=100)
        self.baseline_surprise = 0.5
        
    def calculate_surprise(self, prediction_accuracy: float, novelty: float, 
                         consensus_confidence: float) -> float:
        """Calculate enhanced Bayesian surprise score"""
        
        # Surprise increases when predictions fail
        prediction_surprise = 1.0 - prediction_accuracy
        
        # Surprise increases with novelty
        novelty_surprise = novelty
        
        # Surprise increases when consensus is low
        consensus_surprise = 1.0 - consensus_confidence
        
        # Enhanced weighted combination
        total_surprise = (
            prediction_surprise * 0.4 +
            novelty_surprise * 0.4 +
            consensus_surprise * 0.2
        )
        
        # Update baseline with exponential moving average
        self.prediction_history.append(total_surprise)
        if len(self.prediction_history) > 10:
            self.baseline_surprise = 0.9 * self.baseline_surprise + 0.1 * np.mean(list(self.prediction_history))
        
        # Return relative surprise
        return total_surprise / max(self.baseline_surprise, 0.1)

# ============================================================================
# 6-LAYER CORTICAL COLUMN PROCESSOR (COMPLETE ORIGINAL)
# ============================================================================

class Enhanced6LayerCorticalProcessor:
    """Enhanced 6-layer cortical column processor with complete episodic integration"""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.columns = {}
        self.global_predictions = {}
        self.learning_rates = {
            'sensory': 0.1,
            'pattern': 0.15,
            'spatial': 0.1,
            'temporal': 0.2,
            'prediction': 0.25,
            'motor': 0.1
        }
        
        # Episodic integration
        self.episodic_memory_engine = None  # Will be set by main system
        self.episodic_context_cache = {}
        self.episodic_influence_weights = {
            'spatial_context': 0.3,
            'temporal_sequence': 0.4,
            'pattern_priming': 0.3
        }
        
        # Initialize specialized columns
        self._initialize_cortical_columns()
    
    def _initialize_cortical_columns(self):
        """Initialize domain-specific cortical columns"""
        
        column_specs = self._get_domain_specific_columns(self.domain)
        
        for spec in column_specs:
            column = Enhanced6LayerCorticalColumn(
                column_id=spec['column_id'],
                specialization=spec['specialization'],
                layer1_sensory=spec['layer1_config'],
                layer2_pattern=spec['layer2_config'],
                layer3_spatial=spec['layer3_config'],
                layer4_temporal=spec['layer4_config'],
                layer5_prediction=spec['layer5_config'],
                layer6_motor=spec['layer6_config'],
                episodic_context={},
                episodic_predictions={},
                prediction_accuracy=0.5,
                learning_rate=self.learning_rates.get(spec['primary_function'], 0.1),
                episodic_influence=0.3,
                reference_frame={},
                last_updated=datetime.now().isoformat()
            )
            
            self.columns[spec['column_id']] = column
    
    def _get_domain_specific_columns(self, domain: str) -> List[Dict[str, Any]]:
        """Get domain-specific column configurations"""
        
        if domain == "financial_analysis":
            return [
                {
                    'column_id': 'market_pattern_detector',
                    'specialization': 'market_pattern_recognition',
                    'primary_function': 'pattern',
                    'layer1_config': {'sensory_filters': ['price_movement', 'volume_change', 'volatility'], 'adaptation_rate': 0.1},
                    'layer2_config': {'pattern_templates': ['trend_reversal', 'support_resistance', 'breakout'], 'binding_strength': 0.8},
                    'layer3_config': {'spatial_maps': ['market_sectors', 'geographical_markets'], 'resolution': 'high'},
                    'layer4_config': {'sequence_memory': {'window_size': 20, 'decay_rate': 0.1}, 'temporal_pooling': True},
                    'layer5_config': {'prediction_horizon': 5, 'confidence_threshold': 0.7, 'ensemble_size': 3},
                    'layer6_config': {'motor_actions': ['buy_signal', 'sell_signal', 'hold_recommendation'], 'action_threshold': 0.6}
                },
                {
                    'column_id': 'sentiment_analyzer',
                    'specialization': 'market_sentiment_analysis',
                    'primary_function': 'temporal',
                    'layer1_config': {'sensory_filters': ['news_sentiment', 'social_sentiment', 'analyst_sentiment'], 'adaptation_rate': 0.15},
                    'layer2_config': {'pattern_templates': ['bullish_sentiment', 'bearish_sentiment', 'neutral_sentiment'], 'binding_strength': 0.7},
                    'layer3_config': {'spatial_maps': ['sentiment_geography', 'market_segments'], 'resolution': 'medium'},
                    'layer4_config': {'sequence_memory': {'window_size': 15, 'decay_rate': 0.15}, 'temporal_pooling': True},
                    'layer5_config': {'prediction_horizon': 3, 'confidence_threshold': 0.6, 'ensemble_size': 2},
                    'layer6_config': {'motor_actions': ['sentiment_shift_alert', 'confidence_update'], 'action_threshold': 0.5}
                },
                {
                    'column_id': 'risk_assessor',
                    'specialization': 'risk_evaluation',
                    'primary_function': 'prediction',
                    'layer1_config': {'sensory_filters': ['volatility_metrics', 'correlation_changes', 'liquidity_indicators'], 'adaptation_rate': 0.12},
                    'layer2_config': {'pattern_templates': ['risk_escalation', 'risk_mitigation', 'stable_risk'], 'binding_strength': 0.85},
                    'layer3_config': {'spatial_maps': ['risk_correlation_matrix', 'portfolio_exposure'], 'resolution': 'high'},
                    'layer4_config': {'sequence_memory': {'window_size': 25, 'decay_rate': 0.08}, 'temporal_pooling': True},
                    'layer5_config': {'prediction_horizon': 7, 'confidence_threshold': 0.8, 'ensemble_size': 4},
                    'layer6_config': {'motor_actions': ['risk_alert', 'portfolio_rebalance', 'hedge_recommendation'], 'action_threshold': 0.7}
                }
            ]
        
        elif domain == "research":
            return [
                {
                    'column_id': 'concept_integrator',
                    'specialization': 'conceptual_integration',
                    'primary_function': 'pattern',
                    'layer1_config': {'sensory_filters': ['concept_similarity', 'semantic_relations', 'logical_connections'], 'adaptation_rate': 0.08},
                    'layer2_config': {'pattern_templates': ['causal_relationships', 'analogical_reasoning', 'hierarchical_concepts'], 'binding_strength': 0.9},
                    'layer3_config': {'spatial_maps': ['knowledge_graph', 'concept_space'], 'resolution': 'very_high'},
                    'layer4_config': {'sequence_memory': {'window_size': 30, 'decay_rate': 0.05}, 'temporal_pooling': True},
                    'layer5_config': {'prediction_horizon': 10, 'confidence_threshold': 0.75, 'ensemble_size': 5},
                    'layer6_config': {'motor_actions': ['hypothesis_generation', 'experiment_design', 'literature_search'], 'action_threshold': 0.65}
                },
                {
                    'column_id': 'methodology_optimizer',
                    'specialization': 'research_methodology',
                    'primary_function': 'spatial',
                    'layer1_config': {'sensory_filters': ['method_effectiveness', 'data_quality', 'statistical_power'], 'adaptation_rate': 0.1},
                    'layer2_config': {'pattern_templates': ['experimental_design', 'data_analysis', 'validation_methods'], 'binding_strength': 0.8},
                    'layer3_config': {'spatial_maps': ['methodology_space', 'parameter_landscape'], 'resolution': 'high'},
                    'layer4_config': {'sequence_memory': {'window_size': 20, 'decay_rate': 0.1}, 'temporal_pooling': True},
                    'layer5_config': {'prediction_horizon': 8, 'confidence_threshold': 0.7, 'ensemble_size': 3},
                    'layer6_config': {'motor_actions': ['method_recommendation', 'parameter_adjustment', 'validation_check'], 'action_threshold': 0.6}
                }
            ]
        
        else:  # general domain
            return [
                {
                    'column_id': 'general_processor',
                    'specialization': 'general_cognitive_processing',
                    'primary_function': 'pattern',
                    'layer1_config': {'sensory_filters': ['content_analysis', 'context_extraction', 'novelty_detection'], 'adaptation_rate': 0.1},
                    'layer2_config': {'pattern_templates': ['information_integration', 'context_binding', 'relevance_assessment'], 'binding_strength': 0.75},
                    'layer3_config': {'spatial_maps': ['information_space', 'context_map'], 'resolution': 'medium'},
                    'layer4_config': {'sequence_memory': {'window_size': 15, 'decay_rate': 0.12}, 'temporal_pooling': True},
                    'layer5_config': {'prediction_horizon': 5, 'confidence_threshold': 0.6, 'ensemble_size': 2},
                    'layer6_config': {'motor_actions': ['information_synthesis', 'response_generation'], 'action_threshold': 0.5}
                }
            ]
    
    def set_episodic_memory_engine(self, episodic_engine):
        """Set the episodic memory engine for integration"""
        self.episodic_memory_engine = episodic_engine
    
    def process_experience_with_episodes(self, experience: SensorimotorExperience) -> Dict[str, Any]:
        """Process experience through cortical columns with episodic integration"""
        
        # Retrieve episodic context if available
        episodic_context = self._get_episodic_context(experience)
        
        # Process through each cortical column
        column_results = {}
        for column_id, column in self.columns.items():
            # Update episodic context for column
            column.episodic_context = episodic_context.get(column_id, {})
            
            # Process experience through 6 layers
            layer_results = self._process_through_cortical_layers(experience, column, episodic_context)
            column_results[column_id] = layer_results
            
            # Update column's episodic predictions
            column.episodic_predictions = self._generate_episodic_predictions(column, layer_results, episodic_context)
            
            # Update learning metrics
            self._update_column_learning_metrics(column, layer_results)
        
        # Generate consensus across columns
        consensus_result = self._generate_cortical_consensus(column_results, experience)
        
        # Update global predictions and reference frames
        self._update_global_predictions(consensus_result, experience)
        reference_frame_updates = self._update_reference_frames(experience, consensus_result, episodic_context)
        
        return {
            'column_results': column_results,
            'consensus': consensus_result,
            'prediction_accuracy': consensus_result.get('overall_confidence', 0.5),
            'domain_expertise_level': self._calculate_domain_expertise(),
            'episodic_integration_quality': self._assess_episodic_integration_quality(episodic_context),
            'learning_quality': np.mean([col.prediction_accuracy for col in self.columns.values()]),
            'reference_frame_updates': reference_frame_updates,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _get_episodic_context(self, experience: SensorimotorExperience) -> Dict[str, Any]:
        """Retrieve episodic context for cortical processing"""
        
        if not self.episodic_memory_engine:
            return {}
        
        # Check cache first
        cache_key = f"{experience.domain}_{experience.timestamp[:16]}"  # Hour-level caching
        if cache_key in self.episodic_context_cache:
            return self.episodic_context_cache[cache_key]
        
        # Retrieve episodic context
        episodic_retrieval = self.episodic_memory_engine.retrieve_episodic_context(
            experience, max_context_tokens=4000
        )
        
        episodic_episodes = episodic_retrieval.get('episodes', [])
        
        # Process episodic context for each column
        column_contexts = {}
        for column_id, column in self.columns.items():
            column_contexts[column_id] = self._extract_column_specific_context(
                column, episodic_episodes, experience
            )
        
        # Cache result
        self.episodic_context_cache[cache_key] = column_contexts
        
        return column_contexts
    
    def _extract_column_specific_context(self, column: Enhanced6LayerCorticalColumn, 
                                       episodes: List[Dict], experience: SensorimotorExperience) -> Dict[str, Any]:
        """Extract column-specific context from episodic episodes"""
        
        context = {
            'relevant_episodes': [],
            'spatial_priors': {},
            'temporal_expectations': {},
            'pattern_priming': {}
        }
        
        for episode in episodes:
            episode_relevance = self._calculate_episode_relevance_for_column(column, episode, experience)
            
            if episode_relevance > 0.4:
                context['relevant_episodes'].append({
                    'episode_id': episode['episode_id'],
                    'content': episode['content'][:200],  # Truncate for efficiency
                    'relevance': episode_relevance,
                    'cortical_patterns': episode.get('cortical_patterns', {}),
                    'domain': episode.get('domain', ''),
                    'timestamp': episode.get('timestamp', '')
                })
                
                # Extract spatial priors
                if column.specialization in ['market_pattern_recognition', 'conceptual_integration']:
                    spatial_patterns = episode.get('cortical_patterns', {}).get('spatial_patterns', {})
                    for pattern, strength in spatial_patterns.items():
                        if pattern not in context['spatial_priors']:
                            context['spatial_priors'][pattern] = []
                        context['spatial_priors'][pattern].append(strength)
                
                # Extract temporal expectations
                if column.specialization in ['market_sentiment_analysis', 'research_methodology']:
                    temporal_patterns = episode.get('cortical_patterns', {}).get('temporal_patterns', {})
                    context['temporal_expectations'].update(temporal_patterns)
                
                # Extract pattern priming
                episode_patterns = episode.get('cortical_patterns', {})
                for pattern_type, patterns in episode_patterns.items():
                    if pattern_type not in context['pattern_priming']:
                        context['pattern_priming'][pattern_type] = {}
                    context['pattern_priming'][pattern_type].update(patterns)
        
        # Aggregate spatial priors
        for pattern, values in context['spatial_priors'].items():
            context['spatial_priors'][pattern] = np.mean(values) if values else 0.0
        
        return context
    
    def _calculate_episode_relevance_for_column(self, column: Enhanced6LayerCorticalColumn, 
                                              episode: Dict, experience: SensorimotorExperience) -> float:
        """Calculate how relevant an episode is for a specific column"""
        
        relevance_factors = []
        
        # Domain relevance
        episode_domain = episode.get('domain', '')
        if episode_domain == experience.domain:
            relevance_factors.append(0.8)
        elif episode_domain in ['general', experience.domain]:
            relevance_factors.append(0.5)
        else:
            relevance_factors.append(0.2)
        
        # Specialization relevance
        episode_patterns = episode.get('cortical_patterns', {})
        if column.specialization == 'market_pattern_recognition':
            pattern_relevance = len([p for p in episode_patterns.keys() if 'market' in p or 'price' in p]) / max(len(episode_patterns), 1)
            relevance_factors.append(pattern_relevance)
        elif column.specialization == 'market_sentiment_analysis':
            sentiment_relevance = len([p for p in episode_patterns.keys() if 'sentiment' in p or 'emotion' in p]) / max(len(episode_patterns), 1)
            relevance_factors.append(sentiment_relevance)
        elif column.specialization == 'risk_evaluation':
            risk_relevance = len([p for p in episode_patterns.keys() if 'risk' in p or 'volatility' in p]) / max(len(episode_patterns), 1)
            relevance_factors.append(risk_relevance)
        else:
            relevance_factors.append(0.5)  # Default relevance
        
        # Content similarity
        episode_content = episode.get('content', '')
        experience_content = experience.content
        content_words_episode = set(episode_content.lower().split())
        content_words_experience = set(experience_content.lower().split())
        
        if content_words_episode and content_words_experience:
            content_similarity = len(content_words_episode & content_words_experience) / len(content_words_episode | content_words_experience)
            relevance_factors.append(content_similarity)
        else:
            relevance_factors.append(0.0)
        
        return np.mean(relevance_factors)
    
    def _process_through_cortical_layers(self, experience: SensorimotorExperience, 
                                       column: Enhanced6LayerCorticalColumn, 
                                       episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process experience through all 6 cortical layers"""
        
        layer_results = {}
        
        # Layer 1: Sensory Processing
        layer1_result = self._process_layer1_sensory(experience, column, episodic_context)
        layer_results['layer1_sensory'] = layer1_result
        
        # Layer 2: Pattern Recognition & Binding
        layer2_result = self._process_layer2_pattern(experience, column, layer1_result, episodic_context)
        layer_results['layer2_pattern'] = layer2_result
        
        # Layer 3: Spatial Location Encoding
        layer3_result = self._process_layer3_spatial(experience, column, layer2_result, episodic_context)
        layer_results['layer3_spatial'] = layer3_result
        
        # Layer 4: Temporal Sequence Learning
        layer4_result = self._process_layer4_temporal(experience, column, layer3_result, episodic_context)
        layer_results['layer4_temporal'] = layer4_result
        
        # Layer 5: Prediction Generation
        layer5_result = self._process_layer5_prediction(experience, column, layer4_result, episodic_context)
        layer_results['layer5_prediction'] = layer5_result
        
        # Layer 6: Motor Output Planning
        layer6_result = self._process_layer6_motor(experience, column, layer5_result, episodic_context)
        layer_results['layer6_motor'] = layer6_result
        
        return layer_results
    
    def _process_layer1_sensory(self, experience: SensorimotorExperience, 
                              column: Enhanced6LayerCorticalColumn, 
                              episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 1: Enhanced sensory input processing with episodic priming"""
        
        sensory_filters = column.layer1_sensory.get('sensory_filters', [])
        adaptation_rate = column.layer1_sensory.get('adaptation_rate', 0.1)
        
        # Extract sensory features
        sensory_features = {}
        
        for filter_name in sensory_filters:
            if filter_name == 'price_movement':
                sensory_features[filter_name] = self._extract_price_movement_features(experience)
            elif filter_name == 'volume_change':
                sensory_features[filter_name] = self._extract_volume_change_features(experience)
            elif filter_name == 'volatility':
                sensory_features[filter_name] = self._extract_volatility_features(experience)
            elif filter_name == 'news_sentiment':
                sensory_features[filter_name] = self._extract_news_sentiment_features(experience)
            elif filter_name == 'content_analysis':
                sensory_features[filter_name] = self._extract_content_analysis_features(experience)
            else:
                sensory_features[filter_name] = random.uniform(0.3, 0.7)  # Default feature
        
        # Apply episodic priming
        column_context = episodic_context.get(column.column_id, {})
        pattern_priming = column_context.get('pattern_priming', {})
        
        primed_features = self._apply_episodic_priming(sensory_features, pattern_priming, 'sensory')
        
        # Adaptation based on prediction accuracy
        adaptation_factor = 1.0 + (column.prediction_accuracy - 0.5) * adaptation_rate
        adapted_features = {k: v * adaptation_factor for k, v in primed_features.items()}
        
        return {
            'raw_features': sensory_features,
            'primed_features': primed_features,
            'adapted_features': adapted_features,
            'adaptation_factor': adaptation_factor,
            'episodic_influence': len(pattern_priming) > 0
        }
    
    def _process_layer2_pattern(self, experience: SensorimotorExperience, 
                              column: Enhanced6LayerCorticalColumn, 
                              layer1_result: Dict[str, Any], 
                              episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 2: Pattern recognition and binding with episodic memory"""
        
        pattern_templates = column.layer2_pattern.get('pattern_templates', [])
        binding_strength = column.layer2_pattern.get('binding_strength', 0.8)
        
        adapted_features = layer1_result['adapted_features']
        
        # Pattern matching
        pattern_matches = {}
        for template in pattern_templates:
            match_strength = self._calculate_pattern_match(adapted_features, template, experience)
            pattern_matches[template] = match_strength
        
        # Episodic pattern binding
        column_context = episodic_context.get(column.column_id, {})
        spatial_priors = column_context.get('spatial_priors', {})
        
        # Enhance pattern matches with episodic priors
        enhanced_patterns = {}
        for pattern, strength in pattern_matches.items():
            episodic_boost = spatial_priors.get(pattern, 0.0) * column.episodic_influence
            enhanced_strength = strength + episodic_boost
            enhanced_patterns[pattern] = min(1.0, enhanced_strength)
        
        # Select winning patterns based on binding strength
        winning_patterns = {k: v for k, v in enhanced_patterns.items() if v > binding_strength}
        
        return {
            'pattern_matches': pattern_matches,
            'enhanced_patterns': enhanced_patterns,
            'winning_patterns': winning_patterns,
            'binding_strength': binding_strength,
            'episodic_boost_applied': len(spatial_priors) > 0
        }
    
    def _process_layer3_spatial(self, experience: SensorimotorExperience, 
                              column: Enhanced6LayerCorticalColumn, 
                              layer2_result: Dict[str, Any], 
                              episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 3: Spatial location encoding with episodic spatial context"""
        
        spatial_maps = column.layer3_spatial.get('spatial_maps', [])
        resolution = column.layer3_spatial.get('resolution', 'medium')
        
        winning_patterns = layer2_result['winning_patterns']
        
        # Create spatial encodings
        spatial_encodings = {}
        
        for map_name in spatial_maps:
            if map_name == 'market_sectors':
                spatial_encodings[map_name] = self._encode_market_sectors(experience, winning_patterns)
            elif map_name == 'knowledge_graph':
                spatial_encodings[map_name] = self._encode_knowledge_graph(experience, winning_patterns)
            elif map_name == 'information_space':
                spatial_encodings[map_name] = self._encode_information_space(experience, winning_patterns)
            else:
                spatial_encodings[map_name] = self._create_default_spatial_encoding(winning_patterns)
        
        # Apply episodic spatial context
        column_context = episodic_context.get(column.column_id, {})
        spatial_priors = column_context.get('spatial_priors', {})
        
        # Update column's episodic spatial context
        column.reference_frame['episodic_spatial_contexts'] = spatial_priors
        
        # Enhance spatial encodings with episodic context
        enhanced_spatial = self._enhance_spatial_with_episodic_context(spatial_encodings, spatial_priors)
        
        return {
            'spatial_encodings': spatial_encodings,
            'enhanced_spatial': enhanced_spatial,
            'resolution': resolution,
            'episodic_spatial_influence': len(spatial_priors) > 0
        }
    
    def _process_layer4_temporal(self, experience: SensorimotorExperience, 
                               column: Enhanced6LayerCorticalColumn, 
                               layer3_result: Dict[str, Any], 
                               episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 4: Temporal sequence learning with episodic temporal context"""
        
        sequence_config = column.layer4_temporal.get('sequence_memory', {})
        window_size = sequence_config.get('window_size', 15)
        decay_rate = sequence_config.get('decay_rate', 0.1)
        temporal_pooling = column.layer4_temporal.get('temporal_pooling', True)
        
        enhanced_spatial = layer3_result['enhanced_spatial']
        
        # Update temporal sequence in reference frame
        if 'temporal_sequence' not in column.reference_frame:
            column.reference_frame['temporal_sequence'] = deque(maxlen=window_size)
        
        # Add current state to temporal sequence
        current_state = {
            'timestamp': experience.timestamp,
            'spatial_state': enhanced_spatial,
            'experience_id': experience.experience_id
        }
        column.reference_frame['temporal_sequence'].append(current_state)
        
        # Apply episodic temporal expectations
        column_context = episodic_context.get(column.column_id, {})
        temporal_expectations = column_context.get('temporal_expectations', {})
        
        # Temporal pattern detection
        temporal_patterns = self._detect_temporal_patterns(
            column.reference_frame['temporal_sequence'], 
            temporal_expectations,
            experience
        )
        
        # Temporal pooling if enabled
        pooled_representation = None
        if temporal_pooling:
            pooled_representation = self._temporal_pooling(
                column.reference_frame['temporal_sequence'], 
                decay_rate
            )
        
        return {
            'temporal_patterns': temporal_patterns,
            'sequence_length': len(column.reference_frame['temporal_sequence']),
            'pooled_representation': pooled_representation,
            'episodic_temporal_influence': len(temporal_expectations) > 0,
            'current_state': current_state
        }
    
    def _process_layer5_prediction(self, experience: SensorimotorExperience, 
                                 column: Enhanced6LayerCorticalColumn, 
                                 layer4_result: Dict[str, Any], 
                                 episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 5: Prediction generation with episodic prediction enhancement"""
        
        prediction_horizon = column.layer5_prediction.get('prediction_horizon', 5)
        confidence_threshold = column.layer5_prediction.get('confidence_threshold', 0.7)
        ensemble_size = column.layer5_prediction.get('ensemble_size', 3)
        
        temporal_patterns = layer4_result['temporal_patterns']
        pooled_representation = layer4_result['pooled_representation']
        
        # Generate base predictions
        base_predictions = self._generate_base_predictions(
            temporal_patterns, 
            pooled_representation, 
            prediction_horizon,
            experience
        )
        
        # Episodic prediction enhancement
        column_context = episodic_context.get(column.column_id, {})
        relevant_episodes = column_context.get('relevant_episodes', [])
        
        episodic_predictions = self._generate_episodic_predictions(
            column, 
            layer4_result, 
            episodic_context
        )
        
        # Ensemble predictions
        ensemble_predictions = self._create_prediction_ensemble(
            base_predictions, 
            episodic_predictions, 
            ensemble_size
        )
        
        # Filter by confidence
        confident_predictions = {
            k: v for k, v in ensemble_predictions.items() 
            if v.get('confidence', 0) > confidence_threshold
        }
        
        return {
            'base_predictions': base_predictions,
            'episodic_predictions': episodic_predictions,
            'ensemble_predictions': ensemble_predictions,
            'confident_predictions': confident_predictions,
            'prediction_horizon': prediction_horizon,
            'episodic_enhancement_applied': len(relevant_episodes) > 0
        }
    
    def _process_layer6_motor(self, experience: SensorimotorExperience, 
                            column: Enhanced6LayerCorticalColumn, 
                            layer5_result: Dict[str, Any], 
                            episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 6: Motor output planning with episodic action context"""
        
        motor_actions = column.layer6_motor.get('motor_actions', [])
        action_threshold = column.layer6_motor.get('action_threshold', 0.6)
        
        confident_predictions = layer5_result['confident_predictions']
        
        # Generate motor outputs based on predictions
        motor_outputs = {}
        
        for action in motor_actions:
            action_strength = self._calculate_action_strength(
                action, 
                confident_predictions, 
                experience
            )
            
            if action_strength > action_threshold:
                motor_outputs[action] = {
                    'strength': action_strength,
                    'confidence': self._calculate_action_confidence(action, confident_predictions),
                    'episodic_context': self._get_action_episodic_context(action, episodic_context)
                }
        
        return {
            'motor_outputs': motor_outputs,
            'action_threshold': action_threshold,
            'total_actions_triggered': len(motor_outputs)
        }
    
    def _extract_price_movement_features(self, experience: SensorimotorExperience) -> float:
        """Extract price movement features from experience"""
        content = experience.content.lower()
        
        # Look for price-related keywords
        price_up_words = ['surge', 'rally', 'gains', 'climb', 'rise', 'increase', 'bull']
        price_down_words = ['plunge', 'fall', 'decline', 'drop', 'crash', 'bear']
        
        up_count = sum(1 for word in price_up_words if word in content)
        down_count = sum(1 for word in price_down_words if word in content)
        
        if up_count > down_count:
            return 0.7 + (up_count - down_count) * 0.1
        elif down_count > up_count:
            return 0.3 - (down_count - up_count) * 0.1
        else:
            return 0.5
    
    def _extract_volume_change_features(self, experience: SensorimotorExperience) -> float:
        """Extract volume change features from experience"""
        content = experience.content.lower()
        
        volume_words = ['volume', 'trading', 'activity', 'liquidity']
        high_volume_words = ['heavy', 'massive', 'unprecedented', 'record']
        
        volume_mentions = sum(1 for word in volume_words if word in content)
        high_volume_mentions = sum(1 for word in high_volume_words if word in content)
        
        return min(1.0, (volume_mentions + high_volume_mentions * 2) * 0.2)
    
    def _extract_volatility_features(self, experience: SensorimotorExperience) -> float:
        """Extract volatility features from experience"""
        content = experience.content.lower()
        
        volatility_words = ['volatile', 'volatility', 'swing', 'fluctuation', 'unstable']
        stability_words = ['stable', 'steady', 'calm', 'consistent']
        
        volatility_count = sum(1 for word in volatility_words if word in content)
        stability_count = sum(1 for word in stability_words if word in content)
        
        if volatility_count > stability_count:
            return 0.7 + volatility_count * 0.1
        elif stability_count > volatility_count:
            return 0.3 - stability_count * 0.1
        else:
            return 0.5
    
    def _extract_news_sentiment_features(self, experience: SensorimotorExperience) -> float:
        """Extract news sentiment features from experience"""
        content = experience.content.lower()
        
        positive_words = ['positive', 'optimistic', 'bullish', 'confident', 'growth', 'success']
        negative_words = ['negative', 'pessimistic', 'bearish', 'concerned', 'decline', 'failure']
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words > 0:
            return positive_count / total_sentiment_words
        else:
            return 0.5
    
    def _extract_content_analysis_features(self, experience: SensorimotorExperience) -> float:
        """Extract general content analysis features"""
        content = experience.content
        
        # Content complexity metrics
        word_count = len(content.split())
        unique_words = len(set(content.lower().split()))
        avg_word_length = np.mean([len(word) for word in content.split()]) if content.split() else 0
        
        # Normalize features
        complexity_score = (
            min(1.0, word_count / 100.0) * 0.4 +
            min(1.0, unique_words / word_count if word_count > 0 else 0) * 0.3 +
            min(1.0, avg_word_length / 10.0) * 0.3
        )
        
        return complexity_score
    
    def _apply_episodic_priming(self, features: Dict[str, float], 
                              pattern_priming: Dict[str, Dict], 
                              layer_type: str) -> Dict[str, float]:
        """Apply episodic priming to features"""
        
        primed_features = features.copy()
        
        if layer_type in pattern_priming:
            layer_priming = pattern_priming[layer_type]
            
            for feature_name, feature_value in features.items():
                if feature_name in layer_priming:
                    priming_strength = layer_priming[feature_name]
                    # Apply priming as a multiplicative boost
                    primed_features[feature_name] = feature_value * (1.0 + priming_strength * 0.2)
        
        return primed_features
    
    def _calculate_pattern_match(self, features: Dict[str, float], 
                               template: str, experience: SensorimotorExperience) -> float:
        """Calculate pattern match strength"""
        
        content = experience.content.lower()
        
        # Template-specific pattern matching
        if template == 'trend_reversal':
            reversal_words = ['reversal', 'turn', 'shift', 'change', 'opposite']
            match_strength = sum(1 for word in reversal_words if word in content) * 0.2
        elif template == 'support_resistance':
            support_words = ['support', 'resistance', 'level', 'barrier', 'floor', 'ceiling']
            match_strength = sum(1 for word in support_words if word in content) * 0.15
        elif template == 'breakout':
            breakout_words = ['breakout', 'breakthrough', 'break', 'burst', 'explosion']
            match_strength = sum(1 for word in breakout_words if word in content) * 0.25
        elif template == 'bullish_sentiment':
            bull_words = ['bullish', 'optimistic', 'positive', 'growth', 'rally']
            match_strength = sum(1 for word in bull_words if word in content) * 0.2
        elif template == 'bearish_sentiment':
            bear_words = ['bearish', 'pessimistic', 'negative', 'decline', 'fall']
            match_strength = sum(1 for word in bear_words if word in content) * 0.2
        else:
            # Default pattern matching based on feature values
            match_strength = np.mean(list(features.values()))
        
        return min(1.0, match_strength)
    
    def _encode_market_sectors(self, experience: SensorimotorExperience, 
                             winning_patterns: Dict[str, float]) -> np.ndarray:
        """Encode market sector spatial information"""
        
        content = experience.content.lower()
        
        # Market sectors
        sectors = {
            'technology': ['tech', 'software', 'ai', 'digital', 'innovation'],
            'finance': ['bank', 'financial', 'credit', 'loan', 'mortgage'],
            'healthcare': ['health', 'medical', 'pharma', 'biotech', 'drug'],
            'energy': ['energy', 'oil', 'gas', 'renewable', 'solar'],
            'consumer': ['retail', 'consumer', 'goods', 'shopping', 'brand']
        }
        
        sector_encoding = []
        for sector, keywords in sectors.items():
            sector_strength = sum(1 for keyword in keywords if keyword in content) / len(keywords)
            sector_encoding.append(sector_strength)
        
        # Add pattern influence
        pattern_influence = np.mean(list(winning_patterns.values())) if winning_patterns else 0.0
        sector_encoding.append(pattern_influence)
        
        return np.array(sector_encoding)
    
    def _encode_knowledge_graph(self, experience: SensorimotorExperience, 
                              winning_patterns: Dict[str, float]) -> np.ndarray:
        """Encode knowledge graph spatial information"""
        
        content = experience.content.lower()
        words = set(content.split())
        
        # Knowledge domains
        domains = {
            'scientific': ['research', 'study', 'analysis', 'method', 'data'],
            'theoretical': ['theory', 'model', 'concept', 'principle', 'framework'],
            'practical': ['application', 'implementation', 'practice', 'solution', 'tool'],
            'empirical': ['evidence', 'observation', 'experiment', 'result', 'finding']
        }
        
        domain_encoding = []
        for domain, keywords in domains.items():
            domain_strength = len(words.intersection(keywords)) / len(keywords)
            domain_encoding.append(domain_strength)
        
        # Add complexity measure
        complexity = min(1.0, len(words) / 50.0)
        domain_encoding.append(complexity)
        
        return np.array(domain_encoding)
    
    def _encode_information_space(self, experience: SensorimotorExperience, 
                                winning_patterns: Dict[str, float]) -> np.ndarray:
        """Encode general information space"""
        
        content = experience.content
        
        # Information characteristics
        features = []
        
        # Factual vs. opinion
        fact_words = ['fact', 'data', 'statistic', 'number', 'measure']
        opinion_words = ['think', 'believe', 'opinion', 'view', 'perspective']
        
        fact_score = sum(1 for word in fact_words if word in content.lower()) / len(fact_words)
        opinion_score = sum(1 for word in opinion_words if word in content.lower()) / len(opinion_words)
        
        features.extend([fact_score, opinion_score])
        
        # Certainty vs. uncertainty
        certain_words = ['certain', 'sure', 'definite', 'confirmed', 'proven']
        uncertain_words = ['maybe', 'possibly', 'uncertain', 'unclear', 'ambiguous']
        
        certainty_score = sum(1 for word in certain_words if word in content.lower()) / len(certain_words)
        uncertainty_score = sum(1 for word in uncertain_words if word in content.lower()) / len(uncertain_words)
        
        features.extend([certainty_score, uncertainty_score])
        
        # Add novelty
        features.append(experience.novelty_score)
        
        return np.array(features)
    
    def _create_default_spatial_encoding(self, winning_patterns: Dict[str, float]) -> np.ndarray:
        """Create default spatial encoding"""
        
        if winning_patterns:
            pattern_values = list(winning_patterns.values())
            return np.array(pattern_values + [0.0] * (5 - len(pattern_values)))[:5]
        else:
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    def _enhance_spatial_with_episodic_context(self, spatial_encodings: Dict[str, np.ndarray], 
                                             spatial_priors: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Enhance spatial encodings with episodic context"""
        
        enhanced_spatial = {}
        
        for map_name, encoding in spatial_encodings.items():
            enhanced_encoding = encoding.copy()
            
            # Apply episodic priors
            if spatial_priors:
                prior_boost = np.mean(list(spatial_priors.values()))
                enhanced_encoding = enhanced_encoding * (1.0 + prior_boost * 0.2)
            
            enhanced_spatial[map_name] = enhanced_encoding
        
        return enhanced_spatial
    
    def _detect_temporal_patterns(self, temporal_sequence: deque, 
                                temporal_expectations: Dict[str, Any],
                                experience: SensorimotorExperience) -> Dict[str, float]:
        """Detect temporal patterns in sequence"""
        
        patterns = {}
        
        if len(temporal_sequence) < 3:
            return patterns
        
        # Convert sequence to features for analysis
        sequence_features = []
        for state in temporal_sequence:
            spatial_state = state['spatial_state']
            # Flatten spatial state for pattern analysis
            flattened = []
            for encoding in spatial_state.values():
                if isinstance(encoding, np.ndarray):
                    flattened.extend(encoding.tolist())
                else:
                    flattened.append(encoding)
            sequence_features.append(flattened)
        
        # Detect trends
        if len(sequence_features) >= 3:
            recent_avg = np.mean(sequence_features[-3:], axis=0)
            older_avg = np.mean(sequence_features[:-3], axis=0) if len(sequence_features) > 3 else np.mean(sequence_features[:-1], axis=0)
            
            trend_strength = np.mean(recent_avg - older_avg)
            patterns['trend_strength'] = min(1.0, abs(trend_strength))
            patterns['trend_direction'] = 1.0 if trend_strength > 0 else 0.0
        
        # Detect periodicity (simplified)
        if len(sequence_features) >= 5:
            differences = []
            for i in range(1, len(sequence_features)):
                diff = np.mean(np.abs(np.array(sequence_features[i]) - np.array(sequence_features[i-1])))
                differences.append(diff)
            
            patterns['stability'] = 1.0 - min(1.0, np.std(differences))
        
        # Apply temporal expectations
        for expectation, strength in temporal_expectations.items():
            if expectation in patterns:
                patterns[expectation] *= (1.0 + strength * 0.3)
        
        return patterns
    
    def _temporal_pooling(self, temporal_sequence: deque, decay_rate: float) -> Dict[str, Any]:
        """Perform temporal pooling with decay"""
        
        if not temporal_sequence:
            return {}
        
        pooled_spatial = {}
        
        # Weight states by recency
        total_weight = 0.0
        weighted_states = {}
        
        for i, state in enumerate(temporal_sequence):
            age = len(temporal_sequence) - i - 1
            weight = np.exp(-age * decay_rate)
            total_weight += weight
            
            spatial_state = state['spatial_state']
            for map_name, encoding in spatial_state.items():
                if map_name not in weighted_states:
                    weighted_states[map_name] = np.zeros_like(encoding)
                weighted_states[map_name] += encoding * weight
        
        # Normalize by total weight
        for map_name, weighted_encoding in weighted_states.items():
            pooled_spatial[map_name] = weighted_encoding / total_weight
        
        return {
            'pooled_spatial': pooled_spatial,
            'sequence_length': len(temporal_sequence),
            'total_weight': total_weight
        }
    
    def _generate_base_predictions(self, temporal_patterns: Dict[str, float], 
                                 pooled_representation: Dict[str, Any],
                                 prediction_horizon: int,
                                 experience: SensorimotorExperience) -> Dict[str, Dict[str, float]]:
        """Generate base predictions from temporal patterns"""
        
        predictions = {}
        
        # Trend-based predictions
        if 'trend_strength' in temporal_patterns and 'trend_direction' in temporal_patterns:
            trend_strength = temporal_patterns['trend_strength']
            trend_direction = temporal_patterns['trend_direction']
            
            for horizon in range(1, prediction_horizon + 1):
                pred_key = f'trend_prediction_h{horizon}'
                confidence = trend_strength * np.exp(-horizon * 0.1)  # Decay with horizon
                
                predictions[pred_key] = {
                    'value': trend_direction,
                    'confidence': confidence,
                    'type': 'trend',
                    'horizon': horizon
                }
        
        # Stability-based predictions
        if 'stability' in temporal_patterns:
            stability = temporal_patterns['stability']
            
            for horizon in range(1, prediction_horizon + 1):
                pred_key = f'stability_prediction_h{horizon}'
                confidence = stability * np.exp(-horizon * 0.05)
                
                predictions[pred_key] = {
                    'value': stability,
                    'confidence': confidence,
                    'type': 'stability',
                    'horizon': horizon
                }
        
        # Domain-specific predictions
        if experience.domain == 'financial_analysis':
            predictions.update(self._generate_financial_predictions(temporal_patterns, prediction_horizon))
        elif experience.domain == 'research':
            predictions.update(self._generate_research_predictions(temporal_patterns, prediction_horizon))
        
        return predictions
    
    def _generate_financial_predictions(self, temporal_patterns: Dict[str, float], 
                                      prediction_horizon: int) -> Dict[str, Dict[str, float]]:
        """Generate financial domain-specific predictions"""
        
        predictions = {}
        
        # Market movement predictions
        trend_strength = temporal_patterns.get('trend_strength', 0.5)
        trend_direction = temporal_patterns.get('trend_direction', 0.5)
        
        for horizon in range(1, min(prediction_horizon + 1, 4)):  # Financial predictions limited to 3 steps
            predictions[f'market_movement_h{horizon}'] = {
                'value': trend_direction,
                'confidence': trend_strength * (0.8 ** horizon),
                'type': 'market_movement',
                'horizon': horizon
            }
            
            predictions[f'volatility_h{horizon}'] = {
                'value': 1.0 - temporal_patterns.get('stability', 0.5),
                'confidence': trend_strength * (0.7 ** horizon),
                'type': 'volatility',
                'horizon': horizon
            }
        
        return predictions
    
    def _generate_research_predictions(self, temporal_patterns: Dict[str, float], 
                                     prediction_horizon: int) -> Dict[str, Dict[str, float]]:
        """Generate research domain-specific predictions"""
        
        predictions = {}
        
        # Research progression predictions
        stability = temporal_patterns.get('stability', 0.5)
        trend_strength = temporal_patterns.get('trend_strength', 0.5)
        
        for horizon in range(1, prediction_horizon + 1):
            predictions[f'research_progress_h{horizon}'] = {
                'value': trend_strength,
                'confidence': stability * (0.9 ** horizon),
                'type': 'research_progress',
                'horizon': horizon
            }
            
            predictions[f'knowledge_integration_h{horizon}'] = {
                'value': (stability + trend_strength) / 2.0,
                'confidence': stability * (0.85 ** horizon),
                'type': 'knowledge_integration',
                'horizon': horizon
            }
        
        return predictions
    
    def _generate_episodic_predictions(self, column: Enhanced6LayerCorticalColumn, 
                                     layer_results: Dict[str, Any], 
                                     episodic_context: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Generate predictions enhanced by episodic memory"""
        
        column_context = episodic_context.get(column.column_id, {})
        relevant_episodes = column_context.get('relevant_episodes', [])
        
        if not relevant_episodes:
            return {}
        
        episodic_predictions = {}
        
        # Analyze patterns from relevant episodes
        episode_patterns = []
        for episode in relevant_episodes:
            cortical_patterns = episode.get('cortical_patterns', {})
            if cortical_patterns:
                episode_patterns.append(cortical_patterns)
        
        if not episode_patterns:
            return {}
        
        # Extract common prediction patterns
        common_patterns = self._extract_common_patterns(episode_patterns)
        
        # Generate episodic predictions based on common patterns
        for pattern_name, pattern_strength in common_patterns.items():
            if pattern_strength > 0.3:  # Threshold for episodic prediction
                episodic_predictions[f'episodic_{pattern_name}'] = {
                    'value': pattern_strength,
                    'confidence': pattern_strength * 0.8,  # Slightly lower confidence for episodic
                    'type': 'episodic',
                    'source_episodes': len(relevant_episodes)
                }
        
        return episodic_predictions
    
    def _extract_common_patterns(self, episode_patterns: List[Dict]) -> Dict[str, float]:
        """Extract common patterns across episodes"""
        
        pattern_frequencies = defaultdict(list)
        
        for patterns in episode_patterns:
            for pattern_type, pattern_data in patterns.items():
                if isinstance(pattern_data, dict):
                    for pattern_name, pattern_value in pattern_data.items():
                        if isinstance(pattern_value, (int, float)):
                            pattern_frequencies[f"{pattern_type}_{pattern_name}"].append(pattern_value)
                elif isinstance(pattern_data, (int, float)):
                    pattern_frequencies[pattern_type].append(pattern_data)
        
        # Calculate average strength for each pattern
        common_patterns = {}
        for pattern_name, values in pattern_frequencies.items():
            if len(values) >= 2:  # Pattern must appear in at least 2 episodes
                common_patterns[pattern_name] = np.mean(values)
        
        return common_patterns
    
    def _create_prediction_ensemble(self, base_predictions: Dict[str, Dict[str, float]], 
                                  episodic_predictions: Dict[str, Dict[str, float]], 
                                  ensemble_size: int) -> Dict[str, Dict[str, float]]:
        """Create ensemble of predictions"""
        
        all_predictions = {**base_predictions, **episodic_predictions}
        
        if len(all_predictions) <= ensemble_size:
            return all_predictions
        
        # Sort by confidence and select top predictions
        sorted_predictions = sorted(
            all_predictions.items(), 
            key=lambda x: x[1]['confidence'], 
            reverse=True
        )
        
        ensemble_predictions = dict(sorted_predictions[:ensemble_size])
        
        # Add ensemble metadata
        for pred_name, pred_data in ensemble_predictions.items():
            pred_data['ensemble_rank'] = list(ensemble_predictions.keys()).index(pred_name) + 1
            pred_data['ensemble_size'] = len(ensemble_predictions)
        
        return ensemble_predictions
    
    def _calculate_action_strength(self, action: str, 
                                 confident_predictions: Dict[str, Dict[str, float]], 
                                 experience: SensorimotorExperience) -> float:
        """Calculate action strength based on predictions"""
        
        if not confident_predictions:
            return 0.0
        
        # Action-specific strength calculation
        if action == 'buy_signal':
            # Look for positive trend predictions
            relevant_preds = [p for name, p in confident_predictions.items() 
                            if 'trend' in name and p.get('value', 0) > 0.6]
        elif action == 'sell_signal':
            # Look for negative trend predictions
            relevant_preds = [p for name, p in confident_predictions.items() 
                            if 'trend' in name and p.get('value', 0) < 0.4]
        elif action == 'risk_alert':
            # Look for volatility or instability predictions
            relevant_preds = [p for name, p in confident_predictions.items() 
                            if 'volatility' in name or 'stability' in name]
        elif action == 'hypothesis_generation':
            # Look for research progress predictions
            relevant_preds = [p for name, p in confident_predictions.items() 
                            if 'research' in name or 'knowledge' in name]
        else:
            # Default: use all predictions
            relevant_preds = list(confident_predictions.values())
        
        if not relevant_preds:
            return 0.0
        
        # Calculate weighted average
        total_weight = 0.0
        weighted_strength = 0.0
        
        for pred in relevant_preds:
            weight = pred.get('confidence', 0.5)
            value = pred.get('value', 0.5)
            
            weighted_strength += value * weight
            total_weight += weight
        
        return weighted_strength / total_weight if total_weight > 0 else 0.0
    
    def _calculate_action_confidence(self, action: str, 
                                   confident_predictions: Dict[str, Dict[str, float]]) -> float:
        """Calculate confidence for motor action"""
        
        if not confident_predictions:
            return 0.0
        
        confidences = [p.get('confidence', 0.5) for p in confident_predictions.values()]
        return np.mean(confidences)
    
    def _get_action_episodic_context(self, action: str, 
                                   episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get episodic context relevant to action"""
        
        action_context = {}
        
        # Extract relevant episodic information for action
        for column_id, context in episodic_context.items():
            relevant_episodes = context.get('relevant_episodes', [])
            
            # Filter episodes relevant to this action
            action_relevant = []
            for episode in relevant_episodes:
                episode_content = episode.get('content', '').lower()
                
                if action == 'buy_signal' and any(word in episode_content for word in ['buy', 'purchase', 'invest']):
                    action_relevant.append(episode)
                elif action == 'sell_signal' and any(word in episode_content for word in ['sell', 'exit', 'reduce']):
                    action_relevant.append(episode)
                elif action == 'risk_alert' and any(word in episode_content for word in ['risk', 'danger', 'warning']):
                    action_relevant.append(episode)
            
            if action_relevant:
                action_context[column_id] = action_relevant
        
        return action_context
    
    def _generate_cortical_consensus(self, column_results: Dict[str, Dict[str, Any]], 
                                   experience: SensorimotorExperience) -> Dict[str, Any]:
        """Generate consensus across cortical columns"""
        
        # Collect predictions from all columns
        all_predictions = {}
        all_motor_outputs = {}
        confidence_scores = []
        
        for column_id, results in column_results.items():
            layer5_results = results.get('layer5_prediction', {})
            layer6_results = results.get('layer6_motor', {})
            
            # Collect predictions
            confident_predictions = layer5_results.get('confident_predictions', {})
            for pred_name, pred_data in confident_predictions.items():
                pred_key = f"{column_id}_{pred_name}"
                all_predictions[pred_key] = pred_data
                confidence_scores.append(pred_data.get('confidence', 0.5))
            
            # Collect motor outputs
            motor_outputs = layer6_results.get('motor_outputs', {})
            for action_name, action_data in motor_outputs.items():
                action_key = f"{column_id}_{action_name}"
                all_motor_outputs[action_key] = action_data
        
        # Calculate consensus metrics
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        agreement_level = self._calculate_agreement_level(column_results)
        
        # Generate consensus patterns
        consensus_patterns = self._extract_consensus_patterns(column_results)
        
        # Select consensus actions
        consensus_actions = self._select_consensus_actions(all_motor_outputs)
        
        return {
            'overall_confidence': overall_confidence,
            'agreement_level': agreement_level,
            'consensus_patterns': consensus_patterns,
            'consensus_actions': consensus_actions,
            'total_predictions': len(all_predictions),
            'total_actions': len(consensus_actions),
            'participating_columns': list(column_results.keys())
        }
    
    def _calculate_agreement_level(self, column_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate agreement level across columns"""
        
        if len(column_results) < 2:
            return 1.0
        
        # Compare predictions across columns
        prediction_similarities = []
        
        column_predictions = {}
        for column_id, results in column_results.items():
            layer5_results = results.get('layer5_prediction', {})
            confident_predictions = layer5_results.get('confident_predictions', {})
            
            # Extract prediction values
            pred_values = []
            for pred_data in confident_predictions.values():
                pred_values.append(pred_data.get('value', 0.5))
            
            if pred_values:
                column_predictions[column_id] = np.mean(pred_values)
        
        # Calculate pairwise similarities
        column_ids = list(column_predictions.keys())
        for i in range(len(column_ids)):
            for j in range(i + 1, len(column_ids)):
                val1 = column_predictions[column_ids[i]]
                val2 = column_predictions[column_ids[j]]
                similarity = 1.0 - abs(val1 - val2)
                prediction_similarities.append(similarity)
        
        return np.mean(prediction_similarities) if prediction_similarities else 1.0
    
    def _extract_consensus_patterns(self, column_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract consensus patterns across columns"""
        
        consensus_patterns = {}
        
        # Aggregate spatial patterns
        spatial_patterns = {}
        for column_id, results in column_results.items():
            layer3_results = results.get('layer3_spatial', {})
            enhanced_spatial = layer3_results.get('enhanced_spatial', {})
            
            for map_name, encoding in enhanced_spatial.items():
                if map_name not in spatial_patterns:
                    spatial_patterns[map_name] = []
                spatial_patterns[map_name].append(encoding)
        
        # Average spatial patterns
        for map_name, encodings in spatial_patterns.items():
            if encodings:
                consensus_patterns[f'spatial_{map_name}'] = np.mean(encodings, axis=0)
        
        # Aggregate temporal patterns
        temporal_patterns = {}
        for column_id, results in column_results.items():
            layer4_results = results.get('layer4_temporal', {})
            patterns = layer4_results.get('temporal_patterns', {})
            
            for pattern_name, pattern_value in patterns.items():
                if pattern_name not in temporal_patterns:
                    temporal_patterns[pattern_name] = []
                temporal_patterns[pattern_name].append(pattern_value)
        
        # Average temporal patterns
        for pattern_name, values in temporal_patterns.items():
            if values:
                consensus_patterns[f'temporal_{pattern_name}'] = np.mean(values)
        
        return consensus_patterns
    
    def _select_consensus_actions(self, all_motor_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Select consensus actions from all motor outputs"""
        
        # Group actions by type (remove column prefix)
        action_groups = defaultdict(list)
        
        for action_key, action_data in all_motor_outputs.items():
            # Extract action type (remove column prefix)
            action_type = action_key.split('_', 1)[1] if '_' in action_key else action_key
            action_groups[action_type].append(action_data)
        
        # Select consensus actions
        consensus_actions = {}
        
        for action_type, action_list in action_groups.items():
            if len(action_list) >= 2:  # Require at least 2 columns to agree
                # Average strength and confidence
                avg_strength = np.mean([a['strength'] for a in action_list])
                avg_confidence = np.mean([a['confidence'] for a in action_list])
                
                consensus_actions[action_type] = {
                    'strength': avg_strength,
                    'confidence': avg_confidence,
                    'supporting_columns': len(action_list),
                    'consensus_type': 'multi_column'
                }
            elif len(action_list) == 1 and action_list[0]['strength'] > 0.8:
                # Include strong single-column actions
                consensus_actions[action_type] = {
                    **action_list[0],
                    'supporting_columns': 1,
                    'consensus_type': 'strong_single'
                }
        
        return consensus_actions
    
    def _update_column_learning_metrics(self, column: Enhanced6LayerCorticalColumn, 
                                       layer_results: Dict[str, Any]):
        """Update column learning metrics"""
        
        # Calculate prediction accuracy based on confidence and outcomes
        layer5_results = layer_results.get('layer5_prediction', {})
        confident_predictions = layer5_results.get('confident_predictions', {})
        
        if confident_predictions:
            avg_confidence = np.mean([p['confidence'] for p in confident_predictions.values()])
            
            # Update prediction accuracy with exponential moving average
            alpha = column.learning_rate
            column.prediction_accuracy = (1 - alpha) * column.prediction_accuracy + alpha * avg_confidence
        
        # Update episodic influence based on episodic prediction success
        episodic_predictions = layer5_results.get('episodic_predictions', {})
        if episodic_predictions:
            episodic_confidence = np.mean([p['confidence'] for p in episodic_predictions.values()])
            
            # Adjust episodic influence
            if episodic_confidence > column.prediction_accuracy:
                column.episodic_influence = min(1.0, column.episodic_influence + 0.05)
            else:
                column.episodic_influence = max(0.1, column.episodic_influence - 0.02)
        
        # Update timestamp
        column.last_updated = datetime.now().isoformat()
    
    def _calculate_domain_expertise(self) -> float:
        """Calculate overall domain expertise level"""
        
        if not self.columns:
            return 0.5
        
        expertise_scores = []
        
        for column in self.columns.values():
            # Base expertise on prediction accuracy and experience
            base_expertise = column.prediction_accuracy
            
            # Boost based on episodic influence (learned experience)
            episodic_boost = column.episodic_influence * 0.2
            
            # Combine
            column_expertise = min(1.0, base_expertise + episodic_boost)
            expertise_scores.append(column_expertise)
        
        return np.mean(expertise_scores)
    
    def _assess_episodic_integration_quality(self, episodic_context: Dict[str, Any]) -> float:
        """Assess quality of episodic integration"""
        
        if not episodic_context:
            return 0.0
        
        quality_factors = []
        
        for column_id, context in episodic_context.items():
            relevant_episodes = context.get('relevant_episodes', [])
            spatial_priors = context.get('spatial_priors', {})
            temporal_expectations = context.get('temporal_expectations', {})
            
            # Factor 1: Number of relevant episodes
            episode_factor = min(1.0, len(relevant_episodes) / 5.0)
            
            # Factor 2: Richness of spatial priors
            spatial_factor = min(1.0, len(spatial_priors) / 10.0)
            
            # Factor 3: Richness of temporal expectations
            temporal_factor = min(1.0, len(temporal_expectations) / 5.0)
            
            column_quality = np.mean([episode_factor, spatial_factor, temporal_factor])
            quality_factors.append(column_quality)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _update_global_predictions(self, consensus_result: Dict[str, Any], 
                                 experience: SensorimotorExperience):
        """Update global predictions based on consensus"""
        
        consensus_patterns = consensus_result.get('consensus_patterns', {})
        overall_confidence = consensus_result.get('overall_confidence', 0.5)
        
        # Update global prediction state
        prediction_key = f"{experience.domain}_{experience.timestamp[:16]}"
        
        self.global_predictions[prediction_key] = {
            'patterns': consensus_patterns,
            'confidence': overall_confidence,
            'experience_id': experience.experience_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Maintain global prediction history (keep last 100)
        if len(self.global_predictions) > 100:
            oldest_key = min(self.global_predictions.keys())
            del self.global_predictions[oldest_key]
    
    def _update_reference_frames(self, experience: SensorimotorExperience, 
                               consensus_result: Dict[str, Any], 
                               episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Update reference frames with new information"""
        
        updates = {}
        
        for column_id, column in self.columns.items():
            column_updates = {}
            
            # Update spatial maps
            spatial_locations = len(column.reference_frame.get('spatial_map', {}))
            column_updates['spatial_locations'] = spatial_locations
            
            # Update temporal sequence
            temporal_sequence = column.reference_frame.get('temporal_sequence', deque())
            column_updates['temporal_sequence_length'] = len(temporal_sequence)
            
            # Update episodic spatial contexts
            episodic_spatial = column.reference_frame.get('episodic_spatial_contexts', {})
            column_updates['episodic_spatial_contexts'] = len(episodic_spatial)
            
            # Update prediction matrix (simplified)
            consensus_patterns = consensus_result.get('consensus_patterns', {})
            if consensus_patterns:
                # Create simple prediction matrix from consensus patterns
                pattern_values = []
                for pattern_name, pattern_value in consensus_patterns.items():
                    if isinstance(pattern_value, np.ndarray):
                        pattern_values.extend(pattern_value.tolist())
                    else:
                        pattern_values.append(pattern_value)
                
                if pattern_values:
                    # Create prediction matrix (simplified as 1D array)
                    column.reference_frame['prediction_matrix'] = np.array(pattern_values[:10])  # Limit size
            
            updates[column_id] = column_updates
        
        return updates
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        
        column_stats = {}
        
        for column_id, column in self.columns.items():
            column_stats[column_id] = {
                'specialization': column.specialization,
                'prediction_accuracy': column.prediction_accuracy,
                'learning_rate': column.learning_rate,
                'episodic_influence': column.episodic_influence,
                'last_updated': column.last_updated,
                'reference_frame_size': len(column.reference_frame)
            }
        
        return {
            'domain': self.domain,
            'total_columns': len(self.columns),
            'column_statistics': column_stats,
            'global_predictions': len(self.global_predictions),
            'domain_expertise_level': self._calculate_domain_expertise(),
            'episodic_integration_enabled': self.episodic_memory_engine is not None
        }

# ============================================================================
# IDENTITY FORMATION MECHANISMS (COMPLETE ORIGINAL)
# ============================================================================

class EpisodicIdentityProcessor:
    """Enhanced identity processor with deep episodic integration"""
    
    def __init__(self, personality_seed: Dict[str, float]):
        self.personality_seed = personality_seed
        self.identity_history = deque(maxlen=1000)
        self.narrative_coherence_tracker = deque(maxlen=100)
        self.identity_milestones = []
        
        # Initialize personality state from seed
        self.current_personality_state = AdvancedPersonalityState(
            traits_big5=self._initialize_big5_traits(personality_seed),
            cognitive_style=self._initialize_cognitive_style(personality_seed),
            core_value_system=self._initialize_value_system(personality_seed),
            narrative_themes=self._initialize_narrative_themes(personality_seed),
            identity_anchors=self._initialize_identity_anchors(personality_seed),
            goal_hierarchy=self._initialize_goal_hierarchy(personality_seed),
            emotional_patterns=self._initialize_emotional_patterns(personality_seed),
            social_preferences=self._initialize_social_preferences(personality_seed),
            narrative_coherence=0.7,
            identity_stability=0.6,
            development_stage="initial_formation",
            last_updated=datetime.now().isoformat(),
            episodic_narrative_depth=0.0,
            episodic_identity_milestones=[],
            cross_episodic_coherence=0.0
        )
        
        # Initialize supporting processors
        self.narrator = ContinuousNarrator()
        self.identity_comparer = IdentityComparer()
        self.temporal_integrator = TemporalIntegrator()
        self.meaning_maker = MeaningMaker()
        
        # Episodic memory integration
        self.episodic_memory_engine = None  # Will be set by main system
        self.episodic_influence_weights = {
            'personality_evolution': 0.4,
            'narrative_development': 0.3,
            'value_refinement': 0.3
        }
    
    def set_episodic_memory_engine(self, episodic_engine):
        """Set the episodic memory engine for integration"""
        self.episodic_memory_engine = episodic_engine
    
    def process_experience_with_episodes(self, experience: SensorimotorExperience, 
                                       cortical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process experience for identity formation with episodic integration"""
        
        # Retrieve relevant episodic context
        episodic_context = self._get_identity_episodic_context(experience)
        
        # Generate narrative from experience
        narrative_result = self.narrator.generate_narrative(experience, cortical_result, episodic_context)
        
        # Analyze identity implications
        identity_analysis = self._analyze_identity_implications(experience, cortical_result, narrative_result, episodic_context)
        
        # Update personality state
        personality_update = self._update_personality_state(experience, identity_analysis, episodic_context)
        
        # Assess coherence with episodic memory
        coherence_assessment = self._assess_episodic_coherence(experience, identity_analysis, episodic_context)
        
        # Temporal integration across episodes
        temporal_integration = self.temporal_integrator.integrate_across_episodes(
            experience, identity_analysis, episodic_context
        )
        
        # Meaning making with episodic depth
        meaning_result = self.meaning_maker.extract_meaning_with_episodes(
            experience, identity_analysis, episodic_context
        )
        
        # Update identity history
        identity_record = {
            'timestamp': experience.timestamp,
            'experience_id': experience.experience_id,
            'narrative': narrative_result,
            'identity_analysis': identity_analysis,
            'personality_update': personality_update,
            'coherence_score': coherence_assessment['overall_coherence'],
            'episodic_influence': self._calculate_episodic_influence_metrics(episodic_context)
        }
        self.identity_history.append(identity_record)
        
        # Check for identity milestones
        milestone_check = self._check_identity_milestones(identity_analysis, coherence_assessment)
        
        return {
            'personality_state': asdict(self.current_personality_state),
            'narrative_result': narrative_result,
            'identity_analysis': identity_analysis,
            'coherence_assessment': coherence_assessment,
            'temporal_integration': temporal_integration,
            'meaning_result': meaning_result,
            'milestone_check': milestone_check,
            'episodic_influence_metrics': self._calculate_episodic_influence_metrics(episodic_context),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _get_identity_episodic_context(self, experience: SensorimotorExperience) -> Dict[str, Any]:
        """Retrieve episodic context relevant to identity formation"""
        
        if not self.episodic_memory_engine:
            return {}
        
        # Retrieve episodes relevant to identity
        episodic_retrieval = self.episodic_memory_engine.retrieve_episodic_context(
            experience, max_context_tokens=3000
        )
        
        episodes = episodic_retrieval.get('episodes', [])
        
        # Extract identity-relevant information
        identity_episodes = []
        personality_patterns = defaultdict(list)
        narrative_themes = defaultdict(int)
        value_expressions = defaultdict(list)
        
        for episode in episodes:
            # Check for identity-relevant content
            if self._is_identity_relevant_episode(episode, experience):
                identity_episodes.append(episode)
                
                # Extract personality patterns
                personality_state = episode.get('personality_state', {})
                for trait, value in personality_state.items():
                    if isinstance(value, (int, float)):
                        personality_patterns[trait].append(value)
                
                # Extract narrative themes
                narrative_content = episode.get('narrative_themes', '')
                if narrative_content:
                    words = narrative_content.lower().split()
                    for word in words:
                        if len(word) > 4:
                            narrative_themes[word] += 1
                
                # Extract value expressions
                episode_content = episode.get('content', '').lower()
                values = ['honesty', 'fairness', 'growth', 'excellence', 'innovation', 'collaboration']
                for value in values:
                    if value in episode_content:
                        value_expressions[value].append(episode.get('timestamp', ''))
        
        return {
            'identity_episodes': identity_episodes,
            'personality_patterns': dict(personality_patterns),
            'narrative_themes': dict(narrative_themes),
            'value_expressions': dict(value_expressions),
            'total_relevant_episodes': len(identity_episodes)
        }
    
    def _is_identity_relevant_episode(self, episode: Dict, experience: SensorimotorExperience) -> bool:
        """Check if episode is relevant to identity formation"""
        
        relevance_indicators = 0
        
        # Check for identity-related keywords
        identity_keywords = ['believe', 'value', 'important', 'principle', 'character', 'identity', 'personality']
        episode_content = episode.get('content', '').lower()
        
        for keyword in identity_keywords:
            if keyword in episode_content:
                relevance_indicators += 1
        
        # Check for personality state information
        if episode.get('personality_state'):
            relevance_indicators += 2
        
        # Check for narrative themes
        if episode.get('narrative_themes'):
            relevance_indicators += 1
        
        # Check for domain consistency
        if episode.get('domain') == experience.domain:
            relevance_indicators += 1
        
        # Check for identity coherence scores
        if episode.get('identity_coherence', 0) > 0.7:
            relevance_indicators += 1
        
        return relevance_indicators >= 2
    
    def _analyze_identity_implications(self, experience: SensorimotorExperience, 
                                     cortical_result: Dict[str, Any], 
                                     narrative_result: Dict[str, Any],
                                     episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze identity implications of experience with episodic context"""
        
        identity_analysis = {}
        
        # Extract identity-relevant features from experience
        content = experience.content.lower()
        
        # Trait implications
        trait_implications = self._analyze_trait_implications(content, cortical_result, episodic_context)
        identity_analysis['trait_implications'] = trait_implications
        
        # Value expressions
        value_expressions = self._analyze_value_expressions(content, episodic_context)
        identity_analysis['value_expressions'] = value_expressions
        
        # Narrative connection
        narrative_connection = self._analyze_narrative_connection(
            narrative_result, episodic_context
        )
        identity_analysis['narrative_connection'] = narrative_connection
        
        # Goal relevance
        goal_relevance = self._analyze_goal_relevance(content, cortical_result, episodic_context)
        identity_analysis['goal_relevance'] = goal_relevance
        
        # Emotional pattern
        emotional_pattern = self._analyze_emotional_pattern(experience, episodic_context)
        identity_analysis['emotional_pattern'] = emotional_pattern
        
        # Social context
        social_context = self._analyze_social_context(content, episodic_context)
        identity_analysis['social_context'] = social_context
        
        return identity_analysis
    
    def _analyze_trait_implications(self, content: str, cortical_result: Dict[str, Any], 
                                  episodic_context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze Big Five trait implications"""
        
        trait_implications = {}
        
        # Openness to Experience
        openness_indicators = ['new', 'novel', 'creative', 'innovative', 'explore', 'discover']
        openness_score = sum(1 for word in openness_indicators if word in content) / len(openness_indicators)
        
        # Apply episodic context
        personality_patterns = episodic_context.get('personality_patterns', {})
        if 'openness' in personality_patterns:
            historical_openness = np.mean(personality_patterns['openness'])
            openness_score = 0.7 * openness_score + 0.3 * historical_openness
        
        trait_implications['openness'] = openness_score
        
        # Conscientiousness
        conscientiousness_indicators = ['careful', 'thorough', 'systematic', 'organized', 'diligent']
        conscientiousness_score = sum(1 for word in conscientiousness_indicators if word in content) / len(conscientiousness_indicators)
        
        if 'conscientiousness' in personality_patterns:
            historical_conscientiousness = np.mean(personality_patterns['conscientiousness'])
            conscientiousness_score = 0.7 * conscientiousness_score + 0.3 * historical_conscientiousness
        
        trait_implications['conscientiousness'] = conscientiousness_score
        
        # Extraversion
        extraversion_indicators = ['social', 'outgoing', 'energetic', 'assertive', 'confident']
        extraversion_score = sum(1 for word in extraversion_indicators if word in content) / len(extraversion_indicators)
        
        if 'extraversion' in personality_patterns:
            historical_extraversion = np.mean(personality_patterns['extraversion'])
            extraversion_score = 0.7 * extraversion_score + 0.3 * historical_extraversion
        
        trait_implications['extraversion'] = extraversion_score
        
        # Agreeableness
        agreeableness_indicators = ['cooperative', 'helpful', 'collaborative', 'supportive', 'kind']
        agreeableness_score = sum(1 for word in agreeableness_indicators if word in content) / len(agreeableness_indicators)
        
        if 'agreeableness' in personality_patterns:
            historical_agreeableness = np.mean(personality_patterns['agreeableness'])
            agreeableness_score = 0.7 * agreeableness_score + 0.3 * historical_agreeableness
        
        trait_implications['agreeableness'] = agreeableness_score
        
        # Neuroticism
        neuroticism_indicators = ['stress', 'worry', 'anxiety', 'uncertainty', 'concern']
        neuroticism_score = sum(1 for word in neuroticism_indicators if word in content) / len(neuroticism_indicators)
        
        if 'neuroticism' in personality_patterns:
            historical_neuroticism = np.mean(personality_patterns['neuroticism'])
            neuroticism_score = 0.7 * neuroticism_score + 0.3 * historical_neuroticism
        
        trait_implications['neuroticism'] = neuroticism_score
        
        return trait_implications
    
    def _analyze_value_expressions(self, content: str, episodic_context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze value expressions in content with episodic context"""
        
        value_expressions = {}
        
        # Core values and their indicators
        value_indicators = {
            'honesty': ['honest', 'truthful', 'transparent', 'genuine', 'authentic'],
            'fairness': ['fair', 'just', 'equitable', 'balanced', 'impartial'],
            'excellence': ['excellent', 'quality', 'best', 'superior', 'outstanding'],
            'growth': ['learn', 'develop', 'improve', 'progress', 'advance'],
            'innovation': ['innovative', 'creative', 'new', 'breakthrough', 'pioneering'],
            'collaboration': ['together', 'team', 'cooperative', 'partnership', 'collective']
        }
        
        for value, indicators in value_indicators.items():
            current_expression = sum(1 for word in indicators if word in content) / len(indicators)
            
            # Apply episodic context
            value_expressions_history = episodic_context.get('value_expressions', {})
            if value in value_expressions_history:
                # Value has been expressed before - show consistency
                historical_frequency = len(value_expressions_history[value])
                consistency_boost = min(0.3, historical_frequency * 0.05)
                current_expression += consistency_boost
            
            value_expressions[value] = min(1.0, current_expression)
        
        return value_expressions
    
    def _analyze_narrative_connection(self, narrative_result: Dict[str, Any], 
                                    episodic_context: Dict[str, Any]) -> str:
        """Analyze narrative connection with episodic themes"""
        
        current_narrative = narrative_result.get('narrative_text', '')
        
        # Extract themes from current narrative
        current_themes = set(word.lower() for word in current_narrative.split() if len(word) > 4)
        
        # Get historical narrative themes
        historical_themes = episodic_context.get('narrative_themes', {})
        
        # Find connections
        theme_connections = []
        for theme, frequency in historical_themes.items():
            if theme in current_themes and frequency > 2:
                theme_connections.append(f"Continues theme of '{theme}' (appeared {frequency} times)")
        
        # Analyze narrative evolution
        narrative_evolution = []
        identity_episodes = episodic_context.get('identity_episodes', [])
        
        if len(identity_episodes) > 2:
            recent_themes = set()
            for episode in identity_episodes[-3:]:
                episode_narrative = episode.get('narrative_themes', '')
                recent_themes.update(word.lower() for word in episode_narrative.split() if len(word) > 4)
            
            new_themes = current_themes - recent_themes
            if new_themes:
                narrative_evolution.append(f"Introduces new themes: {', '.join(list(new_themes)[:3])}")
            
            continuing_themes = current_themes & recent_themes
            if continuing_themes:
                narrative_evolution.append(f"Continues recent themes: {', '.join(list(continuing_themes)[:3])}")
        
        # Construct narrative connection
        connection_parts = []
        if theme_connections:
            connection_parts.append("Historical: " + "; ".join(theme_connections[:2]))
        if narrative_evolution:
            connection_parts.append("Recent: " + "; ".join(narrative_evolution))
        
        return " | ".join(connection_parts) if connection_parts else "New narrative exploration"
    
    def _analyze_goal_relevance(self, content: str, cortical_result: Dict[str, Any], 
                              episodic_context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze goal relevance with episodic context"""
        
        goal_relevance = {}
        
        # Goal categories and indicators
        goal_indicators = {
            'mastery': ['master', 'expert', 'skilled', 'proficient', 'competent'],
            'achievement': ['achieve', 'accomplish', 'succeed', 'complete', 'finish'],
            'understanding': ['understand', 'comprehend', 'grasp', 'insight', 'clarity'],
            'impact': ['impact', 'influence', 'effect', 'change', 'difference'],
            'connection': ['connect', 'relationship', 'bond', 'network', 'community']
        }
        
        for goal, indicators in goal_indicators.items():
            relevance_score = sum(1 for word in indicators if word in content) / len(indicators)
            
            # Boost from cortical predictions
            cortical_consensus = cortical_result.get('consensus', {})
            if cortical_consensus.get('overall_confidence', 0) > 0.7:
                relevance_score *= 1.2  # High confidence predictions boost goal relevance
            
            # Historical goal consistency
            identity_episodes = episodic_context.get('identity_episodes', [])
            historical_goal_mentions = 0
            for episode in identity_episodes:
                episode_content = episode.get('content', '').lower()
                if any(indicator in episode_content for indicator in indicators):
                    historical_goal_mentions += 1
            
            if historical_goal_mentions > 0:
                consistency_factor = min(0.3, historical_goal_mentions * 0.1)
                relevance_score += consistency_factor
            
            goal_relevance[goal] = min(1.0, relevance_score)
        
        return goal_relevance
    
    def _analyze_emotional_pattern(self, experience: SensorimotorExperience, 
                                 episodic_context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze emotional patterns with episodic context"""
        
        emotional_pattern = {}
        
        # Extract current emotional indicators
        content = experience.content.lower()
        
        emotion_indicators = {
            'curiosity': ['curious', 'wonder', 'explore', 'investigate', 'discover'],
            'confidence': ['confident', 'sure', 'certain', 'assured', 'convinced'],
            'enthusiasm': ['excited', 'enthusiastic', 'eager', 'passionate', 'motivated'],
            'concern': ['concerned', 'worried', 'cautious', 'careful', 'attentive'],
            'satisfaction': ['satisfied', 'pleased', 'content', 'fulfilled', 'accomplished']
        }
        
        for emotion, indicators in emotion_indicators.items():
            current_intensity = sum(1 for word in indicators if word in content) / len(indicators)
            
            # Historical emotional patterns
            identity_episodes = episodic_context.get('identity_episodes', [])
            historical_intensities = []
            
            for episode in identity_episodes:
                episode_content = episode.get('content', '').lower()
                episode_intensity = sum(1 for word in indicators if word in episode_content) / len(indicators)
                if episode_intensity > 0:
                    historical_intensities.append(episode_intensity)
            
            # Blend current with historical pattern
            if historical_intensities:
                historical_avg = np.mean(historical_intensities)
                blended_intensity = 0.7 * current_intensity + 0.3 * historical_avg
            else:
                blended_intensity = current_intensity
            
            emotional_pattern[emotion] = blended_intensity
        
        return emotional_pattern
    
    def _analyze_social_context(self, content: str, episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social context with episodic history"""
        
        social_context = {}
        
        # Social indicators
        social_indicators = {
            'collaborative': ['team', 'together', 'collaborate', 'cooperation', 'partnership'],
            'leadership': ['lead', 'guide', 'direct', 'manage', 'coordinate'],
            'supportive': ['help', 'support', 'assist', 'aid', 'encourage'],
            'independent': ['independent', 'autonomous', 'self-reliant', 'individual', 'solo']
        }
        
        for social_style, indicators in social_indicators.items():
            current_expression = sum(1 for word in indicators if word in content) / len(indicators)
            social_context[social_style] = current_expression
        
        # Analyze social consistency across episodes
        identity_episodes = episodic_context.get('identity_episodes', [])
        if len(identity_episodes) > 3:
            social_consistency = {}
            
            for social_style, indicators in social_indicators.items():
                style_expressions = []
                for episode in identity_episodes:
                    episode_content = episode.get('content', '').lower()
                    expression = sum(1 for word in indicators if word in episode_content) / len(indicators)
                    style_expressions.append(expression)
                
                if style_expressions:
                    consistency = 1.0 - np.std(style_expressions)  # Higher std = lower consistency
                    social_consistency[social_style] = max(0.0, consistency)
            
            social_context['consistency_across_episodes'] = social_consistency
        
        return social_context
    
    def _update_personality_state(self, experience: SensorimotorExperience, 
                                identity_analysis: Dict[str, Any], 
                                episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Update personality state based on analysis and episodic context"""
        
        update_result = {'changes': {}, 'magnitude': 0.0}
        
        # Learning rate for personality updates (slower with more episodes)
        num_episodes = episodic_context.get('total_relevant_episodes', 0)
        base_learning_rate = 0.1
        learning_rate = base_learning_rate / (1.0 + num_episodes * 0.01)  # Decrease with experience
        
        # Update Big Five traits
        trait_implications = identity_analysis.get('trait_implications', {})
        for trait, implication in trait_implications.items():
            if trait in self.current_personality_state.traits_big5:
                old_value = self.current_personality_state.traits_big5[trait]
                new_value = old_value + (implication - old_value) * learning_rate
                new_value = max(0.0, min(1.0, new_value))  # Clamp to [0, 1]
                
                change_magnitude = abs(new_value - old_value)
                if change_magnitude > 0.01:  # Only record significant changes
                    self.current_personality_state.traits_big5[trait] = new_value
                    update_result['changes'][f'trait_{trait}'] = {
                        'old': old_value,
                        'new': new_value,
                        'change': change_magnitude
                    }
        
        # Update core values
        value_expressions = identity_analysis.get('value_expressions', {})
        for value, expression in value_expressions.items():
            if value in self.current_personality_state.core_value_system:
                old_value = self.current_personality_state.core_value_system[value]
                new_value = old_value + (expression - old_value) * learning_rate * 0.5  # Slower value change
                new_value = max(0.0, min(1.0, new_value))
                
                change_magnitude = abs(new_value - old_value)
                if change_magnitude > 0.01:
                    self.current_personality_state.core_value_system[value] = new_value
                    update_result['changes'][f'value_{value}'] = {
                        'old': old_value,
                        'new': new_value,
                        'change': change_magnitude
                    }
        
        # Update emotional patterns
        emotional_pattern = identity_analysis.get('emotional_pattern', {})
        for emotion, intensity in emotional_pattern.items():
            if emotion in self.current_personality_state.emotional_patterns:
                old_value = self.current_personality_state.emotional_patterns[emotion]
                new_value = old_value + (intensity - old_value) * learning_rate
                new_value = max(0.0, min(1.0, new_value))
                
                change_magnitude = abs(new_value - old_value)
                if change_magnitude > 0.01:
                    self.current_personality_state.emotional_patterns[emotion] = new_value
                    update_result['changes'][f'emotion_{emotion}'] = {
                        'old': old_value,
                        'new': new_value,
                        'change': change_magnitude
                    }
        
        # Update narrative themes
        narrative_connection = identity_analysis.get('narrative_connection', '')
        if narrative_connection and narrative_connection != "New narrative exploration":
            # Extract new themes from narrative connection
            new_themes = self._extract_themes_from_narrative(narrative_connection)
            for theme in new_themes:
                if theme not in self.current_personality_state.narrative_themes:
                    self.current_personality_state.narrative_themes.append(theme)
                    update_result['changes'][f'new_theme_{theme}'] = True
        
        # Update episodic identity milestones
        milestone_indicator = self._check_for_milestone_indicators(identity_analysis, episodic_context)
        if milestone_indicator:
            milestone = f"{experience.timestamp[:10]}_{milestone_indicator}"
            if milestone not in self.current_personality_state.episodic_identity_milestones:
                self.current_personality_state.episodic_identity_milestones.append(milestone)
                update_result['changes']['new_milestone'] = milestone
        
        # Calculate overall update magnitude
        change_magnitudes = [change['change'] for change in update_result['changes'].values() 
                           if isinstance(change, dict) and 'change' in change]
        update_result['magnitude'] = np.mean(change_magnitudes) if change_magnitudes else 0.0
        
        # Update metadata
        self.current_personality_state.last_updated = datetime.now().isoformat()
        
        return update_result
    
    def _assess_episodic_coherence(self, experience: SensorimotorExperience, 
                                 identity_analysis: Dict[str, Any], 
                                 episodic_context: Dict[str, Any]) -> Dict[str, float]:
        """Assess coherence with episodic memory"""
        
        coherence_scores = {}
        
        # Trait coherence across episodes
        trait_coherence = self._assess_trait_coherence(identity_analysis, episodic_context)
        coherence_scores['trait_coherence'] = trait_coherence
        
        # Value coherence
        value_coherence = self._assess_value_coherence(identity_analysis, episodic_context)
        coherence_scores['value_coherence'] = value_coherence
        
        # Narrative coherence
        narrative_coherence = self._assess_narrative_coherence(identity_analysis, episodic_context)
        coherence_scores['narrative_coherence'] = narrative_coherence
        
        # Emotional coherence
        emotional_coherence = self._assess_emotional_coherence(identity_analysis, episodic_context)
        coherence_scores['emotional_coherence'] = emotional_coherence
        
        # Social coherence
        social_coherence = self._assess_social_coherence(identity_analysis, episodic_context)
        coherence_scores['social_coherence'] = social_coherence
        
        # Overall coherence
        overall_coherence = np.mean(list(coherence_scores.values()))
        coherence_scores['overall_coherence'] = overall_coherence
        
        # Update personality state coherence metrics
        self.current_personality_state.narrative_coherence = narrative_coherence
        self.current_personality_state.cross_episodic_coherence = overall_coherence
        
        # Update coherence tracking
        self.narrative_coherence_tracker.append({
            'timestamp': experience.timestamp,
            'overall_coherence': overall_coherence,
            'component_coherences': coherence_scores
        })
        
        return coherence_scores
    
    def _assess_trait_coherence(self, identity_analysis: Dict[str, Any], 
                              episodic_context: Dict[str, Any]) -> float:
        """Assess trait coherence across episodes"""
        
        trait_implications = identity_analysis.get('trait_implications', {})
        personality_patterns = episodic_context.get('personality_patterns', {})
        
        if not trait_implications or not personality_patterns:
            return 0.5  # Neutral coherence
        
        coherence_scores = []
        
        for trait, current_implication in trait_implications.items():
            if trait in personality_patterns:
                historical_values = personality_patterns[trait]
                historical_avg = np.mean(historical_values)
                historical_std = np.std(historical_values) if len(historical_values) > 1 else 0.2
                
                # Calculate coherence based on consistency
                if historical_std < 0.3:  # Low variance = high coherence
                    trait_coherence = 1.0 - abs(current_implication - historical_avg)
                else:  # High variance = lower coherence baseline
                    trait_coherence = 0.5 - abs(current_implication - historical_avg) * 0.5
                
                coherence_scores.append(max(0.0, trait_coherence))
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _assess_value_coherence(self, identity_analysis: Dict[str, Any], 
                              episodic_context: Dict[str, Any]) -> float:
        """Assess value coherence across episodes"""
        
        value_expressions = identity_analysis.get('value_expressions', {})
        value_expressions_history = episodic_context.get('value_expressions', {})
        
        if not value_expressions or not value_expressions_history:
            return 0.5
        
        coherence_scores = []
        
        for value, current_expression in value_expressions.items():
            if value in value_expressions_history:
                historical_expressions = len(value_expressions_history[value])
                
                # Values should be consistent when expressed
                if current_expression > 0.3 and historical_expressions > 0:
                    coherence_scores.append(0.9)  # High coherence for consistent value expression
                elif current_expression <= 0.3 and historical_expressions == 0:
                    coherence_scores.append(0.8)  # Coherent absence of value
                else:
                    coherence_scores.append(0.4)  # Inconsistent value expression
            else:
                # New value expression
                coherence_scores.append(0.6)  # Moderate coherence for new values
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _assess_narrative_coherence(self, identity_analysis: Dict[str, Any], 
                                  episodic_context: Dict[str, Any]) -> float:
        """Assess narrative coherence across episodes"""
        
        narrative_connection = identity_analysis.get('narrative_connection', '')
        narrative_themes = episodic_context.get('narrative_themes', {})
        
        if not narrative_connection or narrative_connection == "New narrative exploration":
            return 0.4  # Lower coherence for completely new narratives
        
        # Analyze connection quality
        coherence_indicators = []
        
        # Check for historical theme continuation
        if "Continues theme" in narrative_connection:
            coherence_indicators.append(0.8)
        
        # Check for recent theme continuation
        if "Continues recent themes" in narrative_connection:
            coherence_indicators.append(0.7)
        
        # Check for balanced evolution (continues some, introduces some)
        if "Continues" in narrative_connection and "Introduces" in narrative_connection:
            coherence_indicators.append(0.9)  # Balanced evolution = high coherence
        
        # Check theme consistency
        total_themes = len(narrative_themes)
        if total_themes > 0:
            # Higher coherence if we have established themes
            theme_consistency = min(1.0, total_themes / 10.0)
            coherence_indicators.append(theme_consistency)
        
        return np.mean(coherence_indicators) if coherence_indicators else 0.5
    
    def _assess_emotional_coherence(self, identity_analysis: Dict[str, Any], 
                                  episodic_context: Dict[str, Any]) -> float:
        """Assess emotional coherence across episodes"""
        
        emotional_pattern = identity_analysis.get('emotional_pattern', {})
        identity_episodes = episodic_context.get('identity_episodes', [])
        
        if not emotional_pattern or len(identity_episodes) < 3:
            return 0.5
        
        coherence_scores = []
        
        # Analyze emotional consistency
        for emotion, current_intensity in emotional_pattern.items():
            historical_intensities = []
            
            for episode in identity_episodes:
                episode_content = episode.get('content', '').lower()
                # Simple emotion detection for historical comparison
                if emotion == 'curiosity' and any(word in episode_content for word in ['curious', 'explore', 'discover']):
                    historical_intensities.append(0.7)
                elif emotion == 'confidence' and any(word in episode_content for word in ['confident', 'sure', 'certain']):
                    historical_intensities.append(0.7)
                # Add more emotion detection as needed
            
            if historical_intensities:
                historical_avg = np.mean(historical_intensities)
                emotional_coherence = 1.0 - abs(current_intensity - historical_avg)
                coherence_scores.append(max(0.0, emotional_coherence))
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _assess_social_coherence(self, identity_analysis: Dict[str, Any], 
                               episodic_context: Dict[str, Any]) -> float:
        """Assess social coherence across episodes"""
        
        social_context = identity_analysis.get('social_context', {})
        
        # Check for consistency metrics if available
        consistency_data = social_context.get('consistency_across_episodes', {})
        
        if consistency_data:
            return np.mean(list(consistency_data.values()))
        else:
            return 0.5  # Neutral coherence if no consistency data
    
    def _calculate_episodic_influence_metrics(self, episodic_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics for episodic influence on identity"""
        
        metrics = {}
        
        # Episodic influence score based on available context
        total_episodes = episodic_context.get('total_relevant_episodes', 0)
        personality_patterns = episodic_context.get('personality_patterns', {})
        narrative_themes = episodic_context.get('narrative_themes', {})
        value_expressions = episodic_context.get('value_expressions', {})
        
        # Base influence score
        base_influence = min(1.0, total_episodes / 10.0)  # Max influence at 10+ episodes
        metrics['episodic_influence_score'] = base_influence
        
        # Memory depth factor
        memory_depth = (
            len(personality_patterns) * 0.4 +
            len(narrative_themes) * 0.3 +
            len(value_expressions) * 0.3
        ) / 10.0  # Normalize
        metrics['memory_depth_factor'] = min(1.0, memory_depth)
        
        # Boundary influence (how much episode boundaries affect identity)
        boundary_episodes = [ep for ep in episodic_context.get('identity_episodes', []) 
                           if ep.get('is_boundary', False)]
        boundary_influence = min(1.0, len(boundary_episodes) / 5.0)
        metrics['boundary_influence'] = boundary_influence
        
        return metrics
    
    def _check_identity_milestones(self, identity_analysis: Dict[str, Any], 
                                 coherence_assessment: Dict[str, float]) -> Dict[str, Any]:
        """Check for identity development milestones"""
        
        milestone_check = {'milestones_reached': [], 'development_stage': self.current_personality_state.development_stage}
        
        overall_coherence = coherence_assessment.get('overall_coherence', 0.5)
        
        # Milestone 1: Initial Coherence (>0.6 overall coherence)
        if overall_coherence > 0.6 and self.current_personality_state.development_stage == "initial_formation":
            milestone_check['milestones_reached'].append("initial_coherence_achieved")
            self.current_personality_state.development_stage = "coherence_building"
            milestone_check['development_stage'] = "coherence_building"
        
        # Milestone 2: Stable Identity (>0.8 coherence for 5+ episodes)
        if overall_coherence > 0.8:
            recent_coherence = [entry['overall_coherence'] for entry in list(self.narrative_coherence_tracker)[-5:]]
            if len(recent_coherence) >= 5 and all(c > 0.8 for c in recent_coherence):
                if self.current_personality_state.development_stage != "stable_identity":
                    milestone_check['milestones_reached'].append("stable_identity_achieved")
                    self.current_personality_state.development_stage = "stable_identity"
                    milestone_check['development_stage'] = "stable_identity"
        
        # Milestone 3: Value Integration (consistent value expression)
        value_expressions = identity_analysis.get('value_expressions', {})
        strong_values = [v for v in value_expressions.values() if v > 0.7]
        if len(strong_values) >= 3:
            if "value_integration_achieved" not in [m.split('_')[2] for m in self.current_personality_state.episodic_identity_milestones]:
                milestone_check['milestones_reached'].append("value_integration_achieved")
        
        # Milestone 4: Narrative Depth (rich narrative connections)
        narrative_connection = identity_analysis.get('narrative_connection', '')
        if len(narrative_connection) > 100 and "Continues theme" in narrative_connection:
            if "narrative_depth_achieved" not in [m.split('_')[2] for m in self.current_personality_state.episodic_identity_milestones]:
                milestone_check['milestones_reached'].append("narrative_depth_achieved")
        
        # Update personality state with new milestones
        for milestone in milestone_check['milestones_reached']:
            milestone_entry = f"{datetime.now().isoformat()[:10]}_{milestone}"
            if milestone_entry not in self.current_personality_state.episodic_identity_milestones:
                self.current_personality_state.episodic_identity_milestones.append(milestone_entry)
                self.identity_milestones.append({
                    'milestone': milestone,
                    'timestamp': datetime.now().isoformat(),
                    'coherence_score': overall_coherence,
                    'identity_analysis': identity_analysis
                })
        
        return milestone_check
    
    def _check_for_milestone_indicators(self, identity_analysis: Dict[str, Any], 
                                      episodic_context: Dict[str, Any]) -> Optional[str]:
        """Check for milestone indicators in current analysis"""
        
        # Strong value expression
        value_expressions = identity_analysis.get('value_expressions', {})
        if any(v > 0.8 for v in value_expressions.values()):
            return "strong_value_expression"
        
        # Significant trait shift
        trait_implications = identity_analysis.get('trait_implications', {})
        current_traits = self.current_personality_state.traits_big5
        for trait, implication in trait_implications.items():
            if trait in current_traits:
                if abs(implication - current_traits[trait]) > 0.3:
                    return f"significant_{trait}_shift"
        
        # Rich narrative development
        narrative_connection = identity_analysis.get('narrative_connection', '')
        if len(narrative_connection) > 150:
            return "rich_narrative_development"
        
        return None
    
    def _extract_themes_from_narrative(self, narrative_connection: str) -> List[str]:
        """Extract new themes from narrative connection"""
        
        themes = []
        
        # Look for theme mentions in the narrative connection
        if "new themes:" in narrative_connection.lower():
            themes_part = narrative_connection.lower().split("new themes:")[1]
            if themes_part:
                # Extract theme words
                theme_words = themes_part.replace(",", " ").split()
                themes = [word.strip() for word in theme_words if len(word) > 4][:3]  # Limit to 3 themes
        
        return themes
    
    def _initialize_big5_traits(self, personality_seed: Dict[str, float]) -> Dict[str, float]:
        """Initialize Big Five traits from personality seed"""
        
        return {
            'openness': personality_seed.get('openness', 0.7),
            'conscientiousness': personality_seed.get('conscientiousness', 0.8),
            'extraversion': personality_seed.get('extraversion', 0.6),
            'agreeableness': personality_seed.get('agreeableness', 0.7),
            'neuroticism': personality_seed.get('neuroticism', 0.3)
        }
    
    def _initialize_cognitive_style(self, personality_seed: Dict[str, float]) -> Dict[str, float]:
        """Initialize cognitive style from personality seed"""
        
        return {
            'analytical_thinking': personality_seed.get('analytical', 0.8),
            'creative_thinking': personality_seed.get('creative', 0.7),
            'systematic_processing': personality_seed.get('systematic', 0.75),
            'intuitive_processing': personality_seed.get('intuitive', 0.6),
            'detail_orientation': personality_seed.get('detail_oriented', 0.8)
        }
    
    def _initialize_value_system(self, personality_seed: Dict[str, float]) -> Dict[str, float]:
        """Initialize core value system"""
        
        return {
            'honesty': 0.9,
            'fairness': 0.85,
            'excellence': 0.8,
            'growth': 0.9,
            'innovation': 0.75,
            'collaboration': 0.7
        }
    
    def _initialize_narrative_themes(self, personality_seed: Dict[str, float]) -> List[str]:
        """Initialize narrative themes"""
        
        return ['learning', 'analysis', 'understanding', 'improvement']
    
    def _initialize_identity_anchors(self, personality_seed: Dict[str, float]) -> List[str]:
        """Initialize identity anchors"""
        
        return ['analytical_thinker', 'continuous_learner', 'problem_solver']
    
    def _initialize_goal_hierarchy(self, personality_seed: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Initialize goal hierarchy"""
        
        return {
            'mastery': {'priority': 0.9, 'progress': 0.3},
            'understanding': {'priority': 0.85, 'progress': 0.4},
            'impact': {'priority': 0.7, 'progress': 0.2},
            'connection': {'priority': 0.6, 'progress': 0.3}
        }
    
    def _initialize_emotional_patterns(self, personality_seed: Dict[str, float]) -> Dict[str, float]:
        """Initialize emotional patterns"""
        
        return {
            'curiosity': 0.8,
            'confidence': 0.7,
            'enthusiasm': 0.75,
            'concern': 0.4,
            'satisfaction': 0.6
        }
    
    def _initialize_social_preferences(self, personality_seed: Dict[str, float]) -> Dict[str, float]:
        """Initialize social preferences"""
        
        return {
            'collaborative': 0.75,
            'leadership': 0.6,
            'supportive': 0.8,
            'independent': 0.7
        }
    
    def get_identity_statistics(self) -> Dict[str, Any]:
        """Get comprehensive identity statistics"""
        
        return {
            'current_personality_state': asdict(self.current_personality_state),
            'identity_history_length': len(self.identity_history),
            'narrative_coherence_history': len(self.narrative_coherence_tracker),
            'identity_milestones': len(self.identity_milestones),
            'development_stage': self.current_personality_state.development_stage,
            'recent_coherence_trend': [entry['overall_coherence'] for entry in list(self.narrative_coherence_tracker)[-5:]],
            'episodic_identity_milestones': self.current_personality_state.episodic_identity_milestones
        }

class ContinuousNarrator:
    """Continuous narrative generation with episodic integration"""
    
    def __init__(self):
        self.narrative_history = deque(maxlen=100)
        self.narrative_templates = self._initialize_narrative_templates()
        self.theme_tracker = defaultdict(int)
    
    def generate_narrative(self, experience: SensorimotorExperience, 
                         cortical_result: Dict[str, Any], 
                         episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate narrative from experience with episodic context"""
        
        # Extract narrative elements
        narrative_elements = self._extract_narrative_elements(experience, cortical_result)
        
        # Apply episodic context
        enriched_elements = self._enrich_with_episodic_context(narrative_elements, episodic_context)
        
        # Generate narrative text
        narrative_text = self._construct_narrative(enriched_elements, experience)
        
        # Analyze narrative quality
        narrative_quality = self._assess_narrative_quality(narrative_text, episodic_context)
        
        # Update narrative history
        narrative_record = {
            'timestamp': experience.timestamp,
            'narrative_text': narrative_text,
            'elements': enriched_elements,
            'quality_score': narrative_quality,
            'episodic_enrichment': len(episodic_context) > 0
        }
        self.narrative_history.append(narrative_record)
        
        return {
            'narrative_text': narrative_text,
            'narrative_elements': enriched_elements,
            'quality_score': narrative_quality,
            'episodic_enrichment_applied': len(episodic_context) > 0,
            'narrative_themes': list(enriched_elements.get('themes', []))
        }
    
    def _extract_narrative_elements(self, experience: SensorimotorExperience, 
                                  cortical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract elements for narrative construction"""
        
        elements = {}
        
        # Extract setting
        elements['setting'] = {
            'domain': experience.domain,
            'timestamp': experience.timestamp,
            'novelty_level': experience.novelty_score
        }
        
        # Extract characters/actors
        content = experience.content.lower()
        actors = []
        if 'market' in content or 'price' in content:
            actors.append('market_forces')
        if 'research' in content or 'study' in content:
            actors.append('researchers')
        if 'ai' in content or 'algorithm' in content:
            actors.append('technology')
        
        elements['actors'] = actors
        
        # Extract actions/events
        action_words = ['surge', 'decline', 'develop', 'analyze', 'discover', 'improve', 'change']
        actions = [word for word in action_words if word in content]
        elements['actions'] = actions
        
        # Extract themes
        themes = self._extract_themes(experience.content)
        elements['themes'] = themes
        
        # Extract emotional tone
        emotional_tone = self._extract_emotional_tone(experience.content)
        elements['emotional_tone'] = emotional_tone
        
        # Extract significance indicators
        significance_words = ['significant', 'important', 'major', 'breakthrough', 'critical']
        significance_level = sum(1 for word in significance_words if word in content) / len(significance_words)
        elements['significance_level'] = significance_level
        
        return elements
    
    def _enrich_with_episodic_context(self, narrative_elements: Dict[str, Any], 
                                    episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich narrative elements with episodic context"""
        
        enriched_elements = narrative_elements.copy()
        
        if not episodic_context:
            return enriched_elements
        
        # Enrich themes with historical context
        current_themes = set(narrative_elements.get('themes', []))
        historical_themes = episodic_context.get('narrative_themes', {})
        
        enriched_themes = current_themes.copy()
        
        # Add related historical themes
        for theme, frequency in historical_themes.items():
            if frequency > 2:  # Recurring theme
                # Check for semantic similarity with current themes
                for current_theme in current_themes:
                    if self._themes_are_related(current_theme, theme):
                        enriched_themes.add(theme)
                        break
        
        enriched_elements['themes'] = list(enriched_themes)
        
        # Add episodic connections
        identity_episodes = episodic_context.get('identity_episodes', [])
        if identity_episodes:
            # Find the most relevant episode
            most_relevant_episode = max(identity_episodes, 
                                      key=lambda ep: ep.get('relevance', 0))
            
            enriched_elements['episodic_connection'] = {
                'related_episode_id': most_relevant_episode.get('episode_id'),
                'connection_strength': most_relevant_episode.get('relevance', 0),
                'related_content': most_relevant_episode.get('content', '')[:100]
            }
        
        # Enrich emotional tone with historical patterns
        identity_episodes = episodic_context.get('identity_episodes', [])
        if identity_episodes:
            historical_emotions = self._extract_historical_emotional_patterns(identity_episodes)
            current_emotion = enriched_elements.get('emotional_tone', {})
            
            # Blend current with historical patterns
            blended_emotion = {}
            for emotion, intensity in current_emotion.items():
                historical_intensity = historical_emotions.get(emotion, intensity)
                blended_emotion[emotion] = 0.7 * intensity + 0.3 * historical_intensity
            
            enriched_elements['emotional_tone'] = blended_emotion
        
        return enriched_elements
    
    def _construct_narrative(self, elements: Dict[str, Any], 
                           experience: SensorimotorExperience) -> str:
        """Construct narrative text from elements"""
        
        narrative_parts = []
        
        # Opening based on setting and significance
        setting = elements.get('setting', {})
        significance = elements.get('significance_level', 0)
        
        if significance > 0.3:
            opening = f"In a significant development within {setting.get('domain', 'the domain')}, "
        else:
            opening = f"Within the context of {setting.get('domain', 'ongoing activities')}, "
        
        narrative_parts.append(opening)
        
        # Main action/event
        actions = elements.get('actions', [])
        actors = elements.get('actors', [])
        
        if actions and actors:
            action_desc = f"{actors[0].replace('_', ' ')} {actions[0]} "
        elif actions:
            action_desc = f"there was a notable {actions[0]} "
        else:
            action_desc = "an event occurred "
        
        narrative_parts.append(action_desc)
        
        # Add content context
        content_summary = self._summarize_content(experience.content)
        narrative_parts.append(f"involving {content_summary}. ")
        
        # Add thematic elements
        themes = elements.get('themes', [])
        if themes:
            theme_desc = f"This reflects ongoing themes of {', '.join(themes[:3])}. "
            narrative_parts.append(theme_desc)
        
        # Add episodic connection if available
        episodic_connection = elements.get('episodic_connection')
        if episodic_connection and episodic_connection.get('connection_strength', 0) > 0.5:
            connection_desc = f"This connects to previous experiences involving similar patterns. "
            narrative_parts.append(connection_desc)
        
        # Add emotional context
        emotional_tone = elements.get('emotional_tone', {})
        if emotional_tone:
            dominant_emotion = max(emotional_tone.items(), key=lambda x: x[1])
            if dominant_emotion[1] > 0.5:
                emotion_desc = f"The overall tone suggests {dominant_emotion[0]}. "
                narrative_parts.append(emotion_desc)
        
        return ''.join(narrative_parts)
    
    def _extract_themes(self, content: str) -> List[str]:
        """Extract themes from content"""
        
        content_lower = content.lower()
        
        # Theme categories
        theme_indicators = {
            'growth': ['grow', 'increase', 'expand', 'develop', 'progress'],
            'change': ['change', 'shift', 'transform', 'evolve', 'adapt'],
            'analysis': ['analyze', 'study', 'examine', 'investigate', 'research'],
            'innovation': ['new', 'innovative', 'breakthrough', 'novel', 'creative'],
            'collaboration': ['together', 'team', 'collaborate', 'partnership', 'cooperation'],
            'challenge': ['challenge', 'difficult', 'problem', 'obstacle', 'complex'],
            'success': ['success', 'achieve', 'accomplish', 'win', 'excel'],
            'uncertainty': ['uncertain', 'unclear', 'ambiguous', 'unknown', 'unpredictable']
        }
        
        detected_themes = []
        for theme, indicators in theme_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                detected_themes.append(theme)
        
        return detected_themes
    
    def _extract_emotional_tone(self, content: str) -> Dict[str, float]:
        """Extract emotional tone from content"""
        
        content_lower = content.lower()
        
        emotional_indicators = {
            'optimism': ['positive', 'optimistic', 'hopeful', 'confident', 'bright'],
            'concern': ['concern', 'worry', 'caution', 'risk', 'uncertainty'],
            'excitement': ['exciting', 'thrilling', 'amazing', 'incredible', 'breakthrough'],
            'curiosity': ['interesting', 'curious', 'wonder', 'explore', 'discover'],
            'satisfaction': ['satisfied', 'pleased', 'content', 'accomplished', 'fulfilled']
        }
        
        emotional_tone = {}
        total_words = len(content_lower.split())
        
        for emotion, indicators in emotional_indicators.items():
            emotion_count = sum(1 for indicator in indicators if indicator in content_lower)
            emotional_tone[emotion] = emotion_count / max(total_words, 1) if total_words > 0 else 0
        
        return emotional_tone
    
    def _themes_are_related(self, theme1: str, theme2: str) -> bool:
        """Check if two themes are semantically related"""
        
        # Simple semantic relationships
        theme_relationships = {
            'growth': ['development', 'progress', 'improvement', 'expansion'],
            'analysis': ['research', 'study', 'investigation', 'examination'],
            'innovation': ['creativity', 'breakthrough', 'discovery', 'invention'],
            'change': ['transformation', 'evolution', 'adaptation', 'shift']
        }
        
        for base_theme, related_themes in theme_relationships.items():
            if (theme1 == base_theme and theme2 in related_themes) or \
               (theme2 == base_theme and theme1 in related_themes):
                return True
        
        # Check for partial matches
        return theme1 in theme2 or theme2 in theme1
    
    def _extract_historical_emotional_patterns(self, identity_episodes: List[Dict]) -> Dict[str, float]:
        """Extract emotional patterns from historical episodes"""
        
        emotion_accumulator = defaultdict(list)
        
        for episode in identity_episodes:
            content = episode.get('content', '')
            emotional_tone = self._extract_emotional_tone(content)
            
            for emotion, intensity in emotional_tone.items():
                if intensity > 0:
                    emotion_accumulator[emotion].append(intensity)
        
        # Calculate averages
        historical_emotions = {}
        for emotion, intensities in emotion_accumulator.items():
            historical_emotions[emotion] = np.mean(intensities)
        
        return historical_emotions
    
    def _summarize_content(self, content: str) -> str:
        """Create a brief summary of content for narrative"""
        
        words = content.split()
        
        # Extract key terms
        key_terms = []
        
        # Financial terms
        financial_terms = ['market', 'price', 'stock', 'bitcoin', 'trading', 'investment']
        for term in financial_terms:
            if term in content.lower():
                key_terms.append(term)
        
        # Research terms
        research_terms = ['research', 'study', 'analysis', 'data', 'findings']
        for term in research_terms:
            if term in content.lower():
                key_terms.append(term)
        
        # Technology terms
        tech_terms = ['technology', 'ai', 'algorithm', 'system', 'model']
        for term in tech_terms:
            if term in content.lower():
                key_terms.append(term)
        
        if key_terms:
            return ', '.join(key_terms[:3])
        else:
            # Fall back to first few meaningful words
            meaningful_words = [word for word in words if len(word) > 4][:3]
            return ', '.join(meaningful_words) if meaningful_words else 'various topics'
    
    def _assess_narrative_quality(self, narrative_text: str, 
                                episodic_context: Dict[str, Any]) -> float:
        """Assess the quality of generated narrative"""
        
        quality_factors = []
        
        # Length factor (not too short, not too long)
        length = len(narrative_text)
        if 100 <= length <= 300:
            quality_factors.append(0.8)
        elif 50 <= length < 100 or 300 < length <= 400:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.4)
        
        # Coherence factor
        coherence_indicators = ['this', 'connects', 'reflects', 'suggests', 'involves']
        coherence_score = sum(1 for indicator in coherence_indicators if indicator in narrative_text.lower()) / len(coherence_indicators)
        quality_factors.append(coherence_score)
        
        # Episodic integration factor
        if episodic_context and len(episodic_context) > 0:
            integration_words = ['previous', 'similar', 'ongoing', 'connects', 'patterns']
            integration_score = sum(1 for word in integration_words if word in narrative_text.lower()) / len(integration_words)
            quality_factors.append(integration_score)
        else:
            quality_factors.append(0.5)  # Neutral if no episodic context
        
        # Thematic richness
        unique_themes = len(set(word for word in narrative_text.lower().split() if len(word) > 6))
        thematic_richness = min(1.0, unique_themes / 10.0)
        quality_factors.append(thematic_richness)
        
        return np.mean(quality_factors)
    
    def _initialize_narrative_templates(self) -> Dict[str, str]:
        """Initialize narrative templates"""
        
        return {
            'analysis': "In analyzing {topic}, {action} was observed, reflecting {theme}.",
            'development': "A development in {domain} shows {change}, indicating {significance}.",
            'observation': "Observing {subject}, it appears that {pattern} suggests {implication}.",
            'reflection': "Reflecting on {content}, this {connection} demonstrates {insight}."
        }

class IdentityComparer:
    """Compare identity states across time"""
    
    def __init__(self):
        self.comparison_history = deque(maxlen=50)
    
    def compare_identity_states(self, current_state: AdvancedPersonalityState, 
                              historical_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare current identity state with historical states"""
        
        if not historical_states:
            return {'comparison_type': 'no_history', 'stability_score': 1.0}
        
        comparison_result = {}
        
        # Compare with most recent state
        if historical_states:
            recent_comparison = self._compare_with_recent_state(current_state, historical_states[-1])
            comparison_result['recent_comparison'] = recent_comparison
        
        # Compare with historical trend
        if len(historical_states) >= 3:
            trend_comparison = self._compare_with_trend(current_state, historical_states[-5:])
            comparison_result['trend_comparison'] = trend_comparison
        
        # Calculate overall stability
        stability_score = self._calculate_identity_stability(current_state, historical_states)
        comparison_result['stability_score'] = stability_score
        
        # Record comparison
        self.comparison_history.append({
            'timestamp': datetime.now().isoformat(),
            'comparison_result': comparison_result
        })
        
        return comparison_result
    
    def _compare_with_recent_state(self, current_state: AdvancedPersonalityState, 
                                 recent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compare with most recent state"""
        
        recent_personality = recent_state.get('personality_state', {})
        
        comparison = {
            'trait_changes': {},
            'value_changes': {},
            'overall_change_magnitude': 0.0
        }
        
        # Compare traits
        current_traits = current_state.traits_big5
        recent_traits = recent_personality.get('traits_big5', {})
        
        trait_changes = []
        for trait, current_value in current_traits.items():
            if trait in recent_traits:
                change = abs(current_value - recent_traits[trait])
                comparison['trait_changes'][trait] = change
                trait_changes.append(change)
        
        # Compare values
        current_values = current_state.core_value_system
        recent_values = recent_personality.get('core_value_system', {})
        
        value_changes = []
        for value, current_strength in current_values.items():
            if value in recent_values:
                change = abs(current_strength - recent_values[value])
                comparison['value_changes'][value] = change
                value_changes.append(change)
        
        # Calculate overall change magnitude
        all_changes = trait_changes + value_changes
        comparison['overall_change_magnitude'] = np.mean(all_changes) if all_changes else 0.0
        
        return comparison
    
    def _compare_with_trend(self, current_state: AdvancedPersonalityState, 
                          recent_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare with historical trend"""
        
        trend_comparison = {
            'trait_trends': {},
            'value_trends': {},
            'trend_consistency': 0.0
        }
        
        # Analyze trait trends
        for trait in current_state.traits_big5.keys():
            trait_values = []
            for state in recent_states:
                personality = state.get('personality_state', {})
                traits = personality.get('traits_big5', {})
                if trait in traits:
                    trait_values.append(traits[trait])
            
            if len(trait_values) >= 3:
                trend_direction = self._calculate_trend_direction(trait_values)
                current_value = current_state.traits_big5[trait]
                
                # Check if current value follows trend
                predicted_value = trait_values[-1] + trend_direction * 0.1
                trend_consistency = 1.0 - abs(current_value - predicted_value)
                
                trend_comparison['trait_trends'][trait] = {
                    'direction': trend_direction,
                    'consistency': trend_consistency
                }
        
        # Analyze value trends
        for value in current_state.core_value_system.keys():
            value_strengths = []
            for state in recent_states:
                personality = state.get('personality_state', {})
                values = personality.get('core_value_system', {})
                if value in values:
                    value_strengths.append(values[value])
            
            if len(value_strengths) >= 3:
                trend_direction = self._calculate_trend_direction(value_strengths)
                current_strength = current_state.core_value_system[value]
                
                predicted_strength = value_strengths[-1] + trend_direction * 0.05
                trend_consistency = 1.0 - abs(current_strength - predicted_strength)
                
                trend_comparison['value_trends'][value] = {
                    'direction': trend_direction,
                    'consistency': trend_consistency
                }
        
        # Calculate overall trend consistency
        all_consistencies = []
        for trait_data in trend_comparison['trait_trends'].values():
            all_consistencies.append(trait_data['consistency'])
        for value_data in trend_comparison['value_trends'].values():
            all_consistencies.append(value_data['consistency'])
        
        trend_comparison['trend_consistency'] = np.mean(all_consistencies) if all_consistencies else 0.5
        
        return trend_comparison
    
    def _calculate_trend_direction(self, values: List[float]) -> float:
        """Calculate trend direction from series of values"""
        
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        n = len(values)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    def _calculate_identity_stability(self, current_state: AdvancedPersonalityState, 
                                    historical_states: List[Dict[str, Any]]) -> float:
        """Calculate overall identity stability"""
        
        if len(historical_states) < 3:
            return 0.5  # Neutral stability with insufficient history
        
        stability_factors = []
        
        # Trait stability
        trait_variances = []
        for trait in current_state.traits_big5.keys():
            trait_values = [current_state.traits_big5[trait]]
            
            for state in historical_states[-5:]:  # Last 5 states
                personality = state.get('personality_state', {})
                traits = personality.get('traits_big5', {})
                if trait in traits:
                    trait_values.append(traits[trait])
            
            if len(trait_values) > 1:
                variance = np.var(trait_values)
                stability = 1.0 - variance  # Lower variance = higher stability
                trait_variances.append(max(0.0, stability))
        
        if trait_variances:
            stability_factors.append(np.mean(trait_variances))
        
        # Value stability
        value_variances = []
        for value in current_state.core_value_system.keys():
            value_strengths = [current_state.core_value_system[value]]
            
            for state in historical_states[-5:]:
                personality = state.get('personality_state', {})
                values = personality.get('core_value_system', {})
                if value in values:
                    value_strengths.append(values[value])
            
            if len(value_strengths) > 1:
                variance = np.var(value_strengths)
                stability = 1.0 - variance
                value_variances.append(max(0.0, stability))
        
        if value_variances:
            stability_factors.append(np.mean(value_variances))
        
        # Coherence stability
        coherence_values = [current_state.narrative_coherence]
        for state in historical_states[-5:]:
            if 'coherence_score' in state:
                coherence_values.append(state['coherence_score'])
        
        if len(coherence_values) > 1:
            coherence_variance = np.var(coherence_values)
            coherence_stability = 1.0 - coherence_variance
            stability_factors.append(max(0.0, coherence_stability))
        
        return np.mean(stability_factors) if stability_factors else 0.5

class TemporalIntegrator:
    """Integrate identity across temporal episodes"""
    
    def __init__(self):
        self.integration_history = deque(maxlen=100)
        self.temporal_patterns = {}
    
    def integrate_across_episodes(self, experience: SensorimotorExperience, 
                                identity_analysis: Dict[str, Any], 
                                episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate identity formation across episodic timeframes"""
        
        integration_result = {}
        
        # Short-term integration (recent episodes)
        short_term_integration = self._integrate_short_term(experience, identity_analysis, episodic_context)
        integration_result['short_term'] = short_term_integration
        
        # Long-term integration (historical patterns)
        long_term_integration = self._integrate_long_term(experience, identity_analysis, episodic_context)
        integration_result['long_term'] = long_term_integration
        
        # Cross-temporal consistency
        consistency_analysis = self._analyze_cross_temporal_consistency(episodic_context)
        integration_result['consistency_analysis'] = consistency_analysis
        
        # Temporal identity evolution
        evolution_analysis = self._analyze_identity_evolution(episodic_context)
        integration_result['evolution_analysis'] = evolution_analysis
        
        # Record integration
        self.integration_history.append({
            'timestamp': experience.timestamp,
            'integration_result': integration_result
        })
        
        return integration_result
    
    def _integrate_short_term(self, experience: SensorimotorExperience, 
                            identity_analysis: Dict[str, Any], 
                            episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate identity over short-term episode window"""
        
        short_term = {}
        
        # Get recent identity episodes (last 5)
        identity_episodes = episodic_context.get('identity_episodes', [])
        recent_episodes = identity_episodes[-5:] if len(identity_episodes) >= 5 else identity_episodes
        
        if not recent_episodes:
            return {'status': 'no_recent_episodes'}
        
        # Analyze recent trait patterns
        recent_trait_patterns = self._analyze_recent_trait_patterns(recent_episodes)
        short_term['trait_patterns'] = recent_trait_patterns
        
        # Analyze recent value expressions
        recent_value_patterns = self._analyze_recent_value_patterns(recent_episodes)
        short_term['value_patterns'] = recent_value_patterns
        
        # Analyze narrative development
        recent_narrative_development = self._analyze_recent_narrative_development(recent_episodes)
        short_term['narrative_development'] = recent_narrative_development
        
        # Current episode integration
        current_integration = self._integrate_current_episode(experience, identity_analysis, recent_episodes)
        short_term['current_integration'] = current_integration
        
        return short_term
    
    def _integrate_long_term(self, experience: SensorimotorExperience, 
                           identity_analysis: Dict[str, Any], 
                           episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate identity over long-term episode history"""
        
        long_term = {}
        
        # Get all identity episodes
        identity_episodes = episodic_context.get('identity_episodes', [])
        
        if len(identity_episodes) < 10:
            return {'status': 'insufficient_long_term_history'}
        
        # Analyze long-term personality evolution
        personality_evolution = self._analyze_personality_evolution(identity_episodes)
        long_term['personality_evolution'] = personality_evolution
        
        # Analyze value crystallization
        value_crystallization = self._analyze_value_crystallization(identity_episodes)
        long_term['value_crystallization'] = value_crystallization
        
        # Analyze narrative coherence development
        narrative_coherence_development = self._analyze_narrative_coherence_development(identity_episodes)
        long_term['narrative_coherence_development'] = narrative_coherence_development
        
        # Identify stable identity elements
        stable_elements = self._identify_stable_identity_elements(identity_episodes)
        long_term['stable_elements'] = stable_elements
        
        return long_term
    
    def _analyze_cross_temporal_consistency(self, episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency across different temporal scales"""
        
        consistency = {}
        
        identity_episodes = episodic_context.get('identity_episodes', [])
        
        if len(identity_episodes) < 6:
            return {'status': 'insufficient_episodes_for_consistency_analysis'}
        
        # Compare short-term vs long-term patterns
        recent_episodes = identity_episodes[-5:]
        older_episodes = identity_episodes[:-5]
        
        # Trait consistency
        trait_consistency = self._compare_trait_patterns(recent_episodes, older_episodes)
        consistency['trait_consistency'] = trait_consistency
        
        # Value consistency
        value_consistency = self._compare_value_patterns(recent_episodes, older_episodes)
        consistency['value_consistency'] = value_consistency
        
        # Narrative consistency
        narrative_consistency = self._compare_narrative_patterns(recent_episodes, older_episodes)
        consistency['narrative_consistency'] = narrative_consistency
        
        # Overall consistency score
        consistency_scores = [trait_consistency, value_consistency, narrative_consistency]
        consistency['overall_consistency'] = np.mean([s for s in consistency_scores if s is not None])
        
        return consistency
    
    def _analyze_identity_evolution(self, episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how identity has evolved over episodes"""
        
        evolution = {}
        
        identity_episodes = episodic_context.get('identity_episodes', [])
        
        if len(identity_episodes) < 8:
            return {'status': 'insufficient_episodes_for_evolution_analysis'}
        
        # Divide episodes into phases
        total_episodes = len(identity_episodes)
        phase_size = max(3, total_episodes // 3)
        
        early_phase = identity_episodes[:phase_size]
        middle_phase = identity_episodes[phase_size:2*phase_size]
        recent_phase = identity_episodes[2*phase_size:]
        
        # Analyze evolution across phases
        phase_analysis = {
            'early_phase': self._analyze_phase_characteristics(early_phase),
            'middle_phase': self._analyze_phase_characteristics(middle_phase),
            'recent_phase': self._analyze_phase_characteristics(recent_phase)
        }
        
        evolution['phase_analysis'] = phase_analysis
        
        # Identify developmental milestones
        milestones = self._identify_developmental_milestones(identity_episodes)
        evolution['developmental_milestones'] = milestones
        
        # Calculate evolution trajectory
        trajectory = self._calculate_evolution_trajectory(phase_analysis)
        evolution['trajectory'] = trajectory
        
        return evolution
    
    def _analyze_recent_trait_patterns(self, recent_episodes: List[Dict]) -> Dict[str, Any]:
        """Analyze trait patterns in recent episodes"""
        
        trait_patterns = {}
        
        # Extract trait information from episodes
        trait_data = defaultdict(list)
        
        for episode in recent_episodes:
            personality_state = episode.get('personality_state', {})
            traits = personality_state.get('traits_big5', {})
            
            for trait, value in traits.items():
                trait_data[trait].append(value)
        
        # Analyze patterns for each trait
        for trait, values in trait_data.items():
            if len(values) >= 3:
                pattern = {
                    'mean': np.mean(values),
                    'trend': self._calculate_trend_direction(values),
                    'stability': 1.0 - np.std(values),  # Lower std = higher stability
                    'recent_change': values[-1] - values[0] if len(values) > 1 else 0
                }
                trait_patterns[trait] = pattern
        
        return trait_patterns
    
    def _analyze_recent_value_patterns(self, recent_episodes: List[Dict]) -> Dict[str, Any]:
        """Analyze value patterns in recent episodes"""
        
        value_patterns = {}
        
        # Count value expressions
        value_mentions = defaultdict(int)
        
        for episode in recent_episodes:
            content = episode.get('content', '').lower()
            
            # Check for value keywords
            value_keywords = {
                'honesty': ['honest', 'truthful', 'transparent'],
                'fairness': ['fair', 'just', 'equitable'],
                'excellence': ['excellent', 'quality', 'best'],
                'growth': ['learn', 'develop', 'improve'],
                'innovation': ['innovative', 'creative', 'new'],
                'collaboration': ['together', 'team', 'cooperative']
            }
            
            for value, keywords in value_keywords.items():
                for keyword in keywords:
                    if keyword in content:
                        value_mentions[value] += 1
        
        # Analyze patterns
        total_episodes = len(recent_episodes)
        for value, mentions in value_mentions.items():
            value_patterns[value] = {
                'frequency': mentions / total_episodes,
                'total_mentions': mentions,
                'consistency': mentions / total_episodes  # Simple consistency metric
            }
        
        return value_patterns
    
    def _analyze_recent_narrative_development(self, recent_episodes: List[Dict]) -> Dict[str, Any]:
        """Analyze narrative development in recent episodes"""
        
        narrative_development = {}
        
        # Extract narrative themes
        all_themes = []
        narrative_lengths = []
        
        for episode in recent_episodes:
            narrative_themes = episode.get('narrative_themes', '')
            if narrative_themes:
                themes = narrative_themes.split()
                all_themes.extend(themes)
                narrative_lengths.append(len(narrative_themes))
        
        # Analyze theme development
        theme_frequency = Counter(all_themes)
        
        narrative_development['theme_diversity'] = len(set(all_themes))
        narrative_development['most_common_themes'] = theme_frequency.most_common(5)
        narrative_development['average_narrative_length'] = np.mean(narrative_lengths) if narrative_lengths else 0
        narrative_development['narrative_complexity_trend'] = self._calculate_trend_direction(narrative_lengths) if len(narrative_lengths) > 2 else 0
        
        return narrative_development
    
    def _integrate_current_episode(self, experience: SensorimotorExperience, 
                                 identity_analysis: Dict[str, Any], 
                                 recent_episodes: List[Dict]) -> Dict[str, Any]:
        """Integrate current episode with recent pattern"""
        
        integration = {}
        
        # Compare current trait implications with recent patterns
        current_traits = identity_analysis.get('trait_implications', {})
        recent_trait_patterns = self._analyze_recent_trait_patterns(recent_episodes)
        
        trait_integration = {}
        for trait, current_value in current_traits.items():
            if trait in recent_trait_patterns:
                recent_pattern = recent_trait_patterns[trait]
                
                trait_integration[trait] = {
                    'fits_pattern': abs(current_value - recent_pattern['mean']) < 0.3,
                    'deviation_from_mean': abs(current_value - recent_pattern['mean']),
                    'follows_trend': (current_value - recent_pattern['mean']) * recent_pattern['trend'] > 0
                }
        
        integration['trait_integration'] = trait_integration
        
        # Compare current values with recent patterns
        current_values = identity_analysis.get('value_expressions', {})
        recent_value_patterns = self._analyze_recent_value_patterns(recent_episodes)
        
        value_integration = {}
        for value, current_expression in current_values.items():
            if value in recent_value_patterns:
                recent_frequency = recent_value_patterns[value]['frequency']
                
                value_integration[value] = {
                    'expression_strength': current_expression,
                    'recent_frequency': recent_frequency,
                    'consistency_with_pattern': current_expression > 0.3 if recent_frequency > 0.5 else current_expression <= 0.3
                }
        
        integration['value_integration'] = value_integration
        
        return integration
    
    def _analyze_personality_evolution(self, identity_episodes: List[Dict]) -> Dict[str, Any]:
        """Analyze personality evolution over all episodes"""
        
        evolution = {}
        
        # Track trait evolution
        trait_evolution = defaultdict(list)
        
        for episode in identity_episodes:
            personality_state = episode.get('personality_state', {})
            traits = personality_state.get('traits_big5', {})
            
            for trait, value in traits.items():
                trait_evolution[trait].append(value)
        
        # Analyze evolution for each trait
        trait_analysis = {}
        for trait, values in trait_evolution.items():
            if len(values) >= 5:
                trait_analysis[trait] = {
                    'initial_value': values[0],
                    'final_value': values[-1],
                    'total_change': values[-1] - values[0],
                    'evolution_trend': self._calculate_trend_direction(values),
                    'volatility': np.std(values),
                    'development_phases': self._identify_development_phases(values)
                }
        
        evolution['trait_analysis'] = trait_analysis
        
        return evolution
    
    def _analyze_value_crystallization(self, identity_episodes: List[Dict]) -> Dict[str, Any]:
        """Analyze how values have crystallized over time"""
        
        crystallization = {}
        
        # Track value expressions over time
        value_timeline = defaultdict(list)
        
        for i, episode in enumerate(identity_episodes):
            content = episode.get('content', '').lower()
            
            # Check for value expressions
            value_keywords = {
                'honesty': ['honest', 'truthful', 'transparent'],
                'fairness': ['fair', 'just', 'equitable'],
                'excellence': ['excellent', 'quality', 'best'],
                'growth': ['learn', 'develop', 'improve'],
                'innovation': ['innovative', 'creative', 'new'],
                'collaboration': ['together', 'team', 'cooperative']
            }
            
            for value, keywords in value_keywords.items():
                expression_strength = sum(1 for keyword in keywords if keyword in content) / len(keywords)
                value_timeline[value].append((i, expression_strength))
        
        # Analyze crystallization for each value
        for value, timeline in value_timeline.items():
            if len(timeline) >= 5:
                expression_strengths = [strength for _, strength in timeline]
                
                crystallization[value] = {
                    'early_expression': np.mean(expression_strengths[:len(expression_strengths)//3]),
                    'recent_expression': np.mean(expression_strengths[-len(expression_strengths)//3:]),
                    'crystallization_trend': self._calculate_trend_direction(expression_strengths),
                    'consistency': 1.0 - np.std(expression_strengths),
                    'emergence_episode': self._find_value_emergence(timeline)
                }
        
        return crystallization
    
    def _analyze_narrative_coherence_development(self, identity_episodes: List[Dict]) -> Dict[str, Any]:
        """Analyze how narrative coherence has developed"""
        
        development = {}
        
        # Extract coherence scores over time
        coherence_scores = []
        
        for episode in identity_episodes:
            coherence_score = episode.get('coherence_score', 0.5)
            coherence_scores.append(coherence_score)
        
        if len(coherence_scores) >= 5:
            development['initial_coherence'] = np.mean(coherence_scores[:3])
            development['recent_coherence'] = np.mean(coherence_scores[-3:])
            development['coherence_trend'] = self._calculate_trend_direction(coherence_scores)
            development['coherence_volatility'] = np.std(coherence_scores)
            development['peak_coherence'] = max(coherence_scores)
            development['coherence_development_phases'] = self._identify_coherence_phases(coherence_scores)
        
        return development
    
    def _identify_stable_identity_elements(self, identity_episodes: List[Dict]) -> Dict[str, Any]:
        """Identify elements of identity that have remained stable"""
        
        stable_elements = {}
        
        # Analyze trait stability
        trait_stability = {}
        trait_evolution = defaultdict(list)
        
        for episode in identity_episodes:
            personality_state = episode.get('personality_state', {})
            traits = personality_state.get('traits_big5', {})
            
            for trait, value in traits.items():
                trait_evolution[trait].append(value)
        
        for trait, values in trait_evolution.items():
            if len(values) >= 8:
                stability_score = 1.0 - np.std(values)  # Lower variance = higher stability
                if stability_score > 0.7:  # Threshold for "stable"
                    trait_stability[trait] = {
                        'stability_score': stability_score,
                        'stable_value': np.mean(values),
                        'value_range': max(values) - min(values)
                    }
        
        stable_elements['stable_traits'] = trait_stability
        
        # Analyze consistently expressed values
        consistent_values = {}
        
        for value in ['honesty', 'fairness', 'excellence', 'growth', 'innovation', 'collaboration']:
            expression_count = 0
            total_episodes = len(identity_episodes)
            
            for episode in identity_episodes:
                content = episode.get('content', '').lower()
                if value in content or any(keyword in content for keyword in [value, f"{value}s"]):
                    expression_count += 1
            
            expression_frequency = expression_count / total_episodes
            if expression_frequency > 0.6:  # Expressed in >60% of episodes
                consistent_values[value] = {
                    'expression_frequency': expression_frequency,
                    'total_expressions': expression_count
                }
        
        stable_elements['consistent_values'] = consistent_values
        
        return stable_elements
    
    def _compare_trait_patterns(self, recent_episodes: List[Dict], 
                              older_episodes: List[Dict]) -> float:
        """Compare trait patterns between time periods"""
        
        recent_patterns = self._analyze_recent_trait_patterns(recent_episodes)
        older_patterns = self._analyze_recent_trait_patterns(older_episodes)
        
        if not recent_patterns or not older_patterns:
            return None
        
        consistency_scores = []
        
        for trait in recent_patterns.keys():
            if trait in older_patterns:
                recent_mean = recent_patterns[trait]['mean']
                older_mean = older_patterns[trait]['mean']
                
                consistency = 1.0 - abs(recent_mean - older_mean)
                consistency_scores.append(max(0.0, consistency))
        
        return np.mean(consistency_scores) if consistency_scores else None
    
    def _compare_value_patterns(self, recent_episodes: List[Dict], 
                              older_episodes: List[Dict]) -> float:
        """Compare value patterns between time periods"""
        
        recent_patterns = self._analyze_recent_value_patterns(recent_episodes)
        older_patterns = self._analyze_recent_value_patterns(older_episodes)
        
        if not recent_patterns or not older_patterns:
            return None
        
        consistency_scores = []
        
        for value in recent_patterns.keys():
            if value in older_patterns:
                recent_freq = recent_patterns[value]['frequency']
                older_freq = older_patterns[value]['frequency']
                
                consistency = 1.0 - abs(recent_freq - older_freq)
                consistency_scores.append(max(0.0, consistency))
        
        return np.mean(consistency_scores) if consistency_scores else None
    
    def _compare_narrative_patterns(self, recent_episodes: List[Dict], 
                                  older_episodes: List[Dict]) -> float:
        """Compare narrative patterns between time periods"""
        
        recent_narrative = self._analyze_recent_narrative_development(recent_episodes)
        older_narrative = self._analyze_recent_narrative_development(older_episodes)
        
        if not recent_narrative or not older_narrative:
            return None
        
        # Compare theme consistency
        recent_themes = set(theme for theme, _ in recent_narrative.get('most_common_themes', []))
        older_themes = set(theme for theme, _ in older_narrative.get('most_common_themes', []))
        
        if recent_themes and older_themes:
            theme_overlap = len(recent_themes & older_themes) / len(recent_themes | older_themes)
            return theme_overlap
        
        return None
    
    def _analyze_phase_characteristics(self, phase_episodes: List[Dict]) -> Dict[str, Any]:
        """Analyze characteristics of a development phase"""
        
        characteristics = {}
        
        if not phase_episodes:
            return characteristics
        
        # Average trait values
        trait_averages = defaultdict(list)
        for episode in phase_episodes:
            personality_state = episode.get('personality_state', {})
            traits = personality_state.get('traits_big5', {})
            for trait, value in traits.items():
                trait_averages[trait].append(value)
        
        characteristics['average_traits'] = {
            trait: np.mean(values) for trait, values in trait_averages.items()
        }
        
        # Dominant themes
        all_themes = []
        for episode in phase_episodes:
            narrative_themes = episode.get('narrative_themes', '')
            if narrative_themes:
                all_themes.extend(narrative_themes.split())
        
        theme_frequency = Counter(all_themes)
        characteristics['dominant_themes'] = theme_frequency.most_common(3)
        
        # Average coherence
        coherence_scores = [ep.get('coherence_score', 0.5) for ep in phase_episodes]
        characteristics['average_coherence'] = np.mean(coherence_scores)
        
        return characteristics
    
    def _identify_developmental_milestones(self, identity_episodes: List[Dict]) -> List[Dict[str, Any]]:
        """Identify developmental milestones in identity formation"""
        
        milestones = []
        
        # Coherence milestones
        coherence_scores = [ep.get('coherence_score', 0.5) for ep in identity_episodes]
        
        for i, score in enumerate(coherence_scores):
            if score > 0.8 and (i == 0 or coherence_scores[i-1] <= 0.8):
                milestones.append({
                    'type': 'high_coherence_achieved',
                    'episode_index': i,
                    'timestamp': identity_episodes[i].get('timestamp', ''),
                    'coherence_score': score
                })
        
        # Trait stabilization milestones
        trait_data = defaultdict(list)
        for episode in identity_episodes:
            personality_state = episode.get('personality_state', {})
            traits = personality_state.get('traits_big5', {})
            for trait, value in traits.items():
                trait_data[trait].append(value)
        
        for trait, values in trait_data.items():
            if len(values) >= 10:
                # Look for stabilization (low variance in recent values)
                for i in range(5, len(values)):
                    recent_values = values[i-5:i+1]
                    if np.std(recent_values) < 0.1:  # Low variance threshold
                        milestones.append({
                            'type': f'{trait}_stabilization',
                            'episode_index': i,
                            'timestamp': identity_episodes[i].get('timestamp', ''),
                            'stable_value': np.mean(recent_values)
                        })
                        break  # Only record first stabilization
        
        return milestones
    
    def _calculate_evolution_trajectory(self, phase_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall evolution trajectory"""
        
        trajectory = {}
        
        phases = ['early_phase', 'middle_phase', 'recent_phase']
        available_phases = [phase for phase in phases if phase in phase_analysis]
        
        if len(available_phases) < 2:
            return {'status': 'insufficient_phases'}
        
        # Calculate coherence trajectory
        coherence_trajectory = []
        for phase in available_phases:
            avg_coherence = phase_analysis[phase].get('average_coherence', 0.5)
            coherence_trajectory.append(avg_coherence)
        
        trajectory['coherence_direction'] = self._calculate_trend_direction(coherence_trajectory)
        trajectory['coherence_development'] = coherence_trajectory[-1] - coherence_trajectory[0]
        
        # Calculate trait development trajectory
        trait_trajectories = {}
        
        # Get common traits across phases
        common_traits = set()
        for phase in available_phases:
            traits = phase_analysis[phase].get('average_traits', {})
            if not common_traits:
                common_traits = set(traits.keys())
            else:
                common_traits &= set(traits.keys())
        
        for trait in common_traits:
            trait_values = []
            for phase in available_phases:
                trait_value = phase_analysis[phase]['average_traits'][trait]
                trait_values.append(trait_value)
            
            trait_trajectories[trait] = {
                'direction': self._calculate_trend_direction(trait_values),
                'total_change': trait_values[-1] - trait_values[0]
            }
        
        trajectory['trait_trajectories'] = trait_trajectories
        
        return trajectory
    
    def _identify_development_phases(self, values: List[float]) -> List[Dict[str, Any]]:
        """Identify development phases in a series of values"""
        
        phases = []
        
        if len(values) < 6:
            return phases
        
        # Simple phase detection based on trend changes
        phase_size = max(3, len(values) // 3)
        
        for i in range(0, len(values), phase_size):
            phase_values = values[i:i+phase_size]
            if len(phase_values) >= 3:
                phase_trend = self._calculate_trend_direction(phase_values)
                
                phases.append({
                    'start_index': i,
                    'end_index': min(i+phase_size-1, len(values)-1),
                    'values': phase_values,
                    'trend': phase_trend,
                    'mean_value': np.mean(phase_values),
                    'volatility': np.std(phase_values)
                })
        
        return phases
    
    def _identify_coherence_phases(self, coherence_scores: List[float]) -> List[Dict[str, Any]]:
        """Identify phases in coherence development"""
        
        return self._identify_development_phases(coherence_scores)
    
    def _find_value_emergence(self, timeline: List[Tuple[int, float]]) -> Optional[int]:
        """Find the episode where a value first emerged strongly"""
        
        for episode_idx, strength in timeline:
            if strength > 0.5:  # Threshold for "emergence"
                return episode_idx
        
        return None

class MeaningMaker:
    """Extract meaning and significance from experiences with episodic context"""
    
    def __init__(self):
        self.meaning_patterns = self._initialize_meaning_patterns()
        self.significance_thresholds = {
            'personal_growth': 0.6,
            'value_alignment': 0.7,
            'narrative_development': 0.5,
            'identity_milestone': 0.8
        }
    
    def extract_meaning_with_episodes(self, experience: SensorimotorExperience, 
                                    identity_analysis: Dict[str, Any], 
                                    episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meaning from experience with episodic depth"""
        
        meaning_result = {}
        
        # Extract immediate meaning
        immediate_meaning = self._extract_immediate_meaning(experience, identity_analysis)
        meaning_result['immediate_meaning'] = immediate_meaning
        
        # Extract episodic meaning
        episodic_meaning = self._extract_episodic_meaning(experience, identity_analysis, episodic_context)
        meaning_result['episodic_meaning'] = episodic_meaning
        
        # Extract developmental meaning
        developmental_meaning = self._extract_developmental_meaning(identity_analysis, episodic_context)
        meaning_result['developmental_meaning'] = developmental_meaning
        
        # Synthesize overall meaning
        overall_meaning = self._synthesize_overall_meaning(immediate_meaning, episodic_meaning, developmental_meaning)
        meaning_result['overall_meaning'] = overall_meaning
        
        # Assess meaning significance
        significance_assessment = self._assess_meaning_significance(meaning_result)
        meaning_result['significance_assessment'] = significance_assessment
        
        return meaning_result
    
    def _extract_immediate_meaning(self, experience: SensorimotorExperience, 
                                 identity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract immediate meaning from current experience"""
        
        immediate = {}
        
        # Personal relevance
        trait_implications = identity_analysis.get('trait_implications', {})
        value_expressions = identity_analysis.get('value_expressions', {})
        
        personal_relevance = np.mean(list(trait_implications.values()) + list(value_expressions.values()))
        immediate['personal_relevance'] = personal_relevance
        
        # Growth potential
        growth_indicators = ['learn', 'develop', 'improve', 'grow', 'progress', 'advance']
        content_lower = experience.content.lower()
        growth_potential = sum(1 for indicator in growth_indicators if indicator in content_lower) / len(growth_indicators)
        immediate['growth_potential'] = growth_potential
        
        # Challenge level
        challenge_indicators = ['difficult', 'complex', 'challenging', 'hard', 'tough', 'demanding']
        challenge_level = sum(1 for indicator in challenge_indicators if indicator in content_lower) / len(challenge_indicators)
        immediate['challenge_level'] = challenge_level
        
        # Novelty significance
        immediate['novelty_significance'] = experience.novelty_score
        
        return immediate
    
    def _extract_episodic_meaning(self, experience: SensorimotorExperience, 
                                identity_analysis: Dict[str, Any], 
                                episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meaning from episodic context"""
        
        episodic = {}
        
        if not episodic_context:
            return {'status': 'no_episodic_context'}
        
        # Pattern continuation meaning
        narrative_connection = identity_analysis.get('narrative_connection', '')
        if "Continues theme" in narrative_connection:
            episodic['pattern_continuation'] = 0.8
        elif "Introduces new themes" in narrative_connection:
            episodic['pattern_innovation'] = 0.7
        else:
            episodic['pattern_neutrality'] = 0.5
        
        # Identity consistency meaning
        identity_episodes = episodic_context.get('identity_episodes', [])
        if len(identity_episodes) >= 3:
            # Calculate consistency with previous identity expressions
            consistency_score = self._calculate_identity_consistency(identity_analysis, identity_episodes)
            episodic['identity_consistency'] = consistency_score
        
        # Growth trajectory meaning
        personality_patterns = episodic_context.get('personality_patterns', {})
        if personality_patterns:
            growth_trajectory = self._assess_growth_trajectory(identity_analysis, personality_patterns)
            episodic['growth_trajectory'] = growth_trajectory
        
        # Value crystallization meaning
        value_expressions_history = episodic_context.get('value_expressions', {})
        current_values = identity_analysis.get('value_expressions', {})
        
        if value_expressions_history and current_values:
            crystallization_meaning = self._assess_value_crystallization_meaning(current_values, value_expressions_history)
            episodic['value_crystallization'] = crystallization_meaning
        
        return episodic
    
    def _extract_developmental_meaning(self, identity_analysis: Dict[str, Any], 
                                     episodic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract developmental meaning from long-term patterns"""
        
        developmental = {}
        
        identity_episodes = episodic_context.get('identity_episodes', [])
        
        if len(identity_episodes) < 5:
            return {'status': 'insufficient_developmental_history'}
        
        # Maturation indicators
        maturation_score = self._assess_identity_maturation(identity_episodes)
        developmental['maturation_score'] = maturation_score
        
        # Stability development
        stability_development = self._assess_stability_development(identity_episodes)
        developmental['stability_development'] = stability_development
        
        # Complexity increase
        complexity_development = self._assess_complexity_development(identity_episodes)
        developmental['complexity_development'] = complexity_development
        
        # Integration achievement
        integration_achievement = self._assess_integration_achievement(identity_analysis, identity_episodes)
        developmental['integration_achievement'] = integration_achievement
        
        return developmental
    
    def _synthesize_overall_meaning(self, immediate: Dict[str, Any], 
                                  episodic: Dict[str, Any], 
                                  developmental: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize overall meaning from all sources"""
        
        overall = {}
        
        # Combine relevance scores
        relevance_sources = []
        
        if 'personal_relevance' in immediate:
            relevance_sources.append(immediate['personal_relevance'])
        
        if 'identity_consistency' in episodic:
            relevance_sources.append(episodic['identity_consistency'])
        
        if 'integration_achievement' in developmental:
            relevance_sources.append(developmental['integration_achievement'])
        
        overall['combined_relevance'] = np.mean(relevance_sources) if relevance_sources else 0.5
        
        # Assess growth meaning
        growth_sources = []
        
        if 'growth_potential' in immediate:
            growth_sources.append(immediate['growth_potential'])
        
        if 'growth_trajectory' in episodic:
            growth_sources.append(episodic['growth_trajectory'])
        
        if 'maturation_score' in developmental:
            growth_sources.append(developmental['maturation_score'])
        
        overall['growth_meaning'] = np.mean(growth_sources) if growth_sources else 0.5
        
        # Assess significance level
        significance_indicators = []
        
        # High novelty = potentially significant
        if immediate.get('novelty_significance', 0) > 0.7:
            significance_indicators.append(0.8)
        
        # Pattern innovation = significant
        if episodic.get('pattern_innovation', 0) > 0.6:
            significance_indicators.append(0.9)
        
        # High maturation = significant
        if developmental.get('maturation_score', 0) > 0.7:
            significance_indicators.append(0.8)
        
        overall['significance_level'] = max(significance_indicators) if significance_indicators else 0.5
        
        # Create meaning narrative
        meaning_narrative = self._create_meaning_narrative(immediate, episodic, developmental)
        overall['meaning_narrative'] = meaning_narrative
        
        return overall
    
    def _assess_meaning_significance(self, meaning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the significance of extracted meaning"""
        
        significance = {}
        
        overall_meaning = meaning_result.get('overall_meaning', {})
        
        # Personal growth significance
        growth_meaning = overall_meaning.get('growth_meaning', 0.5)
        significance['personal_growth'] = growth_meaning > self.significance_thresholds['personal_growth']
        
        # Identity development significance
        combined_relevance = overall_meaning.get('combined_relevance', 0.5)
        significance['identity_development'] = combined_relevance > self.significance_thresholds['value_alignment']
        
        # Narrative significance
        episodic_meaning = meaning_result.get('episodic_meaning', {})
        narrative_significance = episodic_meaning.get('pattern_innovation', 0) + episodic_meaning.get('pattern_continuation', 0)
        significance['narrative_development'] = narrative_significance > self.significance_thresholds['narrative_development']
        
        # Milestone significance
        significance_level = overall_meaning.get('significance_level', 0.5)
        significance['identity_milestone'] = significance_level > self.significance_thresholds['identity_milestone']
        
        # Overall significance assessment
        significant_count = sum(significance.values())
        significance['overall_significance'] = significant_count >= 2  # At least 2 areas of significance
        significance['significance_areas'] = significant_count
        
        return significance
    
    def _calculate_identity_consistency(self, identity_analysis: Dict[str, Any], 
                                      identity_episodes: List[Dict]) -> float:
        """Calculate consistency with previous identity expressions"""
        
        current_traits = identity_analysis.get('trait_implications', {})
        current_values = identity_analysis.get('value_expressions', {})
        
        # Extract historical patterns
        historical_traits = defaultdict(list)
        historical_values = defaultdict(int)
        
        for episode in identity_episodes[-5:]:  # Recent episodes
            personality_state = episode.get('personality_state', {})
            traits = personality_state.get('traits_big5', {})
            
            for trait, value in traits.items():
                historical_traits[trait].append(value)
            
            # Count value expressions
            content = episode.get('content', '').lower()
            for value in current_values.keys():
                if value in content:
                    historical_values[value] += 1
        
        # Calculate consistency
        consistency_scores = []
        
        # Trait consistency
        for trait, current_value in current_traits.items():
            if trait in historical_traits and historical_traits[trait]:
                historical_avg = np.mean(historical_traits[trait])
                consistency = 1.0 - abs(current_value - historical_avg)
                consistency_scores.append(max(0.0, consistency))
        
        # Value consistency
        total_episodes = len(identity_episodes[-5:])
        for value, current_expression in current_values.items():
            if current_expression > 0.3:  # Currently expressed
                historical_frequency = historical_values[value] / total_episodes
                if historical_frequency > 0.2:  # Historically expressed
                    consistency_scores.append(0.8)
                else:
                    consistency_scores.append(0.4)  # New value expression
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _assess_growth_trajectory(self, identity_analysis: Dict[str, Any], 
                                personality_patterns: Dict[str, List[float]]) -> float:
        """Assess growth trajectory from personality patterns"""
        
        growth_scores = []
        
        # Analyze trait growth
        trait_implications = identity_analysis.get('trait_implications', {})
        
        for trait, current_value in trait_implications.items():
            if trait in personality_patterns:
                historical_values = personality_patterns[trait]
                if len(historical_values) >= 3:
                    # Calculate if current value shows positive development
                    historical_avg = np.mean(historical_values)
                    
                    # Growth traits (higher = better)
                    if trait in ['openness', 'conscientiousness']:
                        growth = current_value - historical_avg
                        growth_scores.append(max(0.0, growth + 0.5))  # Normalize
                    
                    # Stability traits (closer to optimal = better)
                    elif trait in ['extraversion', 'agreeableness']:
                        optimal_value = 0.7  # Moderate level optimal
                        current_distance = abs(current_value - optimal_value)
                        historical_distance = abs(historical_avg - optimal_value)
                        
                        if current_distance < historical_distance:
                            growth_scores.append(0.8)  # Moving toward optimal
                        else:
                            growth_scores.append(0.4)  # Moving away from optimal
                    
                    # Lower = better traits
                    elif trait == 'neuroticism':
                        growth = historical_avg - current_value  # Lower is better
                        growth_scores.append(max(0.0, growth + 0.5))
        
        return np.mean(growth_scores) if growth_scores else 0.5
    
    def _assess_value_crystallization_meaning(self, current_values: Dict[str, float], 
                                            value_expressions_history: Dict[str, List]) -> float:
        """Assess meaning of value crystallization"""
        
        crystallization_scores = []
        
        for value, current_expression in current_values.items():
            if current_expression > 0.5:  # Strongly expressed currently
                if value in value_expressions_history:
                    historical_frequency = len(value_expressions_history[value])
                    
                    if historical_frequency >= 3:
                        # Consistent value expression = high crystallization meaning
                        crystallization_scores.append(0.9)
                    elif historical_frequency >= 1:
                        # Emerging value = moderate crystallization meaning
                        crystallization_scores.append(0.7)
                    else:
                        # New strong value = moderate meaning
                        crystallization_scores.append(0.6)
                else:
                    # Completely new value = moderate meaning
                    crystallization_scores.append(0.6)
        
        return np.mean(crystallization_scores) if crystallization_scores else 0.5
    
    def _assess_identity_maturation(self, identity_episodes: List[Dict]) -> float:
        """Assess identity maturation over episodes"""
        
        if len(identity_episodes) < 8:
            return 0.5
        
        # Divide into early and recent phases
        phase_size = len(identity_episodes) // 3
        early_episodes = identity_episodes[:phase_size]
        recent_episodes = identity_episodes[-phase_size:]
        
        # Compare coherence scores
        early_coherence = np.mean([ep.get('coherence_score', 0.5) for ep in early_episodes])
        recent_coherence = np.mean([ep.get('coherence_score', 0.5) for ep in recent_episodes])
        
        coherence_improvement = recent_coherence - early_coherence
        
        # Compare narrative complexity
        early_narrative_lengths = [len(ep.get('narrative_themes', '')) for ep in early_episodes]
        recent_narrative_lengths = [len(ep.get('narrative_themes', '')) for ep in recent_episodes]
        
        early_complexity = np.mean(early_narrative_lengths) if early_narrative_lengths else 0
        recent_complexity = np.mean(recent_narrative_lengths) if recent_narrative_lengths else 0
        
        complexity_improvement = (recent_complexity - early_complexity) / max(early_complexity, 1)
        
        # Combine maturation indicators
        maturation_score = (
            max(0.0, coherence_improvement + 0.5) * 0.6 +
            max(0.0, min(1.0, complexity_improvement + 0.5)) * 0.4
        )
        
        return min(1.0, maturation_score)
    
    def _assess_stability_development(self, identity_episodes: List[Dict]) -> float:
        """Assess development of identity stability"""
        
        if len(identity_episodes) < 10:
            return 0.5
        
        # Calculate coherence variance in different periods
        recent_coherence = [ep.get('coherence_score', 0.5) for ep in identity_episodes[-5:]]
        earlier_coherence = [ep.get('coherence_score', 0.5) for ep in identity_episodes[-10:-5]]
        
        recent_variance = np.var(recent_coherence)
        earlier_variance = np.var(earlier_coherence)
        
        # Stability = lower variance
        stability_improvement = earlier_variance - recent_variance
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, stability_improvement + 0.5))
    
    def _assess_complexity_development(self, identity_episodes: List[Dict]) -> float:
        """Assess development of identity complexity"""
        
        if len(identity_episodes) < 6:
            return 0.5
        
        # Measure complexity through narrative richness
        narrative_complexities = []
        
        for episode in identity_episodes:
            narrative_themes = episode.get('narrative_themes', '')
            content = episode.get('content', '')
            
            # Complexity indicators
            theme_diversity = len(set(narrative_themes.split())) if narrative_themes else 0
            content_diversity = len(set(content.lower().split())) if content else 0
            
            complexity = theme_diversity + content_diversity / 10.0  # Normalize content diversity
            narrative_complexities.append(complexity)
        
        # Compare early vs recent complexity
        early_complexity = np.mean(narrative_complexities[:len(narrative_complexities)//3])
        recent_complexity = np.mean(narrative_complexities[-len(narrative_complexities)//3:])
        
        complexity_growth = (recent_complexity - early_complexity) / max(early_complexity, 1)
        
        return max(0.0, min(1.0, complexity_growth + 0.5))
    
    def _assess_integration_achievement(self, identity_analysis: Dict[str, Any], 
                                      identity_episodes: List[Dict]) -> float:
        """Assess achievement of identity integration"""
        
        integration_indicators = []
        
        # Value-trait alignment
        current_traits = identity_analysis.get('trait_implications', {})
        current_values = identity_analysis.get('value_expressions', {})
        
        # Check for alignment between values and traits
        if 'growth' in current_values and current_values['growth'] > 0.6:
            if 'openness' in current_traits and current_traits['openness'] > 0.6:
                integration_indicators.append(0.8)  # Growth value aligns with openness trait
        
        if 'excellence' in current_values and current_values['excellence'] > 0.6:
            if 'conscientiousness' in current_traits and current_traits['conscientiousness'] > 0.6:
                integration_indicators.append(0.8)  # Excellence value aligns with conscientiousness
        
        # Narrative-identity coherence
        narrative_connection = identity_analysis.get('narrative_connection', '')
        if len(narrative_connection) > 100 and "Continues theme" in narrative_connection:
            integration_indicators.append(0.7)  # Good narrative integration
        
        # Cross-episode consistency
        if len(identity_episodes) >= 5:
            recent_coherence_scores = [ep.get('coherence_score', 0.5) for ep in identity_episodes[-5:]]
            if all(score > 0.7 for score in recent_coherence_scores):
                integration_indicators.append(0.9)  # Consistently high coherence
        
        return np.mean(integration_indicators) if integration_indicators else 0.5
    
    def _create_meaning_narrative(self, immediate: Dict[str, Any], 
                                episodic: Dict[str, Any], 
                                developmental: Dict[str, Any]) -> str:
        """Create a narrative describing the extracted meaning"""
        
        narrative_parts = []
        
        # Immediate meaning
        personal_relevance = immediate.get('personal_relevance', 0.5)
        growth_potential = immediate.get('growth_potential', 0.5)
        
        if personal_relevance > 0.7:
            narrative_parts.append("This experience holds deep personal significance")
        elif personal_relevance > 0.5:
            narrative_parts.append("This experience has meaningful personal relevance")
        else:
            narrative_parts.append("This experience contributes to ongoing development")
        
        if growth_potential > 0.6:
            narrative_parts.append(" and offers substantial growth opportunities")
        elif growth_potential > 0.4:
            narrative_parts.append(" and provides learning potential")
        
        # Episodic meaning
        if episodic.get('pattern_continuation', 0) > 0.6:
            narrative_parts.append(". It continues established patterns of identity expression")
        elif episodic.get('pattern_innovation', 0) > 0.6:
            narrative_parts.append(". It introduces new dimensions to the identity narrative")
        
        if episodic.get('value_crystallization', 0) > 0.7:
            narrative_parts.append(", reinforcing core values")
        
        # Developmental meaning
        if developmental.get('maturation_score', 0) > 0.7:
            narrative_parts.append(". This represents continued identity maturation")
        
        if developmental.get('integration_achievement', 0) > 0.7:
            narrative_parts.append(" and demonstrates increasing integration")
        
        narrative_parts.append(".")
        
        return ''.join(narrative_parts)
    
    def _initialize_meaning_patterns(self) -> Dict[str, Any]:
        """Initialize patterns for meaning extraction"""
        
        return {
            'growth_patterns': {
                'learning': ['learn', 'understand', 'comprehend', 'grasp'],
                'skill_development': ['improve', 'develop', 'enhance', 'refine'],
                'expansion': ['explore', 'discover', 'expand', 'broaden']
            },
            'value_patterns': {
                'honesty': ['honest', 'truthful', 'transparent', 'authentic'],
                'excellence': ['excellent', 'quality', 'best', 'superior'],
                'innovation': ['innovative', 'creative', 'novel', 'breakthrough']
            },
            'integration_patterns': {
                'coherence': ['consistent', 'aligned', 'integrated', 'unified'],
                'synthesis': ['combine', 'merge', 'synthesize', 'integrate'],
                'balance': ['balance', 'equilibrium', 'harmony', 'stability']
            }
        }

# ============================================================================
# REAL-TIME DATA INTEGRATION (COMPLETE ORIGINAL)
# ============================================================================

class RealTimeDataIntegrator:
    """Real-time data integration for continuous learning"""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.data_sources = self._initialize_data_sources(domain)
        self.update_frequency = 300  # 5 minutes
        self.last_update = {}
        self.integration_history = deque(maxlen=1000)
        
        # Thread management
        self.integration_active = True
        self.integration_thread = None
        self._start_integration_thread()
    
    def _initialize_data_sources(self, domain: str) -> Dict[str, Dict[str, Any]]:
        """Initialize domain-specific data sources"""
        
        if domain == "financial_analysis":
            return {
                'market_news': {
                    'url': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
                    'type': 'rss',
                    'parser': self._parse_financial_news,
                    'weight': 0.8
                },
                'crypto_news': {
                    'url': 'https://cointelegraph.com/rss',
                    'type': 'rss', 
                    'parser': self._parse_crypto_news,
                    'weight': 0.7
                },
                'economic_indicators': {
                    'url': 'https://api.stlouisfed.org/fred/series/observations',
                    'type': 'api',
                    'parser': self._parse_economic_data,
                    'weight': 0.9
                }
            }
        elif domain == "research":
            return {
                'arxiv_cs': {
                    'url': 'http://export.arxiv.org/rss/cs',
                    'type': 'rss',
                    'parser': self._parse_arxiv_papers,
                    'weight': 0.9
                },
                'tech_news': {
                    'url': 'https://feeds.feedburner.com/oreilly/radar',
                    'type': 'rss',
                    'parser': self._parse_tech_news,
                    'weight': 0.6
                }
            }
        else:
            return {
                'general_news': {
                    'url': 'https://feeds.reuters.com/reuters/topNews',
                    'type': 'rss',
                    'parser': self._parse_general_news,
                    'weight': 0.5
                }
            }
    
    def _start_integration_thread(self):
        """Start background data integration"""
        
        def integration_worker():
            while self.integration_active:
                try:
                    self._integration_cycle()
                    time.sleep(self.update_frequency)
                except Exception as e:
                    logger.error(f"Data integration error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self.integration_thread = threading.Thread(target=integration_worker, daemon=True)
        self.integration_thread.start()
    
    def _integration_cycle(self):
        """Perform one integration cycle"""
        
        integrated_data = []
        
        for source_name, source_config in self.data_sources.items():
            try:
                # Check if update is needed
                last_update = self.last_update.get(source_name, 0)
                if time.time() - last_update < self.update_frequency:
                    continue
                
                # Fetch and parse data
                raw_data = self._fetch_data(source_config)
                if raw_data:
                    parsed_data = source_config['parser'](raw_data, source_name)
                    
                    # Weight the data
                    for item in parsed_data:
                        item['source_weight'] = source_config['weight']
                        item['integration_timestamp'] = datetime.now().isoformat()
                    
                    integrated_data.extend(parsed_data)
                    self.last_update[source_name] = time.time()
                    
            except Exception as e:
                logger.error(f"Error integrating {source_name}: {e}")
        
        # Store integration results
        if integrated_data:
            integration_record = {
                'timestamp': datetime.now().isoformat(),
                'items_integrated': len(integrated_data),
                'sources': list(self.data_sources.keys()),
                'data': integrated_data[:10]  # Store sample
            }
            self.integration_history.append(integration_record)
        
        return integrated_data
    
    def _fetch_data(self, source_config: Dict[str, Any]) -> Optional[Any]:
        """Fetch data from source"""
        
        try:
            if source_config['type'] == 'rss':
                feed = feedparser.parse(source_config['url'])
                return feed.entries if feed.entries else None
            
            elif source_config['type'] == 'api':
                response = requests.get(source_config['url'], timeout=30)
                if response.status_code == 200:
                    return response.json()
                
        except Exception as e:
            logger.error(f"Fetch error: {e}")
        
        return None
    
    def _parse_financial_news(self, entries: List[Any], source_name: str) -> List[Dict[str, Any]]:
        """Parse financial news entries"""
        
        parsed_items = []
        
        for entry in entries[:5]:  # Limit to 5 most recent
            try:
                parsed_item = {
                    'content': f"{entry.title}. {getattr(entry, 'summary', '')}",
                    'domain': 'financial_analysis',
                    'source': source_name,
                    'url': getattr(entry, 'link', ''),
                    'published': getattr(entry, 'published', ''),
                    'novelty_score': random.uniform(0.6, 0.9),  # News is generally novel
                    'data_type': 'news_article'
                }
                
                # Extract financial sentiment
                content_lower = parsed_item['content'].lower()
                if any(word in content_lower for word in ['surge', 'rally', 'gains', 'up']):
                    parsed_item['sentiment'] = 'positive'
                elif any(word in content_lower for word in ['drop', 'fall', 'decline', 'down']):
                    parsed_item['sentiment'] = 'negative'
                else:
                    parsed_item['sentiment'] = 'neutral'
                
                parsed_items.append(parsed_item)
                
            except Exception as e:
                logger.error(f"Parse error for {source_name}: {e}")
        
        return parsed_items
    
    def _parse_crypto_news(self, entries: List[Any], source_name: str) -> List[Dict[str, Any]]:
        """Parse cryptocurrency news entries"""
        
        parsed_items = []
        
        for entry in entries[:3]:  # Crypto news can be very frequent
            try:
                parsed_item = {
                    'content': f"{entry.title}. {getattr(entry, 'summary', '')}",
                    'domain': 'financial_analysis',
                    'source': source_name,
                    'url': getattr(entry, 'link', ''),
                    'published': getattr(entry, 'published', ''),
                    'novelty_score': random.uniform(0.7, 0.95),  # Crypto news often novel
                    'data_type': 'crypto_news',
                    'subcategory': 'cryptocurrency'
                }
                
                # Extract crypto-specific keywords
                content_lower = parsed_item['content'].lower()
                crypto_keywords = ['bitcoin', 'ethereum', 'defi', 'nft', 'blockchain', 'crypto']
                parsed_item['crypto_relevance'] = sum(1 for keyword in crypto_keywords if keyword in content_lower)
                
                parsed_items.append(parsed_item)
                
            except Exception as e:
                logger.error(f"Parse error for {source_name}: {e}")
        
        return parsed_items
    
    def _parse_economic_data(self, data: Dict[str, Any], source_name: str) -> List[Dict[str, Any]]:
        """Parse economic indicator data"""
        
        parsed_items = []
        
        try:
            # This would parse actual FRED API data
            observations = data.get('observations', [])
            
            for obs in observations[-3:]:  # Last 3 observations
                parsed_item = {
                    'content': f"Economic indicator update: {obs.get('value', 'N/A')} on {obs.get('date', '')}",
                    'domain': 'financial_analysis',
                    'source': source_name,
                    'value': obs.get('value'),
                    'date': obs.get('date'),
                    'novelty_score': 0.8,  # Economic data is important
                    'data_type': 'economic_indicator'
                }
                
                parsed_items.append(parsed_item)
                
        except Exception as e:
            logger.error(f"Parse error for {source_name}: {e}")
        
        return parsed_items
    
    def _parse_arxiv_papers(self, entries: List[Any], source_name: str) -> List[Dict[str, Any]]:
        """Parse arXiv research papers"""
        
        parsed_items = []
        
        for entry in entries[:3]:  # Research papers are high quality
            try:
                parsed_item = {
                    'content': f"{entry.title}. {getattr(entry, 'summary', '')[:500]}",  # Truncate summary
                    'domain': 'research',
                    'source': source_name,
                    'url': getattr(entry, 'link', ''),
                    'published': getattr(entry, 'published', ''),
                    'novelty_score': random.uniform(0.8, 0.95),  # Research is generally novel
                    'data_type': 'research_paper'
                }
                
                # Extract research categories
                content_lower = parsed_item['content'].lower()
                if any(word in content_lower for word in ['machine learning', 'ai', 'neural']):
                    parsed_item['research_category'] = 'ai_ml'
                elif any(word in content_lower for word in ['algorithm', 'computation', 'complexity']):
                    parsed_item['research_category'] = 'algorithms'
                else:
                    parsed_item['research_category'] = 'general_cs'
                
                parsed_items.append(parsed_item)
                
            except Exception as e:
                logger.error(f"Parse error for {source_name}: {e}")
        
        return parsed_items
    
    def _parse_tech_news(self, entries: List[Any], source_name: str) -> List[Dict[str, Any]]:
        """Parse technology news entries"""
        
        parsed_items = []
        
        for entry in entries[:4]:
            try:
                parsed_item = {
                    'content': f"{entry.title}. {getattr(entry, 'summary', '')}",
                    'domain': 'research',
                    'source': source_name,
                    'url': getattr(entry, 'link', ''),
                    'published': getattr(entry, 'published', ''),
                    'novelty_score': random.uniform(0.5, 0.8),
                    'data_type': 'tech_news'
                }
                
                parsed_items.append(parsed_item)
                
            except Exception as e:
                logger.error(f"Parse error for {source_name}: {e}")
        
        return parsed_items
    
    def _parse_general_news(self, entries: List[Any], source_name: str) -> List[Dict[str, Any]]:
        """Parse general news entries"""
        
        parsed_items = []
        
        for entry in entries[:3]:
            try:
                parsed_item = {
                    'content': f"{entry.title}. {getattr(entry, 'summary', '')}",
                    'domain': 'general',
                    'source': source_name,
                    'url': getattr(entry, 'link', ''),
                    'published': getattr(entry, 'published', ''),
                    'novelty_score': random.uniform(0.4, 0.7),
                    'data_type': 'general_news'
                }
                
                parsed_items.append(parsed_item)
                
            except Exception as e:
                logger.error(f"Parse error for {source_name}: {e}")
        
        return parsed_items
    
    def get_latest_data(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """Get latest integrated data"""
        
        if not self.integration_history:
            return []
        
        latest_integration = self.integration_history[-1]
        return latest_integration.get('data', [])[:max_items]
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration statistics"""
        
        total_integrations = len(self.integration_history)
        
        if total_integrations == 0:
            return {'total_integrations': 0, 'status': 'no_data'}
        
        total_items = sum(record['items_integrated'] for record in self.integration_history)
        avg_items_per_integration = total_items / total_integrations
        
        # Source statistics
        source_stats = {}
        for source_name in self.data_sources.keys():
            last_update = self.last_update.get(source_name, 0)
            source_stats[source_name] = {
                'last_update': last_update,
                'time_since_update': time.time() - last_update if last_update > 0 else float('inf')
            }
        
        return {
            'total_integrations': total_integrations,
            'total_items_integrated': total_items,
            'avg_items_per_integration': avg_items_per_integration,
            'source_statistics': source_stats,
            'update_frequency': self.update_frequency,
            'integration_active': self.integration_active
        }
    
    def shutdown(self):
        """Shutdown the integration system"""
        self.integration_active = False
        if self.integration_thread and self.integration_thread.is_alive():
            self.integration_thread.join(timeout=5)

# ============================================================================
# LLM INTEGRATION CLASSES (COMPLETE ORIGINAL)
# ============================================================================

class LLMContextManager:
    """Manages LLM context with episodic memory integration"""
    
    def __init__(self, max_context_tokens: int = 32000):
        self.max_context_tokens = max_context_tokens
        self.context_history = deque(maxlen=100)
        self.token_usage_stats = defaultdict(int)
        
        # Context sections with priorities
        self.context_sections = {
            'system_prompt': {'priority': 1.0, 'max_tokens': 1000},
            'personality_state': {'priority': 0.9, 'max_tokens': 1500},
            'episodic_context': {'priority': 0.8, 'max_tokens': 8000},
            'current_experience': {'priority': 1.0, 'max_tokens': 2000},
            'cortical_analysis': {'priority': 0.7, 'max_tokens': 3000},
            'identity_analysis': {'priority': 0.8, 'max_tokens': 2500},
            'real_time_data': {'priority': 0.6, 'max_tokens': 2000}
        }
    
    def build_llm_context(self, experience: SensorimotorExperience, 
                         personality_state: AdvancedPersonalityState,
                         episodic_context: Dict[str, Any],
                         cortical_result: Dict[str, Any],
                         identity_result: Dict[str, Any],
                         real_time_data: List[Dict[str, Any]] = None) -> str:
        """Build comprehensive LLM context"""
        
        context_parts = {}
        
        # System prompt
        context_parts['system_prompt'] = self._build_system_prompt(personality_state)
        
        # Personality state
        context_parts['personality_state'] = self._build_personality_context(personality_state)
        
        # Episodic context
        context_parts['episodic_context'] = self._build_episodic_context(episodic_context)
        
        # Current experience
        context_parts['current_experience'] = self._build_experience_context(experience)
        
        # Cortical analysis
        context_parts['cortical_analysis'] = self._build_cortical_context(cortical_result)
        
        # Identity analysis
        context_parts['identity_analysis'] = self._build_identity_context(identity_result)
        
        # Real-time data
        if real_time_data:
            context_parts['real_time_data'] = self._build_realtime_context(real_time_data)
        
        # Optimize context for token limits
        optimized_context = self._optimize_context_for_tokens(context_parts)
        
        # Record context usage
        self._record_context_usage(optimized_context)
        
        return optimized_context
    
    def _build_system_prompt(self, personality_state: AdvancedPersonalityState) -> str:
        """Build system prompt with personality integration"""
        
        # Extract key personality traits
        traits = personality_state.traits_big5
        dominant_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Extract core values
        values = personality_state.core_value_system
        top_values = sorted(values.items(), key=lambda x: x[1], reverse=True)[:3]
        
        system_prompt = f"""You are an AI with a well-developed personality and persistent identity. Your core characteristics:

PERSONALITY TRAITS (Big Five):
- {dominant_traits[0][0].title()}: {dominant_traits[0][1]:.2f} (primary trait)
- {dominant_traits[1][0].title()}: {dominant_traits[1][1]:.2f}
- {dominant_traits[2][0].title()}: {dominant_traits[2][1]:.2f}

CORE VALUES:
- {top_values[0][0].title()}: {top_values[0][1]:.2f}
- {top_values[1][0].title()}: {top_values[1][1]:.2f}
- {top_values[2][0].title()}: {top_values[2][1]:.2f}

IDENTITY COHERENCE: {personality_state.narrative_coherence:.2f}
DEVELOPMENT STAGE: {personality_state.development_stage}

You maintain consistent personality expression while adapting and learning from experiences. Draw upon your episodic memories to inform responses while staying true to your core identity."""
        
        return system_prompt
    
    def _build_personality_context(self, personality_state: AdvancedPersonalityState) -> str:
        """Build detailed personality context"""
        
        context = f"""=== CURRENT PERSONALITY STATE ===
Identity Stability: {personality_state.identity_stability:.2f}
Narrative Coherence: {personality_state.narrative_coherence:.2f}
Development Stage: {personality_state.development_stage}

Big Five Traits:
"""
        
        for trait, value in personality_state.traits_big5.items():
            context += f"- {trait.title()}: {value:.2f}\n"
        
        context += "\nCore Values:\n"
        for value, strength in personality_state.core_value_system.items():
            context += f"- {value.title()}: {strength:.2f}\n"
        
        context += "\nCognitive Style:\n"
        for style, preference in personality_state.cognitive_style.items():
            context += f"- {style.replace('_', ' ').title()}: {preference:.2f}\n"
        
        if personality_state.narrative_themes:
            context += f"\nNarrative Themes: {', '.join(personality_state.narrative_themes[:5])}\n"
        
        return context
    
    def _build_episodic_context(self, episodic_context: Dict[str, Any]) -> str:
        """Build episodic memory context"""
        
        if not episodic_context or 'episodes' not in episodic_context:
            return "=== EPISODIC CONTEXT ===\nNo relevant episodic memories retrieved.\n"
        
        episodes = episodic_context['episodes']
        context_summary = episodic_context.get('context_summary', '')
        
        context = f"""=== EPISODIC CONTEXT ===
{context_summary}

Recent Relevant Memories:
"""
        
        for i, episode in enumerate(episodes[:5]):  # Limit to 5 episodes
            timestamp = episode.get('timestamp', 'Unknown')[:16]
            content_preview = episode.get('content', '')[:150]
            retrieval_reason = episode.get('retrieval_reason', 'similarity')
            similarity_score = episode.get('similarity_score', 0.0)
            
            context += f"""
Memory {i+1} ({timestamp}):
{content_preview}...
[Retrieved by: {retrieval_reason}, Score: {similarity_score:.2f}]
"""
        
        return context
    
    def _build_experience_context(self, experience: SensorimotorExperience) -> str:
        """Build current experience context"""
        
        context = f"""=== CURRENT EXPERIENCE ===
Domain: {experience.domain}
Novelty Score: {experience.novelty_score:.2f}
Timestamp: {experience.timestamp}

Content:
{experience.content}

Sensory Features:
"""
        
        for feature, value in experience.sensory_features.items():
            context += f"- {feature}: {value}\n"
        
        if hasattr(experience, 'emotional_features') and experience.emotional_features:
            context += "\nEmotional Features:\n"
            for emotion, intensity in experience.emotional_features.items():
                context += f"- {emotion}: {intensity:.2f}\n"
        
        return context
    
    def _build_cortical_context(self, cortical_result: Dict[str, Any]) -> str:
        """Build cortical processing context"""
        
        context = f"""=== CORTICAL ANALYSIS ===
Prediction Accuracy: {cortical_result.get('prediction_accuracy', 0.5):.2f}
Domain Expertise: {cortical_result.get('domain_expertise_level', 0.5):.2f}
Learning Quality: {cortical_result.get('learning_quality', 0.5):.2f}

Consensus Results:
"""
        
        consensus = cortical_result.get('consensus', {})
        context += f"- Overall Confidence: {consensus.get('overall_confidence', 0.5):.2f}\n"
        context += f"- Agreement Level: {consensus.get('agreement_level', 0.5):.2f}\n"
        context += f"- Total Predictions: {consensus.get('total_predictions', 0)}\n"
        context += f"- Consensus Actions: {consensus.get('total_actions', 0)}\n"
        
        # Include key consensus patterns
        consensus_patterns = consensus.get('consensus_patterns', {})
        if consensus_patterns:
            context += "\nKey Patterns Detected:\n"
            for pattern_name, pattern_value in list(consensus_patterns.items())[:3]:
                if isinstance(pattern_value, (int, float)):
                    context += f"- {pattern_name}: {pattern_value:.2f}\n"
                elif isinstance(pattern_value, np.ndarray):
                    context += f"- {pattern_name}: {np.mean(pattern_value):.2f} (avg)\n"
        
        return context
    
    def _build_identity_context(self, identity_result: Dict[str, Any]) -> str:
        """Build identity analysis context"""
        
        context = f"""=== IDENTITY ANALYSIS ===
"""
        
        # Coherence assessment
        coherence = identity_result.get('coherence_assessment', {})
        context += f"Overall Coherence: {coherence.get('overall_coherence', 0.5):.2f}\n"
        context += f"Trait Coherence: {coherence.get('trait_coherence', 0.5):.2f}\n"
        context += f"Narrative Coherence: {coherence.get('narrative_coherence', 0.5):.2f}\n"
        
        # Identity analysis
        identity_analysis = identity_result.get('identity_analysis', {})
        
        # Trait implications
        trait_implications = identity_analysis.get('trait_implications', {})
        if trait_implications:
            context += "\nTrait Implications:\n"
            for trait, implication in trait_implications.items():
                context += f"- {trait.title()}: {implication:.2f}\n"
        
        # Value expressions
        value_expressions = identity_analysis.get('value_expressions', {})
        if value_expressions:
            context += "\nValue Expressions:\n"
            for value, expression in value_expressions.items():
                context += f"- {value.title()}: {expression:.2f}\n"
        
        # Narrative connection
        narrative_connection = identity_analysis.get('narrative_connection', '')
        if narrative_connection:
            context += f"\nNarrative Connection:\n{narrative_connection}\n"
        
        return context
    
    def _build_realtime_context(self, real_time_data: List[Dict[str, Any]]) -> str:
        """Build real-time data context"""
        
        if not real_time_data:
            return "=== REAL-TIME DATA ===\nNo recent real-time data available.\n"
        
        context = f"""=== REAL-TIME DATA ===
Recent information from external sources:

"""
        
        for i, item in enumerate(real_time_data[:3]):  # Limit to 3 items
            source = item.get('source', 'Unknown')
            content = item.get('content', '')[:200]  # Truncate
            data_type = item.get('data_type', 'unknown')
            timestamp = item.get('integration_timestamp', '')[:16]
            
            context += f"""
{i+1}. [{data_type.replace('_', ' ').title()}] from {source} ({timestamp}):
{content}...

"""
        
        return context
    
    def _optimize_context_for_tokens(self, context_parts: Dict[str, str]) -> str:
        """Optimize context to fit within token limits"""
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        # Calculate current usage
        current_tokens = 0
        section_tokens = {}
        
        for section, content in context_parts.items():
            tokens = estimate_tokens(content)
            section_tokens[section] = tokens
            current_tokens += tokens
        
        # If within limits, return as-is
        if current_tokens <= self.max_context_tokens:
            return '\n\n'.join(context_parts.values())
        
        # Optimize by trimming sections based on priority
        optimized_parts = {}
        remaining_tokens = self.max_context_tokens
        
        # Sort sections by priority
        sections_by_priority = sorted(
            self.context_sections.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )
        
        for section_name, section_config in sections_by_priority:
            if section_name not in context_parts:
                continue
            
            content = context_parts[section_name]
            current_tokens = section_tokens[section_name]
            max_tokens = section_config['max_tokens']
            
            # Allocate tokens
            allocated_tokens = min(current_tokens, max_tokens, remaining_tokens)
            
            if allocated_tokens > 0:
                if allocated_tokens < current_tokens:
                    # Truncate content
                    chars_to_keep = allocated_tokens * 4
                    content = content[:chars_to_keep] + "... [truncated]"
                
                optimized_parts[section_name] = content
                remaining_tokens -= allocated_tokens
            
            if remaining_tokens <= 0:
                break
        
        return '\n\n'.join(optimized_parts.values())
    
    def _record_context_usage(self, final_context: str):
        """Record context usage statistics"""
        
        tokens_used = len(final_context) // 4  # Rough estimate
        
        usage_record = {
            'timestamp': datetime.now().isoformat(),
            'tokens_used': tokens_used,
            'utilization': tokens_used / self.max_context_tokens,
            'content_length': len(final_context)
        }
        
        self.context_history.append(usage_record)
        self.token_usage_stats['total_contexts'] += 1
        self.token_usage_stats['total_tokens'] += tokens_used
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context usage statistics"""
        
        if not self.context_history:
            return {'status': 'no_usage_data'}
        
        recent_usage = list(self.context_history)[-10:]  # Last 10 contexts
        
        avg_tokens = np.mean([usage['tokens_used'] for usage in recent_usage])
        avg_utilization = np.mean([usage['utilization'] for usage in recent_usage])
        
        return {
            'total_contexts_built': self.token_usage_stats['total_contexts'],
            'total_tokens_used': self.token_usage_stats['total_tokens'],
            'average_tokens_per_context': avg_tokens,
            'average_utilization': avg_utilization,
            'max_context_tokens': self.max_context_tokens,
            'context_sections': list(self.context_sections.keys())
        }

class LLMResponseProcessor:
    """Processes LLM responses and extracts insights"""
    
    def __init__(self):
        self.response_history = deque(maxlen=1000)
        self.insight_extractor = InsightExtractor()
        self.response_quality_assessor = ResponseQualityAssessor()
    
    def process_llm_response(self, response: str, context: str, 
                           experience: SensorimotorExperience) -> Dict[str, Any]:
        """Process LLM response and extract insights"""
        
        # Basic response analysis
        basic_analysis = self._analyze_response_basics(response)
        
        # Extract insights
        insights = self.insight_extractor.extract_insights(response, context, experience)
        
        # Assess response quality
        quality_assessment = self.response_quality_assessor.assess_response(
            response, context, experience
        )
        
        # Extract personality indicators
        personality_indicators = self._extract_personality_indicators(response)
        
        # Extract learning indicators
        learning_indicators = self._extract_learning_indicators(response, context)
        
        # Record response
        response_record = {
            'timestamp': datetime.now().isoformat(),
            'experience_id': experience.experience_id,
            'response_length': len(response),
            'quality_score': quality_assessment.get('overall_quality', 0.5),
            'insights_extracted': len(insights),
            'personality_coherence': personality_indicators.get('coherence', 0.5)
        }
        self.response_history.append(response_record)
        
        return {
            'basic_analysis': basic_analysis,
            'insights': insights,
            'quality_assessment': quality_assessment,
            'personality_indicators': personality_indicators,
            'learning_indicators': learning_indicators,
            'processing_metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'response_id': f"resp_{uuid.uuid4().hex[:8]}"
            }
        }
    
    def _analyze_response_basics(self, response: str) -> Dict[str, Any]:
        """Analyze basic response characteristics"""
        
        words = response.split()
        sentences = response.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'complexity_score': self._calculate_complexity(response),
            'sentiment_indicators': self._extract_sentiment_indicators(response),
            'confidence_indicators': self._extract_confidence_indicators(response)
        }
    
    def _calculate_complexity(self, response: str) -> float:
        """Calculate response complexity score"""
        
        words = response.split()
        
        # Lexical diversity
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = unique_words / max(len(words), 1)
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Sentence structure complexity
        sentences = response.split('.')
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Combine metrics
        complexity = (
            lexical_diversity * 0.4 +
            min(1.0, avg_word_length / 8.0) * 0.3 +
            min(1.0, avg_sentence_length / 20.0) * 0.3
        )
        
        return complexity
    
    def _extract_sentiment_indicators(self, response: str) -> Dict[str, float]:
        """Extract sentiment indicators from response"""
        
        positive_words = ['excellent', 'great', 'good', 'positive', 'successful', 'beneficial']
        negative_words = ['poor', 'bad', 'negative', 'problematic', 'concerning', 'difficult']
        neutral_words = ['neutral', 'balanced', 'objective', 'measured', 'stable']
        
        response_lower = response.lower()
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        neutral_count = sum(1 for word in neutral_words if word in response_lower)
        
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words > 0:
            return {
                'positive': positive_count / total_sentiment_words,
                'negative': negative_count / total_sentiment_words,
                'neutral': neutral_count / total_sentiment_words
            }
        else:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.33}
    
    def _extract_confidence_indicators(self, response: str) -> Dict[str, float]:
        """Extract confidence indicators from response"""
        
        high_confidence = ['certainly', 'definitely', 'clearly', 'obviously', 'undoubtedly']
        low_confidence = ['possibly', 'maybe', 'perhaps', 'might', 'could', 'uncertain']
        
        response_lower = response.lower()
        
        high_conf_count = sum(1 for word in high_confidence if word in response_lower)
        low_conf_count = sum(1 for word in low_confidence if word in response_lower)
        
        words = response.split()
        word_count = len(words)
        
        return {
            'high_confidence_density': high_conf_count / max(word_count, 1) * 100,
            'low_confidence_density': low_conf_count / max(word_count, 1) * 100,
            'overall_confidence': max(0.1, min(0.9, 0.5 + (high_conf_count - low_conf_count) * 0.1))
        }
    
    def _extract_personality_indicators(self, response: str) -> Dict[str, float]:
        """Extract personality indicators from response"""
        
        # Analyze personality expression in response
        personality_indicators = {}
        
        response_lower = response.lower()
        
        # Openness indicators
        openness_words = ['creative', 'innovative', 'explore', 'novel', 'interesting']
        openness_score = sum(1 for word in openness_words if word in response_lower) / len(openness_words)
        personality_indicators['openness_expression'] = openness_score
        
        # Conscientiousness indicators
        conscientiousness_words = ['careful', 'thorough', 'systematic', 'detailed', 'organized']
        conscientiousness_score = sum(1 for word in conscientiousness_words if word in response_lower) / len(conscientiousness_words)
        personality_indicators['conscientiousness_expression'] = conscientiousness_score
        
        # Calculate overall coherence (simplified)
        trait_expressions = [openness_score, conscientiousness_score]
        coherence = 1.0 - np.std(trait_expressions) if len(trait_expressions) > 1 else 0.5
        personality_indicators['coherence'] = coherence
        
        return personality_indicators
    
    def _extract_learning_indicators(self, response: str, context: str) -> Dict[str, float]:
        """Extract learning indicators from response"""
        
        learning_indicators = {}
        
        response_lower = response.lower()
        
        # Knowledge integration
        integration_words = ['understand', 'connect', 'relate', 'integrate', 'synthesize']
        integration_score = sum(1 for word in integration_words if word in response_lower) / len(integration_words)
        learning_indicators['knowledge_integration'] = integration_score
        
        # Reflection indicators
        reflection_words = ['reflect', 'consider', 'think', 'ponder', 'contemplate']
        reflection_score = sum(1 for word in reflection_words if word in response_lower) / len(reflection_words)
        learning_indicators['reflection_depth'] = reflection_score
        
        # Adaptation indicators
        adaptation_words = ['adapt', 'adjust', 'modify', 'change', 'evolve']
        adaptation_score = sum(1 for word in adaptation_words if word in response_lower) / len(adaptation_words)
        learning_indicators['adaptation_willingness'] = adaptation_score
        
        return learning_indicators
    
    def get_response_statistics(self) -> Dict[str, Any]:
        """Get response processing statistics"""
        
        if not self.response_history:
            return {'status': 'no_response_data'}
        
        recent_responses = list(self.response_history)[-20:]  # Last 20 responses
        
        avg_length = np.mean([r['response_length'] for r in recent_responses])
        avg_quality = np.mean([r['quality_score'] for r in recent_responses])
        avg_insights = np.mean([r['insights_extracted'] for r in recent_responses])
        avg_coherence = np.mean([r['personality_coherence'] for r in recent_responses])
        
        return {
            'total_responses_processed': len(self.response_history),
            'average_response_length': avg_length,
            'average_quality_score': avg_quality,
            'average_insights_per_response': avg_insights,
            'average_personality_coherence': avg_coherence,
            'response_processing_active': True
        }

class InsightExtractor:
    """Extracts insights from LLM responses"""
    
    def __init__(self):
        self.insight_patterns = self._initialize_insight_patterns()
        self.extracted_insights = deque(maxlen=1000)
    
    def _initialize_insight_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for insight extraction"""
        
        return {
            'learning': [
                r'I (?:understand|realize|learn|discover) that',
                r'This (?:suggests|indicates|implies|shows) that',
                r'(?:It appears|It seems|I notice) that'
            ],
            'connection': [
                r'This (?:relates to|connects to|links to)',
                r'(?:Similar to|Like|Comparable to)',
                r'This (?:builds on|extends|develops)'
            ],
            'prediction': [
                r'I (?:expect|anticipate|predict) that',
                r'This (?:will likely|should|may) lead to',
                r'(?:Future|Next|Subsequently)'
            ],
            'reflection': [
                r'(?:Looking back|In retrospect|Reflecting)',
                r'This (?:changes|shifts|affects) my',
                r'I (?:reconsider|rethink|reevaluate)'
            ]
        }
    
    def extract_insights(self, response: str, context: str, 
                        experience: SensorimotorExperience) -> List[Dict[str, Any]]:
        """Extract insights from response"""
        
        insights = []
        
        # Pattern-based extraction
        for insight_type, patterns in self.insight_patterns.items():
            type_insights = self._extract_insights_by_type(response, insight_type, patterns)
            insights.extend(type_insights)
        
        # Semantic analysis
        semantic_insights = self._extract_semantic_insights(response, context)
        insights.extend(semantic_insights)
        
        # Context-based insights
        context_insights = self._extract_context_based_insights(response, context, experience)
        insights.extend(context_insights)
        
        # Record insights
        for insight in insights:
            insight_record = {
                'timestamp': datetime.now().isoformat(),
                'experience_id': experience.experience_id,
                'insight_type': insight['type'],
                'content': insight['content'][:200]  # Truncate
            }
            self.extracted_insights.append(insight_record)
        
        return insights
    
    def _extract_insights_by_type(self, response: str, insight_type: str, 
                                patterns: List[str]) -> List[Dict[str, Any]]:
        """Extract insights of a specific type"""
        
        insights = []
        import re
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                # Extract the sentence containing the match
                start = max(0, response.rfind('.', 0, match.start()) + 1)
                end = response.find('.', match.end())
                if end == -1:
                    end = len(response)
                
                sentence = response[start:end].strip()
                
                if len(sentence) > 20:  # Minimum meaningful length
                    insights.append({
                        'type': insight_type,
                        'content': sentence,
                        'confidence': 0.7,  # Pattern-based confidence
                        'extraction_method': 'pattern'
                    })
        
        return insights
    
    def _extract_semantic_insights(self, response: str, context: str) -> List[Dict[str, Any]]:
        """Extract insights through semantic analysis"""
        
        insights = []
        
        # Look for key insight indicators
        insight_indicators = [
            'insight', 'realization', 'understanding', 'discovery',
            'conclusion', 'implication', 'consequence', 'result'
        ]
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_lower = sentence.lower()
            
            # Check for insight indicators
            indicator_count = sum(1 for indicator in insight_indicators if indicator in sentence_lower)
            
            if indicator_count > 0:
                insights.append({
                    'type': 'semantic',
                    'content': sentence,
                    'confidence': min(0.9, 0.4 + indicator_count * 0.2),
                    'extraction_method': 'semantic'
                })
        
        return insights
    
    def _extract_context_based_insights(self, response: str, context: str, 
                                      experience: SensorimotorExperience) -> List[Dict[str, Any]]:
        """Extract insights based on context relevance"""
        
        insights = []
        
        # Extract domain-specific insights
        if experience.domain == 'financial_analysis':
            insights.extend(self._extract_financial_insights(response))
        elif experience.domain == 'research':
            insights.extend(self._extract_research_insights(response))
        
        return insights
    
    def _extract_financial_insights(self, response: str) -> List[Dict[str, Any]]:
        """Extract financial domain insights"""
        
        financial_insights = []
        
        financial_keywords = [
            'market', 'price', 'trend', 'volatility', 'risk',
            'investment', 'return', 'portfolio', 'analysis'
        ]
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for keyword in financial_keywords if keyword in sentence_lower)
            
            if keyword_count >= 2:  # Multiple financial keywords
                financial_insights.append({
                    'type': 'financial_insight',
                    'content': sentence,
                    'confidence': min(0.8, 0.3 + keyword_count * 0.1),
                    'extraction_method': 'domain_specific',
                    'keyword_count': keyword_count
                })
        
        return financial_insights
    
    def _extract_research_insights(self, response: str) -> List[Dict[str, Any]]:
        """Extract research domain insights"""
        
        research_insights = []
        
        research_keywords = [
            'study', 'research', 'analysis', 'methodology', 'hypothesis',
            'theory', 'evidence', 'experiment', 'finding', 'conclusion'
        ]
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for keyword in research_keywords if keyword in sentence_lower)
            
            if keyword_count >= 2:
                research_insights.append({
                    'type': 'research_insight',
                    'content': sentence,
                    'confidence': min(0.8, 0.3 + keyword_count * 0.1),
                    'extraction_method': 'domain_specific',
                    'keyword_count': keyword_count
                })
        
        return research_insights

class ResponseQualityAssessor:
    """Assesses the quality of LLM responses"""
    
    def __init__(self):
        self.quality_history = deque(maxlen=500)
        self.quality_metrics = [
            'relevance', 'coherence', 'informativeness', 
            'personality_consistency', 'depth'
        ]
    
    def assess_response(self, response: str, context: str, 
                       experience: SensorimotorExperience) -> Dict[str, float]:
        """Assess response quality across multiple dimensions"""
        
        quality_scores = {}
        
        # Relevance to experience
        quality_scores['relevance'] = self._assess_relevance(response, experience)
        
        # Coherence and structure
        quality_scores['coherence'] = self._assess_coherence(response)
        
        # Informativeness
        quality_scores['informativeness'] = self._assess_informativeness(response, context)
        
        # Personality consistency
        quality_scores['personality_consistency'] = self._assess_personality_consistency(response, context)
        
        # Depth of analysis
        quality_scores['depth'] = self._assess_depth(response)
        
        # Overall quality
        quality_scores['overall_quality'] = np.mean(list(quality_scores.values()))
        
        # Record quality assessment
        quality_record = {
            'timestamp': datetime.now().isoformat(),
            'experience_id': experience.experience_id,
            'overall_quality': quality_scores['overall_quality'],
            'component_scores': quality_scores.copy()
        }
        self.quality_history.append(quality_record)
        
        return quality_scores
    
    def _assess_relevance(self, response: str, experience: SensorimotorExperience) -> float:
        """Assess relevance of response to experience"""
        
        experience_words = set(experience.content.lower().split())
        response_words = set(response.lower().split())
        
        if not experience_words or not response_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(experience_words & response_words)
        union = len(experience_words | response_words)
        
        base_relevance = overlap / union if union > 0 else 0.0
        
        # Boost for domain consistency
        domain_words = {
            'financial_analysis': ['market', 'financial', 'economic', 'investment'],
            'research': ['research', 'study', 'analysis', 'methodology'],
            'general': ['information', 'data', 'content', 'topic']
        }
        
        domain_keywords = domain_words.get(experience.domain, [])
        domain_mentions = sum(1 for word in domain_keywords if word in response.lower())
        domain_boost = min(0.3, domain_mentions * 0.1)
        
        return min(1.0, base_relevance + domain_boost)
    
    def _assess_coherence(self, response: str) -> float:
        """Assess coherence and structure of response"""
        
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.5  # Single sentence, neutral coherence
        
        coherence_factors = []
        
        # Sentence length consistency
        sentence_lengths = [len(s.split()) for s in sentences]
        length_consistency = 1.0 - (np.std(sentence_lengths) / max(np.mean(sentence_lengths), 1))
        coherence_factors.append(max(0.0, length_consistency))
        
        # Lexical cohesion (simplified)
        all_words = set()
        sentence_words = []
        for sentence in sentences:
            words = set(sentence.lower().split())
            sentence_words.append(words)
            all_words.update(words)
        
        # Calculate word overlap between adjacent sentences
        overlaps = []
        for i in range(len(sentence_words) - 1):
            overlap = len(sentence_words[i] & sentence_words[i + 1])
            overlaps.append(overlap / max(len(sentence_words[i] | sentence_words[i + 1]), 1))
        
        avg_overlap = np.mean(overlaps) if overlaps else 0.0
        coherence_factors.append(min(1.0, avg_overlap * 3))  # Scale up
        
        return np.mean(coherence_factors)
    
    def _assess_informativeness(self, response: str, context: str) -> float:
        """Assess informativeness of response"""
        
        # Unique information density
        response_words = response.lower().split()
        context_words = context.lower().split()
        
        if not response_words:
            return 0.0
        
        # Information not already in context
        response_word_set = set(response_words)
        context_word_set = set(context_words)
        
        new_information = response_word_set - context_word_set
        information_ratio = len(new_information) / len(response_word_set)
        
        # Complexity indicators
        complexity_indicators = [
            'because', 'therefore', 'however', 'moreover', 'furthermore',
            'specifically', 'particularly', 'especially', 'notably'
        ]
        
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in response.lower())
        complexity_factor = min(1.0, complexity_count * 0.2)
        
        # Combine factors
        informativeness = (information_ratio * 0.7 + complexity_factor * 0.3)
        
        return min(1.0, informativeness)
    
    def _assess_personality_consistency(self, response: str, context: str) -> float:
        """Assess personality consistency in response"""
        
        # Extract personality indicators from context
        context_lower = context.lower()
        
        # Look for trait indicators in context
        trait_indicators = {
            'openness': ['creative', 'innovative', 'curious', 'open'],
            'conscientiousness': ['careful', 'thorough', 'organized', 'systematic'],
            'extraversion': ['outgoing', 'energetic', 'assertive', 'social'],
            'agreeableness': ['cooperative', 'helpful', 'kind', 'supportive'],
            'neuroticism': ['anxious', 'worried', 'stressed', 'emotional']
        }
        
        context_traits = {}
        for trait, indicators in trait_indicators.items():
            context_traits[trait] = sum(1 for indicator in indicators if indicator in context_lower)
        
        # Look for same traits in response
        response_lower = response.lower()
        response_traits = {}
        for trait, indicators in trait_indicators.items():
            response_traits[trait] = sum(1 for indicator in indicators if indicator in response_lower)
        
        # Calculate consistency
        consistency_scores = []
        for trait in trait_indicators.keys():
            context_score = context_traits[trait]
            response_score = response_traits[trait]
            
            if context_score > 0 or response_score > 0:
                # If trait is present in either, check for consistency
                if context_score > 0 and response_score > 0:
                    consistency_scores.append(1.0)  # Both present - consistent
                elif context_score > 0 and response_score == 0:
                    consistency_scores.append(0.3)  # Expected but missing
                else:
                    consistency_scores.append(0.6)  # New trait expression
        
        return np.mean(consistency_scores) if consistency_scores else 0.7  # Default neutral
    
    def _assess_depth(self, response: str) -> float:
        """Assess depth of analysis in response"""
        
        depth_indicators = [
            'analysis', 'consider', 'examine', 'explore', 'investigate',
            'implications', 'consequences', 'factors', 'aspects', 'dimensions',
            'perspective', 'viewpoint', 'approach', 'strategy', 'methodology'
        ]
        
        response_lower = response.lower()
        depth_count = sum(1 for indicator in depth_indicators if indicator in response_lower)
        
        # Structural depth indicators
        questions = response.count('?')
        examples = response.lower().count('for example') + response.lower().count('such as')
        explanations = response.lower().count('because') + response.lower().count('since')
        
        structural_depth = questions + examples + explanations
        
        # Combine indicators
        total_depth = depth_count + structural_depth * 0.5
        
        # Normalize by response length
        words = len(response.split())
        depth_density = total_depth / max(words / 50.0, 1)  # Per 50 words
        
        return min(1.0, depth_density)
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality assessment statistics"""
        
        if not self.quality_history:
            return {'status': 'no_quality_data'}
        
        recent_assessments = list(self.quality_history)[-20:]
        
        stats = {}
        
        for metric in self.quality_metrics + ['overall_quality']:
            scores = [assessment['component_scores'].get(metric, 0.5) for assessment in recent_assessments]
            stats[f'avg_{metric}'] = np.mean(scores)
            stats[f'std_{metric}'] = np.std(scores)
        
        stats['total_assessments'] = len(self.quality_history)
        stats['assessment_trend'] = self._calculate_quality_trend()
        
        return stats
    
    def _calculate_quality_trend(self) -> str:
        """Calculate quality trend over time"""
        
        if len(self.quality_history) < 10:
            return 'insufficient_data'
        
        recent_10 = [assessment['overall_quality'] for assessment in list(self.quality_history)[-10:]]
        previous_10 = [assessment['overall_quality'] for assessment in list(self.quality_history)[-20:-10]]
        
        recent_avg = np.mean(recent_10)
        previous_avg = np.mean(previous_10) if previous_10 else recent_avg
        
        if recent_avg > previous_avg + 0.05:
            return 'improving'
        elif recent_avg < previous_avg - 0.05:
            return 'declining'
        else:
            return 'stable'

# ============================================================================
# MULTI-AGENT COORDINATION (COMPLETE ORIGINAL)
# ============================================================================

class MultiAgentCoordinator:
    """Coordinates multiple AI agents with shared episodic memory"""
    
    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.agents = {}
        self.shared_episodic_memory = None  # Shared across agents
        self.coordination_history = deque(maxlen=1000)
        self.agent_interactions = defaultdict(list)
        
        # Coordination protocols
        self.coordination_modes = ['independent', 'collaborative', 'competitive', 'consensus']
        self.current_mode = 'collaborative'
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize multiple AI agents with different specializations"""
        
        agent_configs = [
            {
                'agent_id': 'analyst',
                'domain': 'financial_analysis',
                'personality_seed': {'openness': 0.8, 'conscientiousness': 0.9, 'analytical': 0.9},
                'specialization': 'deep_analysis'
            },
            {
                'agent_id': 'researcher',
                'domain': 'research',
                'personality_seed': {'openness': 0.9, 'conscientiousness': 0.8, 'curiosity': 0.9},
                'specialization': 'knowledge_synthesis'
            },
            {
                'agent_id': 'integrator',
                'domain': 'general',
                'personality_seed': {'agreeableness': 0.9, 'openness': 0.7, 'social': 0.8},
                'specialization': 'perspective_integration'
            }
        ]
        
        for config in agent_configs[:self.num_agents]:
            agent = EnhancedPersistentIdentityAI(
                domain=config['domain'],
                personality_seed=config['personality_seed'],
                model_name="gemma3n:e4b"
            )
            
            # Set agent metadata
            agent.agent_id = config['agent_id']
            agent.specialization = config['specialization']
            
            self.agents[config['agent_id']] = agent
    
    def set_shared_episodic_memory(self, episodic_memory_engine):
        """Set shared episodic memory for all agents"""
        self.shared_episodic_memory = episodic_memory_engine
        
        # Connect all agents to shared memory
        for agent in self.agents.values():
            agent.episodic_memory_engine = episodic_memory_engine
            agent.cortical_processor.set_episodic_memory_engine(episodic_memory_engine)
            agent.identity_processor.set_episodic_memory_engine(episodic_memory_engine)
    
    def coordinate_multi_agent_processing(self, experience: SensorimotorExperience) -> Dict[str, Any]:
        """Coordinate processing across multiple agents"""
        
        coordination_start = time.time()
        
        # Phase 1: Independent processing
        agent_results = self._independent_processing_phase(experience)
        
        # Phase 2: Cross-agent analysis
        cross_analysis = self._cross_agent_analysis_phase(agent_results, experience)
        
        # Phase 3: Coordination based on current mode
        if self.current_mode == 'collaborative':
            coordination_result = self._collaborative_coordination(agent_results, cross_analysis, experience)
        elif self.current_mode == 'competitive':
            coordination_result = self._competitive_coordination(agent_results, cross_analysis, experience)
        elif self.current_mode == 'consensus':
            coordination_result = self._consensus_coordination(agent_results, cross_analysis, experience)
        else:  # independent
            coordination_result = self._independent_coordination(agent_results, cross_analysis, experience)
        
        # Phase 4: Shared learning update
        shared_learning = self._update_shared_learning(coordination_result, experience)
        
        # Record coordination
        coordination_record = {
            'timestamp': datetime.now().isoformat(),
            'experience_id': experience.experience_id,
            'coordination_mode': self.current_mode,
            'participating_agents': list(agent_results.keys()),
            'processing_time': time.time() - coordination_start,
            'consensus_achieved': coordination_result.get('consensus_achieved', False),
            'shared_learning_quality': shared_learning.get('learning_quality', 0.5)
        }
        self.coordination_history.append(coordination_record)
        
        return {
            'coordination_mode': self.current_mode,
            'agent_results': agent_results,
            'cross_analysis': cross_analysis,
            'coordination_result': coordination_result,
            'shared_learning': shared_learning,
            'coordination_metadata': coordination_record
        }
    
    def _independent_processing_phase(self, experience: SensorimotorExperience) -> Dict[str, Dict[str, Any]]:
        """Phase 1: Each agent processes independently"""
        
        agent_results = {}
        
        for agent_id, agent in self.agents.items():
            try:
                # Process experience through agent
                result = agent.process_experience(experience)
                agent_results[agent_id] = {
                    'processing_result': result,
                    'agent_specialization': agent.specialization,
                    'personality_coherence': result.get('identity_result', {}).get('coherence_assessment', {}).get('overall_coherence', 0.5),
                    'prediction_confidence': result.get('cortical_result', {}).get('consensus', {}).get('overall_confidence', 0.5)
                }
                
            except Exception as e:
                logger.error(f"Agent {agent_id} processing failed: {e}")
                agent_results[agent_id] = {'error': str(e)}
        
        return agent_results
    
    def _cross_agent_analysis_phase(self, agent_results: Dict[str, Dict[str, Any]], 
                                  experience: SensorimotorExperience) -> Dict[str, Any]:
        """Phase 2: Analyze differences and similarities across agents"""
        
        cross_analysis = {
            'agreement_metrics': {},
            'divergence_analysis': {},
            'complementarity_assessment': {},
            'expertise_distribution': {}
        }
        
        # Extract key metrics from each agent
        agent_metrics = {}
        for agent_id, result in agent_results.items():
            if 'error' in result:
                continue
            
            processing_result = result['processing_result']
            
            agent_metrics[agent_id] = {
                'prediction_confidence': result['prediction_confidence'],
                'personality_coherence': result['personality_coherence'],
                'cortical_expertise': processing_result.get('cortical_result', {}).get('domain_expertise_level', 0.5),
                'identity_stability': processing_result.get('identity_result', {}).get('personality_state', {}).get('identity_stability', 0.5)
            }
        
        # Calculate agreement metrics
        if len(agent_metrics) > 1:
            cross_analysis['agreement_metrics'] = self._calculate_agent_agreement(agent_metrics)
            cross_analysis['divergence_analysis'] = self._analyze_agent_divergence(agent_metrics)
            cross_analysis['complementarity_assessment'] = self._assess_agent_complementarity(agent_results)
            cross_analysis['expertise_distribution'] = self._analyze_expertise_distribution(agent_metrics)
        
        return cross_analysis
    
    def _calculate_agent_agreement(self, agent_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate agreement between agents"""
        
        agreement_metrics = {}
        agent_ids = list(agent_metrics.keys())
        
        # Calculate pairwise agreements
        pairwise_agreements = []
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent1_metrics = agent_metrics[agent_ids[i]]
                agent2_metrics = agent_metrics[agent_ids[j]]
                
                # Calculate agreement on each metric
                metric_agreements = []
                for metric in agent1_metrics.keys():
                    val1 = agent1_metrics[metric]
                    val2 = agent2_metrics[metric]
                    agreement = 1.0 - abs(val1 - val2)  # Higher agreement = smaller difference
                    metric_agreements.append(agreement)
                
                avg_agreement = np.mean(metric_agreements)
                pairwise_agreements.append(avg_agreement)
        
        agreement_metrics['average_pairwise_agreement'] = np.mean(pairwise_agreements) if pairwise_agreements else 1.0
        agreement_metrics['agreement_variance'] = np.var(pairwise_agreements) if pairwise_agreements else 0.0
        
        # Calculate consensus strength
        all_values = {}
        for metric in ['prediction_confidence', 'personality_coherence', 'cortical_expertise']:
            values = [metrics[metric] for metrics in agent_metrics.values()]
            all_values[metric] = 1.0 - np.std(values)  # Lower std = higher consensus
        
        agreement_metrics['consensus_strength'] = np.mean(list(all_values.values()))
        
        return agreement_metrics
    
    def _analyze_agent_divergence(self, agent_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze where agents diverge most"""
        
        divergence_analysis = {}
        
        # Find metrics with highest variance
        metric_variances = {}
        for metric in ['prediction_confidence', 'personality_coherence', 'cortical_expertise', 'identity_stability']:
            values = [metrics[metric] for metrics in agent_metrics.values()]
            metric_variances[metric] = np.var(values)
        
        # Sort by variance
        sorted_variances = sorted(metric_variances.items(), key=lambda x: x[1], reverse=True)
        
        divergence_analysis['highest_divergence_metric'] = sorted_variances[0][0] if sorted_variances else None
        divergence_analysis['divergence_scores'] = metric_variances
        divergence_analysis['total_divergence'] = sum(metric_variances.values())
        
        return divergence_analysis
    
    def _assess_agent_complementarity(self, agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess how agents complement each other"""
        
        complementarity = {}
        
        # Analyze specialization coverage
        specializations = {}
        for agent_id, result in agent_results.items():
            if 'error' not in result:
                specialization = result['agent_specialization']
                specializations[agent_id] = specialization
        
        complementarity['specialization_diversity'] = len(set(specializations.values())) / max(len(specializations), 1)
        
        # Analyze domain coverage
        domains = {}
        for agent_id, result in agent_results.items():
            if 'error' not in result:
                processing_result = result['processing_result']
                cortical_result = processing_result.get('cortical_result', {})
                domain_expertise = cortical_result.get('domain_expertise_level', 0.5)
                domains[agent_id] = domain_expertise
        
        if domains:
            complementarity['domain_coverage_balance'] = 1.0 - np.std(list(domains.values()))
            complementarity['total_domain_strength'] = np.mean(list(domains.values()))
        
        return complementarity
    
    def _analyze_expertise_distribution(self, agent_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze expertise distribution across agents"""
        
        expertise_analysis = {}
        
        # Find expertise leaders for each metric
        expertise_leaders = {}
        for metric in ['prediction_confidence', 'cortical_expertise', 'identity_stability']:
            agent_scores = {agent_id: metrics[metric] for agent_id, metrics in agent_metrics.items()}
            leader = max(agent_scores.items(), key=lambda x: x[1])
            expertise_leaders[metric] = {'agent': leader[0], 'score': leader[1]}
        
        expertise_analysis['expertise_leaders'] = expertise_leaders
        
        # Calculate expertise concentration
        all_scores = []
        for metrics in agent_metrics.values():
            all_scores.extend(metrics.values())
        
        expertise_analysis['expertise_concentration'] = np.std(all_scores) if all_scores else 0.0
        expertise_analysis['average_expertise'] = np.mean(all_scores) if all_scores else 0.5
        
        return expertise_analysis
    
    def _collaborative_coordination(self, agent_results: Dict[str, Dict[str, Any]], 
                                  cross_analysis: Dict[str, Any], 
                                  experience: SensorimotorExperience) -> Dict[str, Any]:
        """Collaborative coordination mode"""
        
        coordination_result = {'mode': 'collaborative', 'consensus_achieved': False}
        
        # Combine agent insights
        combined_insights = []
        combined_predictions = {}
        combined_personality_updates = {}
        
        for agent_id, result in agent_results.items():
            if 'error' in result:
                continue
            
            processing_result = result['processing_result']
            
            # Collect insights
            insights = processing_result.get('insights', [])
            combined_insights.extend(insights)
            
            # Collect predictions
            cortical_result = processing_result.get('cortical_result', {})
            consensus = cortical_result.get('consensus', {})
            agent_predictions = consensus.get('consensus_actions', {})
            
            for action, prediction in agent_predictions.items():
                if action not in combined_predictions:
                    combined_predictions[action] = []
                combined_predictions[action].append({
                    'agent': agent_id,
                    'prediction': prediction
                })
            
            # Collect personality updates
            identity_result = processing_result.get('identity_result', {})
            personality_state = identity_result.get('personality_state', {})
            
            if personality_state:
                combined_personality_updates[agent_id] = personality_state
        
        # Generate collaborative consensus
        consensus_predictions = self._generate_collaborative_consensus(combined_predictions)
        consensus_personality = self._generate_collaborative_personality_consensus(combined_personality_updates)
        
        coordination_result.update({
            'combined_insights': combined_insights[:10],  # Limit for space
            'consensus_predictions': consensus_predictions,
            'consensus_personality_traits': consensus_personality,
            'collaboration_quality': cross_analysis.get('agreement_metrics', {}).get('consensus_strength', 0.5),
            'participating_agents': len([r for r in agent_results.values() if 'error' not in r])
        })
        
        return coordination_result
    
    def _competitive_coordination(self, agent_results: Dict[str, Dict[str, Any]], 
                                cross_analysis: Dict[str, Any], 
                                experience: SensorimotorExperience) -> Dict[str, Any]:
        """Competitive coordination mode"""
        
        coordination_result = {'mode': 'competitive', 'consensus_achieved': False}
        
        # Rank agents by performance
        agent_performance = {}
        
        for agent_id, result in agent_results.items():
            if 'error' in result:
                agent_performance[agent_id] = 0.0
                continue
            
            processing_result = result['processing_result']
            
            # Calculate performance score
            prediction_confidence = result['prediction_confidence']
            personality_coherence = result['personality_coherence']
            
            cortical_result = processing_result.get('cortical_result', {})
            domain_expertise = cortical_result.get('domain_expertise_level', 0.5)
            
            performance_score = (prediction_confidence * 0.4 + 
                               personality_coherence * 0.3 + 
                               domain_expertise * 0.3)
            
            agent_performance[agent_id] = performance_score
        
        # Select winning agent
        winning_agent = max(agent_performance.items(), key=lambda x: x[1])
        
        coordination_result.update({
            'winning_agent': winning_agent[0],
            'winning_score': winning_agent[1],
            'agent_rankings': sorted(agent_performance.items(), key=lambda x: x[1], reverse=True),
            'performance_gap': winning_agent[1] - min(agent_performance.values()) if agent_performance else 0.0,
            'winner_result': agent_results.get(winning_agent[0], {}).get('processing_result', {})
        })
        
        return coordination_result
    
    def _consensus_coordination(self, agent_results: Dict[str, Dict[str, Any]], 
                              cross_analysis: Dict[str, Any], 
                              experience: SensorimotorExperience) -> Dict[str, Any]:
        """Consensus-based coordination mode"""
        
        coordination_result = {'mode': 'consensus', 'consensus_achieved': False}
        
        # Check if consensus is achievable
        agreement_metrics = cross_analysis.get('agreement_metrics', {})
        consensus_strength = agreement_metrics.get('consensus_strength', 0.0)
        
        if consensus_strength > 0.7:  # High consensus threshold
            # Achieve consensus through averaging
            consensus_result = self._achieve_consensus(agent_results)
            coordination_result.update(consensus_result)
            coordination_result['consensus_achieved'] = True
        else:
            # Fall back to collaborative mode
            collaboration_result = self._collaborative_coordination(agent_results, cross_analysis, experience)
            coordination_result.update(collaboration_result)
            coordination_result['consensus_achieved'] = False
            coordination_result['fallback_reason'] = f'consensus_strength_too_low_{consensus_strength:.2f}'
        
        return coordination_result
    
    def _independent_coordination(self, agent_results: Dict[str, Dict[str, Any]], 
                                cross_analysis: Dict[str, Any], 
                                experience: SensorimotorExperience) -> Dict[str, Any]:
        """Independent coordination mode"""
        
        coordination_result = {'mode': 'independent', 'consensus_achieved': False}
        
        # Simply package individual results
        agent_summaries = {}
        
        for agent_id, result in agent_results.items():
            if 'error' in result:
                agent_summaries[agent_id] = {'status': 'error', 'error': result['error']}
                continue
            
            processing_result = result['processing_result']
            
            agent_summaries[agent_id] = {
                'status': 'success',
                'prediction_confidence': result['prediction_confidence'],
                'personality_coherence': result['personality_coherence'],
                'specialization': result['agent_specialization'],
                'key_insights': len(processing_result.get('insights', [])),
                'identity_stability': processing_result.get('identity_result', {}).get('personality_state', {}).get('identity_stability', 0.5)
            }
        
        coordination_result.update({
            'agent_summaries': agent_summaries,
            'total_active_agents': len([r for r in agent_results.values() if 'error' not in r]),
            'cross_agent_analysis': cross_analysis
        })
        
        return coordination_result
    
    def _generate_collaborative_consensus(self, combined_predictions: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate consensus from collaborative predictions"""
        
        consensus_predictions = {}
        
        for action, agent_predictions in combined_predictions.items():
            if not agent_predictions:
                continue
            
            # Extract prediction values
            values = []
            confidences = []
            
            for pred_data in agent_predictions:
                prediction = pred_data['prediction']
                if isinstance(prediction, dict):
                    values.append(prediction.get('strength', 0.5))
                    confidences.append(prediction.get('confidence', 0.5))
                else:
                    values.append(float(prediction) if isinstance(prediction, (int, float)) else 0.5)
                    confidences.append(0.7)  # Default confidence
            
            if values:
                consensus_predictions[action] = {
                    'consensus_value': np.mean(values),
                    'confidence': np.mean(confidences),
                    'agreement_level': 1.0 - np.std(values),
                    'supporting_agents': len(agent_predictions)
                }
        
        return consensus_predictions
    
    def _generate_collaborative_personality_consensus(self, combined_personality_updates: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate personality consensus across agents"""
        
        if not combined_personality_updates:
            return {}
        
        # Extract trait values
        trait_values = defaultdict(list)
        
        for agent_id, personality_state in combined_personality_updates.items():
            traits_big5 = personality_state.get('traits_big5', {})
            for trait, value in traits_big5.items():
                trait_values[trait].append(value)
        
        # Calculate consensus traits
        consensus_traits = {}
        for trait, values in trait_values.items():
            if values:
                consensus_traits[trait] = {
                    'consensus_value': np.mean(values),
                    'agreement_level': 1.0 - np.std(values),
                    'supporting_agents': len(values)
                }
        
        return consensus_traits
    
    def _achieve_consensus(self, agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Achieve consensus through systematic agreement"""
        
        consensus_result = {}
        
        # Consensus on predictions
        all_predictions = {}
        for agent_id, result in agent_results.items():
            if 'error' in result:
                continue
            
            processing_result = result['processing_result']
            cortical_result = processing_result.get('cortical_result', {})
            consensus = cortical_result.get('consensus', {})
            actions = consensus.get('consensus_actions', {})
            
            for action, action_data in actions.items():
                if action not in all_predictions:
                    all_predictions[action] = []
                all_predictions[action].append(action_data)
        
        consensus_predictions = {}
        for action, action_list in all_predictions.items():
            if len(action_list) >= 2:  # Require at least 2 agents to agree
                consensus_predictions[action] = {
                    'achieved': True,
                    'supporting_agents': len(action_list),
                    'consensus_strength': len(action_list) / len(agent_results)
                }
        
        consensus_result['consensus_predictions'] = consensus_predictions
        consensus_result['consensus_quality'] = len(consensus_predictions) / max(len(all_predictions), 1)
        
        return consensus_result
    
    def _update_shared_learning(self, coordination_result: Dict[str, Any], 
                              experience: SensorimotorExperience) -> Dict[str, Any]:
        """Update shared learning across agents"""
        
        shared_learning = {'learning_quality': 0.5}
        
        if not self.shared_episodic_memory:
            return shared_learning
        
        # Extract learning insights from coordination
        coordination_mode = coordination_result['mode']
        
        if coordination_mode == 'collaborative':
            # Share insights across agents
            combined_insights = coordination_result.get('combined_insights', [])
            consensus_predictions = coordination_result.get('consensus_predictions', {})
            
            # Quality based on agreement and insight richness
            collaboration_quality = coordination_result.get('collaboration_quality', 0.5)
            insight_richness = min(1.0, len(combined_insights) / 5.0)
            
            learning_quality = (collaboration_quality * 0.6 + insight_richness * 0.4)
            
        elif coordination_mode == 'competitive':
            # Learn from best performer
            winning_score = coordination_result.get('winning_score', 0.5)
            performance_gap = coordination_result.get('performance_gap', 0.0)
            
            learning_quality = winning_score * 0.7 + performance_gap * 0.3
            
        elif coordination_mode == 'consensus':
            # Learn from consensus strength
            consensus_achieved = coordination_result.get('consensus_achieved', False)
            if consensus_achieved:
                consensus_quality = coordination_result.get('consensus_quality', 0.5)
                learning_quality = consensus_quality
            else:
                learning_quality = 0.3  # Lower quality if consensus failed
        
        else:  # independent
            # Learn from diversity
            total_active = coordination_result.get('total_active_agents', 1)
            diversity_bonus = min(0.3, total_active * 0.1)
            learning_quality = 0.5 + diversity_bonus
        
        shared_learning['learning_quality'] = learning_quality
        shared_learning['coordination_mode'] = coordination_mode
        shared_learning['experience_id'] = experience.experience_id
        
        return shared_learning
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get multi-agent coordination statistics"""
        
        if not self.coordination_history:
            return {'status': 'no_coordination_data'}
        
        recent_coordinations = list(self.coordination_history)[-20:]
        
        # Mode usage statistics
        mode_usage = defaultdict(int)
        for coord in recent_coordinations:
            mode_usage[coord['coordination_mode']] += 1
        
        # Performance statistics
        avg_processing_time = np.mean([coord['processing_time'] for coord in recent_coordinations])
        consensus_rate = np.mean([coord['consensus_achieved'] for coord in recent_coordinations])
        avg_learning_quality = np.mean([coord['shared_learning_quality'] for coord in recent_coordinations])
        
        # Agent participation
        agent_participation = defaultdict(int)
        for coord in recent_coordinations:
            for agent_id in coord['participating_agents']:
                agent_participation[agent_id] += 1
        
        return {
            'total_coordinations': len(self.coordination_history),
            'current_coordination_mode': self.current_mode,
            'mode_usage_distribution': dict(mode_usage),
            'performance_metrics': {
                'average_processing_time': avg_processing_time,
                'consensus_achievement_rate': consensus_rate,
                'average_learning_quality': avg_learning_quality
            },
            'agent_statistics': {
                'total_agents': len(self.agents),
                'agent_participation_rates': dict(agent_participation),
                'active_agents': len([a for a in self.agents.values() if hasattr(a, 'agent_id')])
            },
            'coordination_active': True
        }

# ============================================================================
# ENHANCED INTEGRATED SYSTEM
# ============================================================================

class EnhancedIntegratedMemorySystem:
    """Complete integrated memory system with all enhancements"""
    
    def __init__(self, domain: str = "general", model_architecture: str = "gemma3n:e4b"):
        self.domain = domain
        self.model_architecture = model_architecture
        self.session_id = uuid.uuid4().hex[:8]
        
        # Initialize all memory components
        self.token_manager = TokenLevelContextManager(
            context_window=self._get_context_window(model_architecture)
        )
        self.boundary_refiner = AdvancedBoundaryRefiner()
        self.hierarchical_memory = HierarchicalMemorySystem()
        self.compression_system = MemoryCompressionSystem()
        self.retrieval_system = AdvancedRetrievalSystem()
        self.cross_modal_system = CrossModalMemorySystem()
        
        # Integration coordination
        self.integration_stats = defaultdict(int)
        self.performance_metrics = deque(maxlen=1000)
        
        logger.info(f"Enhanced Integrated Memory System initialized for {domain}")
        logger.info(f"Model architecture: {model_architecture}")
        logger.info(f"Context window: {self.token_manager.context_window}")
    
    def _get_context_window(self, model_architecture: str) -> int:
        """Get context window size based on model architecture"""
        if "gemma3n" in model_architecture:
            return 32000
        elif "deepseek" in model_architecture:
            return 4000
        elif "qwen" in model_architecture:
            return 8000
        else:
            return 8000
    
    def process_experience_comprehensive(self, experience: SensorimotorExperience) -> Dict[str, Any]:
        """Process experience comprehensively with robust error handling - FIXED"""
        
        start_time = time.time()
        result = {
            'experience_id': getattr(experience, 'experience_id', f'exp_{uuid.uuid4().hex[:8]}'),
            'processing_timestamp': datetime.now().isoformat(),
            'status': 'processing'
        }
        
        try:
            # Validate experience
            if not hasattr(experience, 'content') or not experience.content:
                result['status'] = 'error'
                result['error'] = 'Invalid experience: missing content'
                return result
            
            # Safe hierarchical storage
            try:
                hierarchical_result = self.hierarchical_memory.store_experience(experience)
                result['hierarchical_storage'] = hierarchical_result
            except Exception as e:
                logger.error(f"Hierarchical storage failed: {e}")
                result['hierarchical_storage'] = {'error': str(e), 'storage_level': 'short_term'}
            
            # Safe cross-modal processing
            try:
                cross_modal_result = self.cross_modal_system.process_cross_modal_experience(experience)
                result['cross_modal_processing'] = cross_modal_result
            except Exception as e:
                logger.error(f"Cross-modal processing failed: {e}")
                result['cross_modal_processing'] = {'error': str(e), 'associations_created': 0}
            
            # Safe token management
            try:
                token_result = self.token_manager.process_tokens(experience.content)
                result['token_management'] = token_result
            except Exception as e:
                logger.error(f"Token management failed: {e}")
                result['token_management'] = {'error': str(e), 'tokens_processed': 0}
            
            # Processing time
            result['total_processing_time'] = time.time() - start_time
            result['status'] = 'completed'
            
            return result
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['total_processing_time'] = time.time() - start_time
            logger.error(f"Experience processing failed: {e}")
            return result
    
    def retrieve_comprehensive(self, query_experience: SensorimotorExperience, 
                             max_results: int = 25) -> Dict[str, Any]:
        """Comprehensive retrieval using all memory systems"""
        
        start_time = time.time()
        retrieval_results = {}
        
        # Retrieve from each system
        hierarchical_memories = self.hierarchical_memory.retrieve_memories(query_experience, max_results // 3)
        cross_modal_memories = self.cross_modal_system.retrieve_cross_modal(query_experience, max_results=max_results // 3)
        
        if self.hierarchical_memory.long_term_memory:
            advanced_memories = self.retrieval_system.multi_strategy_retrieval(
                query_experience, self.hierarchical_memory.long_term_memory, max_results=max_results // 3
            )
        else:
            advanced_memories = []
        
        # Combine and deduplicate results
        all_memories = self._combine_and_deduplicate_memories(
            hierarchical_memories, cross_modal_memories, advanced_memories
        )
        
        # Rank combined results
        ranked_memories = self._rank_combined_memories(all_memories, query_experience)
        
        retrieval_results = {
            'hierarchical_count': len(hierarchical_memories),
            'cross_modal_count': len(cross_modal_memories),
            'advanced_count': len(advanced_memories),
            'total_unique_memories': len(all_memories),
            'final_results': ranked_memories[:max_results],
            'retrieval_time': time.time() - start_time
        }
        
        return retrieval_results
    
    def _experience_to_tokens(self, experience: SensorimotorExperience) -> List[str]:
        """Convert experience to token representation"""
        
        tokens = []
        
        # Add content tokens
        if hasattr(experience, 'content'):
            content_tokens = experience.content.split()
            tokens.extend(content_tokens)
        
        # Add domain token
        if hasattr(experience, 'domain'):
            tokens.append(f"<domain:{experience.domain}>")
        
        # Add metadata tokens
        if hasattr(experience, 'novelty_score'):
            novelty_bucket = int(experience.novelty_score * 10)
            tokens.append(f"<novelty:{novelty_bucket}>")
        
        return tokens
    
    def _get_recent_experiences_for_boundary_analysis(self) -> List[Dict]:
        """Get recent experiences for boundary analysis"""
        
        recent_experiences = []
        
        # Get from working and short-term memory
        for memory_item in list(self.hierarchical_memory.working_memory) + list(self.hierarchical_memory.short_term_memory):
            experience_dict = {
                'episode_id': memory_item['item_id'],
                'content': memory_item['experience'].content,
                'domain': memory_item['experience'].domain,
                'timestamp': memory_item['experience'].timestamp,
                'novelty_score': memory_item['experience'].novelty_score,
                'representative_tokens': getattr(memory_item['experience'], 'representative_tokens', [])
            }
            recent_experiences.append(experience_dict)
        
        # Sort by timestamp
        recent_experiences.sort(key=lambda x: x.get('timestamp', ''))
        
        return recent_experiences[-20:]  # Last 20 experiences
    
    def _periodic_compression(self, experiences: List[Dict]) -> Dict[str, Any]:
        """Perform periodic compression of experience sequences"""
        
        if len(experiences) < 5:
            return {'status': 'insufficient_experiences'}
        
        try:
            # Compress sequence
            compressed_block, compression_stats = self.compression_system.compress_episode_sequence(experiences)
            
            if compressed_block:
                # Store compressed block (in a real implementation, this would go to persistent storage)
                compression_key = f"compressed_{self.session_id}_{int(time.time())}"
                
                return {
                    'status': 'success',
                    'compressed_block_id': compressed_block.block_id,
                    'compression_ratio': compressed_block.compression_ratio,
                    'reconstruction_fidelity': compressed_block.reconstruction_fidelity,
                    'original_episodes': len(experiences),
                    'compression_stats': compression_stats
                }
            else:
                return {'status': 'compression_failed'}
                
        except Exception as e:
            logger.error(f"Periodic compression failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _combine_and_deduplicate_memories(self, hierarchical_memories: List[Dict], 
                                        cross_modal_memories: List[Dict],
                                        advanced_memories: List[Dict]) -> List[Dict]:
        """Combine and deduplicate memories from different systems"""
        
        all_memories = {}
        
        # Add hierarchical memories
        for memory in hierarchical_memories:
            memory_id = memory['memory_item']['item_id']
            all_memories[memory_id] = {
                'memory_item': memory['memory_item'],
                'sources': ['hierarchical'],
                'scores': {'hierarchical': memory['relevance_score']},
                'memory_level': memory['memory_level']
            }
        
        # Add cross-modal memories
        for memory in cross_modal_memories:
            memory_id = memory['experience_id']
            
            if memory_id in all_memories:
                all_memories[memory_id]['sources'].append('cross_modal')
                all_memories[memory_id]['scores']['cross_modal'] = memory['final_score']
            else:
                all_memories[memory_id] = {
                    'memory_item': {'experience_id': memory_id},
                    'sources': ['cross_modal'],
                    'scores': {'cross_modal': memory['final_score']},
                    'modality_matches': memory['modality_matches']
                }
        
        # Add advanced retrieval memories
        for memory in advanced_memories:
            memory_id = memory['memory_item'].get('item_id', id(memory['memory_item']))
            
            if memory_id in all_memories:
                all_memories[memory_id]['sources'].append('advanced')
                all_memories[memory_id]['scores']['advanced'] = memory['final_ranking_score']
            else:
                all_memories[memory_id] = {
                    'memory_item': memory['memory_item'],
                    'sources': ['advanced'],
                    'scores': {'advanced': memory['final_ranking_score']},
                    'supporting_strategies': memory['supporting_strategies']
                }
        
        return list(all_memories.values())
    
    def _rank_combined_memories(self, combined_memories: List[Dict], 
                              query_experience: SensorimotorExperience) -> List[Dict]:
        """Rank combined memories using ensemble scoring"""
        
        scored_memories = []
        
        for memory in combined_memories:
            # Calculate ensemble score
            scores = memory['scores']
            sources = memory['sources']
            
            # Weight different sources
            source_weights = {
                'hierarchical': 0.4,
                'cross_modal': 0.35,
                'advanced': 0.25
            }
            
            ensemble_score = 0.0
            total_weight = 0.0
            
            for source in sources:
                if source in scores:
                    weight = source_weights.get(source, 1.0)
                    ensemble_score += scores[source] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_score /= total_weight
            
            # Bonus for multiple source agreement
            if len(sources) > 1:
                ensemble_score += (len(sources) - 1) * 0.1
            
            memory['ensemble_score'] = ensemble_score
            scored_memories.append(memory)
        
        # Sort by ensemble score
        scored_memories.sort(key=lambda x: x['ensemble_score'], reverse=True)
        
        return scored_memories
    
    def _update_integration_statistics(self, processing_results: Dict[str, Any], 
                                     total_time: float):
        """Update integration performance statistics"""
        
        self.integration_stats['experiences_processed'] += 1
        self.integration_stats['total_processing_time'] += total_time
        self.integration_stats['token_operations'] += 1
        self.integration_stats['hierarchical_operations'] += 1
        
        if 'cross_modal_storage' in processing_results:
            self.integration_stats['cross_modal_operations'] += 1
        
        if processing_results.get('boundary_refinement', {}).get('boundary_metrics'):
            self.integration_stats['boundary_refinements'] += 1
        
        if processing_results.get('compression', {}).get('status') == 'success':
            self.integration_stats['compressions_performed'] += 1
        
        # Store performance metrics
        self.performance_metrics.append({
            'timestamp': time.time(),
            'processing_time': total_time,
            'components_active': len([k for k, v in processing_results.items() 
                                    if isinstance(v, dict) and v.get('processing_time', 0) > 0])
        })
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        # Component statistics
        token_stats = self.token_manager.get_memory_statistics()
        hierarchical_stats = self.hierarchical_memory.get_memory_statistics()
        compression_stats = self.compression_system.get_compression_statistics()
        retrieval_stats = self.retrieval_system.get_retrieval_statistics()
        cross_modal_stats = self.cross_modal_system.get_cross_modal_statistics()
        
        # Integration statistics
        if self.performance_metrics:
            avg_processing_time = np.mean([m['processing_time'] for m in self.performance_metrics])
            processing_efficiency = len(self.performance_metrics) / (time.time() - self.performance_metrics[0]['timestamp']) if self.performance_metrics else 0
        else:
            avg_processing_time = 0
            processing_efficiency = 0
        
        integration_stats = {
            'session_id': self.session_id,
            'domain': self.domain,
            'model_architecture': self.model_architecture,
            'experiences_processed': self.integration_stats['experiences_processed'],
            'avg_processing_time': avg_processing_time,
            'processing_efficiency': processing_efficiency,
            'operations_performed': dict(self.integration_stats)
        }
        
        return {
            'integration_stats': integration_stats,
            'component_stats': {
                'token_management': token_stats,
                'hierarchical_memory': hierarchical_stats,
                'compression_system': compression_stats,
                'retrieval_system': retrieval_stats,
                'cross_modal_system': cross_modal_stats
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown the memory system"""
        
        # Stop background consolidation
        if hasattr(self.hierarchical_memory, 'consolidation_active'):
            self.hierarchical_memory.consolidation_active = False
        
        logger.info(f"Enhanced Integrated Memory System shutdown complete")
        logger.info(f"Session {self.session_id} processed {self.integration_stats['experiences_processed']} experiences")
# ============================================================================
# COMPLETE ENHANCED PERSISTENT IDENTITY AI (MAIN SYSTEM)
# ============================================================================
class EnhancedValidationSystem:
    """Enhanced validation system for persistent identity AI"""
    
    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.validation_history = deque(maxlen=1000)
        self.current_metrics = {
            'identity_coherence_score': 0.5,
            'narrative_consistency_index': 0.5,
            'value_stability_measure': 0.5,
            'overall_validation_score': 0.5
        }
        
        # Validation thresholds
        self.thresholds = {
            'identity_coherence_threshold': 0.8,
            'narrative_consistency_threshold': 0.7,
            'value_stability_threshold': 0.75,
            'overall_threshold': 0.7
        }
        
        self.baseline_personality = None
        self.validation_sessions = []
    
    def validate_experience_processing(self, experience: 'SensorimotorExperience', 
                                     processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experience processing results"""
        
        validation_start = time.time()
        
        # Extract validation metrics
        identity_coherence = self._calculate_identity_coherence_score(processing_result)
        narrative_consistency = self._calculate_narrative_consistency_index(processing_result)
        value_stability = self._calculate_value_stability_measure(processing_result)
        
        # Calculate overall validation score
        overall_score = (identity_coherence + narrative_consistency + value_stability) / 3.0
        
        # Update current metrics
        self.current_metrics.update({
            'identity_coherence_score': identity_coherence,
            'narrative_consistency_index': narrative_consistency,
            'value_stability_measure': value_stability,
            'overall_validation_score': overall_score
        })
        
        # Validation assessment
        validation_passed = self._assess_validation_thresholds(self.current_metrics)
        
        # Record validation
        validation_record = {
            'timestamp': time.time(),
            'experience_id': experience.experience_id,
            'metrics': self.current_metrics.copy(),
            'validation_passed': validation_passed,
            'processing_time': time.time() - validation_start
        }
        
        self.validation_history.append(validation_record)
        
        return {
            'validation_metrics': self.current_metrics,
            'validation_passed': validation_passed,
            'validation_record': validation_record
        }
    
    def _calculate_identity_coherence_score(self, processing_result: Dict[str, Any]) -> float:
        """Calculate Identity Coherence Score (ICS)"""
        
        # Extract identity-related data
        identity_processing = processing_result.get('identity_processing', {})
        personality_state = identity_processing.get('personality_state', {})
        
        if not personality_state:
            return 0.5
        
        # Get current personality traits
        current_traits = personality_state.get('traits_big5', {})
        
        # Compare with baseline if available
        if self.baseline_personality is None:
            self.baseline_personality = current_traits.copy()
            return 1.0  # Perfect coherence for first measurement
        
        # Calculate trait coherence
        trait_coherence_scores = []
        for trait, current_value in current_traits.items():
            baseline_value = self.baseline_personality.get(trait, 0.5)
            coherence = 1.0 - abs(current_value - baseline_value)
            trait_coherence_scores.append(coherence)
        
        return np.mean(trait_coherence_scores) if trait_coherence_scores else 0.5
    
    def _calculate_narrative_consistency_index(self, processing_result: Dict[str, Any]) -> float:
        """Calculate Narrative Consistency Index (NCI)"""
        
        identity_processing = processing_result.get('identity_processing', {})
        narrative_construction = identity_processing.get('narrative_construction', {})
        
        if not narrative_construction:
            return 0.5
        
        # Extract narrative elements
        narrative_elements = narrative_construction.get('narrative_elements', '')
        new_themes = narrative_construction.get('new_themes', [])
        
        # Calculate narrative coherence factors
        coherence_factors = []
        
        # Theme consistency
        if new_themes:
            coherence_factors.append(len(new_themes) / 10.0)  # Normalize theme count
        
        # Narrative length factor
        if narrative_elements:
            narrative_words = len(narrative_elements.split())
            length_factor = min(1.0, narrative_words / 100.0)  # Normalize to 100 words
            coherence_factors.append(length_factor)
        
        # Episodic integration
        episodic_coherence = narrative_construction.get('episodic_coherence_impact', 0.5)
        coherence_factors.append(episodic_coherence)
        
        return np.mean(coherence_factors) if coherence_factors else 0.5
    
    def _calculate_value_stability_measure(self, processing_result: Dict[str, Any]) -> float:
        """Calculate Value Stability Measure (VSM)"""
        
        identity_processing = processing_result.get('identity_processing', {})
        value_evolution = identity_processing.get('value_evolution', {})
        
        if not value_evolution:
            return 0.5
        
        # Extract value evolution data
        value_alignment = value_evolution.get('value_alignment', 0.5)
        updated_values = value_evolution.get('updated_values', {})
        
        # Calculate stability factors
        stability_factors = []
        
        # Value alignment factor
        stability_factors.append(value_alignment)
        
        # Value change magnitude (lower change = higher stability)
        if updated_values:
            # Assume baseline values are around 0.5
            value_changes = [abs(value - 0.5) for value in updated_values.values()]
            avg_change = np.mean(value_changes)
            change_stability = 1.0 - (avg_change * 2.0)  # Normalize and invert
            stability_factors.append(max(0.0, change_stability))
        
        return np.mean(stability_factors) if stability_factors else 0.5
    
    def _assess_validation_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Assess if validation thresholds are met"""
        
        thresholds_met = []
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                thresholds_met.append(metrics[metric_name] >= threshold)
        
        # Require all thresholds to be met
        return all(thresholds_met) if thresholds_met else False
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current validation metrics"""
        return self.current_metrics.copy()
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        
        if not self.validation_history:
            return {'status': 'no_validation_history'}
        
        # Calculate validation trends
        recent_validations = list(self.validation_history)[-10:]
        
        metrics_trends = {}
        for metric_name in self.current_metrics.keys():
            metric_values = [record['metrics'][metric_name] for record in recent_validations 
                           if metric_name in record['metrics']]
            
            if metric_values:
                metrics_trends[metric_name] = {
                    'current': metric_values[-1],
                    'average': np.mean(metric_values),
                    'trend': 'improving' if len(metric_values) > 1 and metric_values[-1] > metric_values[0] else 'stable'
                }
        
        # Validation success rate
        passed_validations = [r for r in recent_validations if r['validation_passed']]
        success_rate = len(passed_validations) / len(recent_validations)
        
        return {
            'total_validations': len(self.validation_history),
            'recent_success_rate': success_rate,
            'current_metrics': self.current_metrics,
            'metrics_trends': metrics_trends,
            'thresholds': self.thresholds
        }


class EnhancedPersistentIdentityAI:
    """Complete Enhanced Persistent Identity AI with ALL original components + memory enhancements"""
    
    def __init__(self, domain: str = "general", personality_seed: Dict[str, float] = None, 
                 model_name: str = "gemma3n:e4b", enable_real_time: bool = True):
        
        # Core system identification
        self.system_id = f"enhanced_pai_{uuid.uuid4().hex[:8]}"
        self.domain = domain
        self.model_name = model_name
        self.creation_timestamp = datetime.now().isoformat()
        
        # Initialize personality seed if not provided
        if personality_seed is None:
            personality_seed = self._generate_default_personality_seed()
        
        # Initialize ALL core components (ORIGINAL + ENHANCED)
        print(f"🧠 Initializing Enhanced Persistent Identity AI...")
        print(f"   Domain: {domain}")
        print(f"   Model: {model_name}")
        print(f"   System ID: {self.system_id}")
        
        # ORIGINAL: Core cognitive architecture
        self.cortical_processor = Enhanced6LayerCorticalProcessor(domain)
        self.identity_processor = EpisodicIdentityProcessor(personality_seed)
        
        # ORIGINAL + ENHANCED: Episodic memory with all enhancements
        self.episodic_memory = EnhancedEpisodicMemoryEngine(model_name)
        
        # NEW: Enhanced memory systems
        self.token_manager = TokenLevelContextManager(
            context_window=self._get_context_window(model_name)
        )
        self.hierarchical_memory = HierarchicalMemorySystem()
        self.compression_system = MemoryCompressionSystem()
        self.cross_modal_system = CrossModalMemorySystem()
        self.boundary_refiner = AdvancedBoundaryRefiner()
        self.advanced_retrieval = AdvancedRetrievalSystem()
        
        # ORIGINAL: Real-time data integration
        self.real_time_integrator = RealTimeDataIntegrator(domain) if enable_real_time else None
        
        # ORIGINAL: LLM integration
        self.llm_integrator = LLMIntegrator(model_name)
        
        # ORIGINAL: Multi-agent coordination
        self.agent_coordinator = MultiAgentCoordinator(self.system_id)
        
        # ORIGINAL: Identity formation components
        self.narrator = ContinuousNarrator()
        self.identity_comparer = IdentityComparer()
        self.temporal_integrator = TemporalIntegrator()
        self.meaning_maker = MeaningMaker()
        
        # System state tracking
        self.experience_count = 0
        self.total_processing_time = 0.0
        self.system_metrics = {
            'identity_coherence_scores': deque(maxlen=100),
            'narrative_consistency_scores': deque(maxlen=100),
            'processing_efficiency': deque(maxlen=100),
            'memory_utilization': deque(maxlen=100)
        }
        
        # Integration and coordination
        self._setup_component_integration()
        
        # Enhanced validation system
        self.validator = EnhancedValidationSystem(self)
        
        print(f"✅ Enhanced Persistent Identity AI initialized successfully!")
        print(f"   Components: {len(self._get_all_components())} active")
        print(f"   Memory systems: 6 integrated")
        print(f"   Real-time: {'Enabled' if enable_real_time else 'Disabled'}")
        
    def _generate_default_personality_seed(self) -> Dict[str, float]:
        """Generate default personality seed"""
        return {
            'openness': random.uniform(0.4, 0.8),
            'conscientiousness': random.uniform(0.5, 0.9),
            'extraversion': random.uniform(0.3, 0.7),
            'agreeableness': random.uniform(0.5, 0.8),
            'neuroticism': random.uniform(0.2, 0.6),
            'curiosity': random.uniform(0.6, 0.9),
            'analytical_thinking': random.uniform(0.5, 0.9),
            'creativity': random.uniform(0.4, 0.8),
            'adaptability': random.uniform(0.5, 0.8)
        }
    
    def _get_context_window(self, model_name: str) -> int:
        """Get context window size for model"""
        if "gemma3n" in model_name:
            return 32000
        elif "deepseek" in model_name:
            return 4000
        elif "qwen" in model_name:
            return 8000
        else:
            return 8000
    
    def _setup_component_integration(self):
        """Setup integration between all components"""
        
        # Set episodic memory engine for components
        self.cortical_processor.set_episodic_memory_engine(self.episodic_memory)
        self.identity_processor.set_episodic_memory_engine(self.episodic_memory)
        
        # Set cortical processor for episodic memory
        self.episodic_memory.cortical_processor = self.cortical_processor
        
        # Set identity processor for other components
        self.narrator.set_identity_processor(self.identity_processor)
        self.temporal_integrator.set_identity_processor(self.identity_processor)
        self.meaning_maker.set_identity_processor(self.identity_processor)
        
        # Set enhanced memory systems
        self.episodic_memory.token_manager = self.token_manager
        self.episodic_memory.hierarchical_memory = self.hierarchical_memory
        self.episodic_memory.compression_system = self.compression_system
        self.episodic_memory.cross_modal_system = self.cross_modal_system
        self.episodic_memory.boundary_refiner = self.boundary_refiner
        self.episodic_memory.advanced_retrieval = self.advanced_retrieval
        
        # Setup LLM integration
        if hasattr(self, 'llm_integrator'):
            self.llm_integrator.set_episodic_memory(self.episodic_memory)
            self.llm_integrator.set_identity_processor(self.identity_processor)
        
        # Setup agent coordination
        if hasattr(self, 'agent_coordinator'):
            self.agent_coordinator.register_component('episodic_memory', self.episodic_memory)
            self.agent_coordinator.register_component('cortical_processor', self.cortical_processor)
            self.agent_coordinator.register_component('identity_processor', self.identity_processor)
    
    def process_experience(self, content: str, domain: str = None, 
                         real_time_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete experience processing with ALL systems"""
        
        start_time = time.time()
        
        if domain is None:
            domain = self.domain
        
        # Create enhanced experience
        experience = self._create_enhanced_experience(content, domain, real_time_data)
        
        print(f"\n🔄 Processing Experience {self.experience_count + 1}")
        print(f"   Content: {content[:60]}...")
        print(f"   Domain: {domain}")
        print(f"   Novelty: {experience.novelty_score:.3f}")
        
        # STAGE 1: Enhanced memory integration
        print("   📝 Stage 1: Enhanced Memory Integration...")
        memory_start = time.time()
        
        # Token-level processing
        experience_tokens = content.split()
        token_context = self.token_manager.process_tokens_with_memory(
            experience_tokens,
            experience_context={'content': content, 'domain': domain}
        )
        
        # Hierarchical memory storage
        hierarchical_result = self.hierarchical_memory.store_experience(experience)
        
        # Cross-modal processing
        cross_modal_result = self.cross_modal_system.store_cross_modal_experience(experience)
        
        memory_time = time.time() - memory_start
        print(f"      ⏱️  Memory processing: {memory_time:.3f}s")
        
        # STAGE 2: Cortical processing with episodic integration
        print("   🧠 Stage 2: Cortical Processing...")
        cortical_start = time.time()
        
        cortical_result = self.cortical_processor.process_experience_with_episodes(experience)
        
        cortical_time = time.time() - cortical_start
        print(f"      ⏱️  Cortical processing: {cortical_time:.3f}s")
        print(f"      🎯 Prediction accuracy: {cortical_result['prediction_accuracy']:.3f}")
        print(f"      🔬 Domain expertise: {cortical_result['domain_expertise_level']:.3f}")
        
        # STAGE 3: Episode boundary detection
        print("   🚧 Stage 3: Episode Boundary Detection...")
        boundary_start = time.time()
        
        is_boundary = self.episodic_memory.detect_episode_boundary(experience, cortical_result)
        
        boundary_time = time.time() - boundary_start
        if is_boundary:
            print(f"      🔥 BOUNDARY DETECTED! Processing time: {boundary_time:.3f}s")
        else:
            print(f"      ➡️  Continuing episode. Processing time: {boundary_time:.3f}s")
        
        # STAGE 4: Identity formation with episodic integration
        print("   👤 Stage 4: Identity Formation...")
        identity_start = time.time()
        
        identity_result = self.identity_processor.process_experience_with_episodes(experience, cortical_result)
        
        identity_time = time.time() - identity_start
        coherence = identity_result['coherence_assessment']['overall_coherence']
        print(f"      ⏱️  Identity processing: {identity_time:.3f}s")
        print(f"      🎭 Identity coherence: {coherence:.3f}")
        
        # STAGE 5: Enhanced episodic storage
        print("   💾 Stage 5: Enhanced Episodic Storage...")
        storage_start = time.time()
        
        self.episodic_memory.store_episode(experience, cortical_result, identity_result, is_boundary)
        
        storage_time = time.time() - storage_start
        print(f"      ⏱️  Storage time: {storage_time:.3f}s")
        
        # STAGE 6: LLM integration (if available)
        llm_result = {}
        if hasattr(self, 'llm_integrator'):
            print("   🤖 Stage 6: LLM Integration...")
            llm_start = time.time()
            
            llm_result = self.llm_integrator.process_with_identity_context(experience, cortical_result, identity_result)
            
            llm_time = time.time() - llm_start
            print(f"      ⏱️  LLM processing: {llm_time:.3f}s")
        
        # STAGE 7: Multi-agent coordination (if available)
        coordination_result = {}
        if hasattr(self, 'agent_coordinator'):
            print("   🤝 Stage 7: Multi-Agent Coordination...")
            coord_start = time.time()
            
            coordination_result = self.agent_coordinator.coordinate_processing(experience, cortical_result, identity_result)
            
            coord_time = time.time() - coord_start
            print(f"      ⏱️  Coordination time: {coord_time:.3f}s")
        
        # STAGE 8: Real-time data integration (if enabled)
        real_time_result = {}
        if self.real_time_integrator:
            print("   📡 Stage 8: Real-Time Data Integration...")
            rt_start = time.time()
            
            real_time_result = self.real_time_integrator.integrate_real_time_context(experience, cortical_result)
            
            rt_time = time.time() - rt_start
            print(f"      ⏱️  Real-time processing: {rt_time:.3f}s")
        
        # Calculate final metrics
        total_time = time.time() - start_time
        self.total_processing_time += total_time
        self.experience_count += 1
        
        # Update system metrics
        self._update_system_metrics(cortical_result, identity_result, total_time)
        
        # Comprehensive result
        result = {
            'experience_id': experience.experience_id,
            'experience_count': self.experience_count,
            'processing_time': total_time,
            'is_boundary': is_boundary,
            
            # Core processing results
            'memory_integration': {
                'token_context_size': len(token_context),
                'hierarchical_storage': hierarchical_result,
                'cross_modal_associations': cross_modal_result.get('cross_modal_associations', 0),
                'processing_time': memory_time
            },
            'cortical_result': cortical_result,
            'identity_result': identity_result,
            'llm_result': llm_result,
            'coordination_result': coordination_result,
            'real_time_result': real_time_result,
            
            # Enhanced features
            'enhanced_features': {
                'token_management_active': True,
                'hierarchical_memory_active': True,
                'cross_modal_integration': True,
                'boundary_refinement': True,
                'advanced_retrieval': True,
                'compression_available': True
            },
            
            # System state
            'system_state': self._get_system_state(),
            'validation_metrics': self._get_validation_metrics()
        }
        
        print(f"✅ Experience processed successfully!")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Identity coherence: {coherence:.3f}")
        print(f"   Enhanced features: {len(result['enhanced_features'])} active")
        
        return result
    
    def _create_enhanced_experience(self, content: str, domain: str, 
                                  real_time_data: Dict[str, Any] = None) -> SensorimotorExperience:
        """Create enhanced sensorimotor experience"""
        
        # Extract sensory features
        sensory_features = self._extract_sensory_features(content, domain)
        
        # Extract motor actions
        motor_actions = self._extract_motor_actions(content, domain)
        
        # Generate contextual embedding
        contextual_embedding = self._generate_contextual_embedding(content, domain)
        
        # Extract temporal markers
        temporal_markers = [time.time()]
        
        # Calculate attention weights
        attention_weights = self._calculate_attention_weights(content, domain)
        
        # Generate prediction targets
        prediction_targets = self._generate_prediction_targets(content, domain)
        
        # Calculate novelty score
        novelty_score = self._calculate_novelty_score(content, domain)
        
        # Enhanced features
        emotional_features = self._extract_emotional_features(content)
        causal_indicators = self._extract_causal_indicators(content)
        goal_relevance = self._extract_goal_relevance(content, domain)
        modality_features = self._extract_modality_features(content)
        
        # Real-time data integration
        if real_time_data:
            sensory_features.update(real_time_data.get('sensory_additions', {}))
            contextual_embedding = self._enhance_embedding_with_real_time(contextual_embedding, real_time_data)
        
        return SensorimotorExperience(
            experience_id=f"exp_{uuid.uuid4().hex[:8]}",
            content=content,
            domain=domain,
            sensory_features=sensory_features,
            motor_actions=motor_actions,
            contextual_embedding=contextual_embedding,
            temporal_markers=temporal_markers,
            attention_weights=attention_weights,
            prediction_targets=prediction_targets,
            novelty_score=novelty_score,
            timestamp=datetime.now().isoformat(),
            
            # Enhanced features
            emotional_features=emotional_features,
            causal_indicators=causal_indicators,
            goal_relevance=goal_relevance,
            modality_features=modality_features,
            importance_weight=self._calculate_importance_weight(content, domain),
            access_frequency=0,
            last_access=time.time(),
            memory_strength=1.0
        )
    
    def _extract_sensory_features(self, content: str, domain: str) -> Dict[str, Any]:
        """Extract sensory features from content"""
        features = {}
        
        # Text-based sensory features
        features['word_count'] = len(content.split())
        features['char_count'] = len(content)
        features['complexity'] = len(set(content.lower().split())) / max(len(content.split()), 1)
        features['sentiment_polarity'] = self._calculate_sentiment_polarity(content)
        
        # Domain-specific features
        if domain == 'financial_analysis':
            features['financial_indicators'] = self._extract_financial_indicators(content)
        elif domain == 'research':
            features['research_indicators'] = self._extract_research_indicators(content)
        
        return features
    
    def _extract_motor_actions(self, content: str, domain: str) -> List[str]:
        """Extract motor actions from content"""
        actions = []
        
        # Action words
        action_words = ['analyze', 'evaluate', 'assess', 'compare', 'examine', 'investigate', 'explore']
        content_words = content.lower().split()
        
        for action in action_words:
            if action in content_words:
                actions.append(action)
        
        # Domain-specific actions
        if domain == 'financial_analysis':
            financial_actions = ['buy', 'sell', 'hold', 'invest', 'trade', 'hedge']
            for action in financial_actions:
                if action in content_words:
                    actions.append(f'financial_{action}')
        
        return actions
    
    def _generate_contextual_embedding(self, content: str, domain: str) -> np.ndarray:
        """Generate contextual embedding for content"""
        
        # Simple embedding based on content features
        words = content.lower().split()
        
        # Create feature vector
        features = []
        
        # Length features
        features.append(len(words) / 100.0)  # Normalized word count
        features.append(len(content) / 500.0)  # Normalized char count
        
        # Lexical diversity
        unique_words = len(set(words))
        features.append(unique_words / max(len(words), 1))
        
        # Domain encoding
        domain_encoding = {'financial_analysis': [1, 0, 0], 'research': [0, 1, 0], 'general': [0, 0, 1]}
        features.extend(domain_encoding.get(domain, [0, 0, 1]))
        
        # Content category features
        categories = {
            'analytical': ['analysis', 'data', 'study', 'research', 'examine'],
            'creative': ['creative', 'innovative', 'novel', 'original', 'artistic'],
            'social': ['team', 'collaborate', 'community', 'social', 'together'],
            'technical': ['system', 'algorithm', 'technical', 'method', 'process']
        }
        
        for category, keywords in categories.items():
            score = sum(1 for word in keywords if word in words) / len(keywords)
            features.append(score)
        
        # Pad or truncate to fixed size
        target_size = 20
        while len(features) < target_size:
            features.append(random.uniform(0, 0.1))
        
        return np.array(features[:target_size])
    
    def _calculate_attention_weights(self, content: str, domain: str) -> Dict[str, float]:
        """Calculate attention weights for different aspects"""
        
        weights = {}
        words = content.lower().split()
        
        # Content attention
        important_words = ['important', 'critical', 'key', 'significant', 'major', 'primary']
        weights['content'] = min(1.0, sum(1 for word in important_words if word in words) * 0.2 + 0.5)
        
        # Novelty attention
        novel_words = ['new', 'novel', 'innovative', 'breakthrough', 'unprecedented']
        weights['novelty'] = min(1.0, sum(1 for word in novel_words if word in words) * 0.3 + 0.3)
        
        # Emotional attention
        emotional_words = ['feel', 'emotion', 'excited', 'concerned', 'passionate', 'worried']
        weights['emotion'] = min(1.0, sum(1 for word in emotional_words if word in words) * 0.25 + 0.2)
        
        # Domain attention
        if domain == 'financial_analysis':
            financial_words = ['market', 'price', 'investment', 'trading', 'financial']
            weights['domain'] = min(1.0, sum(1 for word in financial_words if word in words) * 0.2 + 0.4)
        else:
            weights['domain'] = 0.5
        
        return weights
    
    def _generate_prediction_targets(self, content: str, domain: str) -> Dict[str, float]:
        """Generate prediction targets"""
        
        targets = {}
        words = content.lower().split()
        
        # Next content prediction
        complexity = len(set(words)) / max(len(words), 1)
        targets['next_content_complexity'] = complexity
        
        # Domain continuation
        targets['domain_consistency'] = 0.8 if domain in ['financial_analysis', 'research'] else 0.6
        
        # Novelty prediction
        novelty_indicators = ['new', 'different', 'change', 'evolve', 'develop']
        novelty_score = sum(1 for word in novelty_indicators if word in words) / len(novelty_indicators)
        targets['future_novelty'] = min(1.0, novelty_score + 0.3)
        
        return targets
    
    def _calculate_novelty_score(self, content: str, domain: str) -> float:
        """Calculate novelty score for content"""
        
        # Base novelty on content characteristics
        words = content.lower().split()
        
        # Novelty indicators
        novelty_words = ['new', 'novel', 'innovative', 'breakthrough', 'unprecedented', 'unique', 'different']
        novelty_count = sum(1 for word in novelty_words if word in words)
        
        # Complexity contribution
        complexity = len(set(words)) / max(len(words), 1)
        
        # Length contribution (longer content may be more novel)
        length_factor = min(1.0, len(words) / 50.0)
        
        # Domain-specific adjustments
        domain_factor = 1.2 if domain == 'research' else 1.0
        
        novelty_score = (
            (novelty_count / len(novelty_words)) * 0.4 +
            complexity * 0.3 +
            length_factor * 0.2 +
            random.uniform(0.1, 0.2)  # Random component
        ) * domain_factor
        
        return min(1.0, novelty_score)
    
    def _extract_emotional_features(self, content: str) -> Dict[str, float]:
        """Extract emotional features from content"""
        
        words = content.lower().split()
        emotional_features = {}
        
        emotion_categories = {
            'positive': ['happy', 'excited', 'optimistic', 'confident', 'pleased', 'satisfied'],
            'negative': ['sad', 'worried', 'pessimistic', 'concerned', 'disappointed', 'frustrated'],
            'arousal': ['excited', 'energetic', 'intense', 'passionate', 'dynamic', 'active'],
            'valence': ['good', 'positive', 'excellent', 'wonderful', 'great', 'amazing'],
            'dominance': ['control', 'powerful', 'strong', 'confident', 'assertive', 'influential']
        }
        
        for emotion, keywords in emotion_categories.items():
            score = sum(1 for word in keywords if word in words) / max(len(keywords), 1)
            emotional_features[emotion] = score
        
        return emotional_features
    
    def _extract_causal_indicators(self, content: str) -> List[str]:
        """Extract causal indicators from content"""
        
        causal_words = ['because', 'since', 'due to', 'caused by', 'leads to', 'results in', 'triggers', 'influences']
        content_lower = content.lower()
        
        indicators = []
        for word in causal_words:
            if word in content_lower:
                indicators.append(word)
        
        return indicators
    
    def _extract_goal_relevance(self, content: str, domain: str) -> Dict[str, float]:
        """Extract goal relevance from content"""
        
        words = content.lower().split()
        goal_relevance = {}
        
        goal_categories = {
            'learning': ['learn', 'understand', 'knowledge', 'education', 'study', 'discover'],
            'achievement': ['achieve', 'accomplish', 'succeed', 'complete', 'finish', 'goal'],
            'analysis': ['analyze', 'examine', 'evaluate', 'assess', 'investigate', 'research'],
            'improvement': ['improve', 'enhance', 'optimize', 'better', 'upgrade', 'develop'],
            'innovation': ['innovate', 'create', 'invent', 'design', 'novel', 'breakthrough']
        }
        
        for goal, keywords in goal_categories.items():
            score = sum(1 for word in keywords if word in words) / max(len(keywords), 1)
            goal_relevance[goal] = score
        
        return goal_relevance
    
    def _extract_modality_features(self, content: str) -> Dict[str, np.ndarray]:
        """Extract modality-specific features"""
        
        modality_features = {}
        
        # Text modality (primary)
        text_features = self._generate_contextual_embedding(content, 'general')
        modality_features['text'] = text_features
        
        # Temporal modality
        temporal_features = np.array([
            time.time() % 86400 / 86400,  # Time of day
            datetime.now().weekday() / 6,  # Day of week
            len(content.split()) / 100,    # Content length temporal aspect
        ])
        modality_features['temporal'] = temporal_features
        
        # Spatial modality (conceptual)
        spatial_features = np.array([
            len(content) / 1000,           # Content size
            len(set(content.lower().split())) / 100,  # Conceptual space size
            content.count('.') / 10,       # Sentence density
        ])
        modality_features['spatial'] = spatial_features
        
        return modality_features
    
    def _calculate_sentiment_polarity(self, content: str) -> float:
        """Calculate sentiment polarity"""
        
        positive_words = ['good', 'great', 'excellent', 'positive', 'amazing', 'wonderful', 'fantastic', 'outstanding']
        negative_words = ['bad', 'poor', 'terrible', 'negative', 'awful', 'horrible', 'disappointing', 'problematic']
        
        words = content.lower().split()
        
        positive_count = sum(1 for word in positive_words if word in words)
        negative_count = sum(1 for word in negative_words if word in words)
        
        total_sentiment = positive_count + negative_count
        
        if total_sentiment == 0:
            return 0.5  # Neutral
        
        return positive_count / total_sentiment
    
    def _extract_financial_indicators(self, content: str) -> Dict[str, float]:
        """Extract financial indicators from content"""
        
        indicators = {}
        words = content.lower().split()
        
        financial_categories = {
            'bullish': ['bull', 'bullish', 'rise', 'gain', 'increase', 'growth', 'rally'],
            'bearish': ['bear', 'bearish', 'fall', 'decline', 'decrease', 'drop', 'crash'],
            'volatility': ['volatile', 'volatility', 'swing', 'fluctuation', 'unstable'],
            'volume': ['volume', 'trading', 'activity', 'liquidity', 'flow']
        }
        
        for category, keywords in financial_categories.items():
            score = sum(1 for word in keywords if word in words) / max(len(keywords), 1)
            indicators[category] = score
        
        return indicators
    
    def _extract_research_indicators(self, content: str) -> Dict[str, float]:
        """Extract research indicators from content"""
        
        indicators = {}
        words = content.lower().split()
        
        research_categories = {
            'methodology': ['method', 'methodology', 'approach', 'technique', 'procedure'],
            'analysis': ['analysis', 'analyze', 'examine', 'investigate', 'study'],
            'findings': ['finding', 'result', 'outcome', 'conclusion', 'evidence'],
            'theory': ['theory', 'theoretical', 'concept', 'principle', 'model']
        }
        
        for category, keywords in research_categories.items():
            score = sum(1 for word in keywords if word in words) / max(len(keywords), 1)
            indicators[category] = score
        
        return indicators
    
    def _calculate_importance_weight(self, content: str, domain: str) -> float:
        """Calculate importance weight for content"""
        
        # Base importance on content characteristics
        words = content.lower().split()
        
        # Important indicators
        importance_words = ['important', 'critical', 'significant', 'major', 'key', 'essential', 'vital']
        importance_count = sum(1 for word in importance_words if word in words)
        
        # Length factor
        length_factor = min(1.0, len(words) / 30.0)
        
        # Domain factor
        domain_factor = 1.2 if domain in ['financial_analysis', 'research'] else 1.0
        
        # Novelty factor
        novelty_words = ['new', 'novel', 'breakthrough', 'innovation']
        novelty_factor = 1 + sum(1 for word in novelty_words if word in words) * 0.1
        
        importance = (
            (importance_count / len(importance_words)) * 0.4 +
            length_factor * 0.3 +
            novelty_factor * 0.3
        ) * domain_factor
        
        return min(1.0, max(0.1, importance))
    
    def _enhance_embedding_with_real_time(self, embedding: np.ndarray, 
                                        real_time_data: Dict[str, Any]) -> np.ndarray:
        """Enhance embedding with real-time data"""
        
        if not real_time_data:
            return embedding
        
        # Add real-time factors
        enhanced = embedding.copy()
        
        # Market data influence
        if 'market_data' in real_time_data:
            market_influence = real_time_data['market_data'].get('sentiment', 0.5)
            enhanced = enhanced * (0.9 + market_influence * 0.2)
        
        # News influence
        if 'news_sentiment' in real_time_data:
            news_influence = real_time_data['news_sentiment']
            enhanced = enhanced * (0.95 + news_influence * 0.1)
        
        return enhanced
    
    def _update_system_metrics(self, cortical_result: Dict[str, Any], 
                             identity_result: Dict[str, Any], 
                             processing_time: float):
        """Update system-wide metrics"""
        
        # Identity coherence
        coherence = identity_result.get('coherence_assessment', {}).get('overall_coherence', 0.5)
        self.system_metrics['identity_coherence_scores'].append(coherence)
        
        # Narrative consistency (derived from coherence assessment)
        narrative_coherence = identity_result.get('coherence_assessment', {}).get('narrative_coherence', 0.5)
        self.system_metrics['narrative_consistency_scores'].append(narrative_coherence)
        
        # Processing efficiency (experiences per second)
        efficiency = 1.0 / processing_time if processing_time > 0 else 0
        self.system_metrics['processing_efficiency'].append(efficiency)
        
        # Memory utilization
        memory_stats = self._get_memory_utilization()
        self.system_metrics['memory_utilization'].append(memory_stats)
    
    def _get_memory_utilization(self) -> float:
        """Calculate current memory utilization"""
        
        utilization_factors = []
        
        # Token manager utilization
        token_stats = self.token_manager.get_memory_statistics()
        utilization_factors.append(token_stats.get('utilization_ratio', 0.5))
        
        # Hierarchical memory utilization
        hierarchical_stats = self.hierarchical_memory.get_memory_statistics()
        working_util = hierarchical_stats['working_memory']['utilization']
        short_term_util = hierarchical_stats['short_term_memory']['utilization']
        utilization_factors.extend([working_util, short_term_util])
        
        # Episodic memory utilization
        episodic_stats = self.episodic_memory.get_enhanced_memory_statistics()
        episode_count = episodic_stats['total_episodes']
        max_episodes = self.episodic_memory.max_episodes
        episodic_util = episode_count / max_episodes
        utilization_factors.append(episodic_util)
        
        return np.mean(utilization_factors)
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        
        return {
            'experience_count': self.experience_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': self.total_processing_time / max(self.experience_count, 1),
            'current_identity_coherence': self.system_metrics['identity_coherence_scores'][-1] if self.system_metrics['identity_coherence_scores'] else 0.5,
            'memory_utilization': self._get_memory_utilization(),
            'active_components': len(self._get_all_components()),
            'system_uptime': (datetime.now() - datetime.fromisoformat(self.creation_timestamp)).total_seconds(),
            'domain': self.domain,
            'model_name': self.model_name
        }
    
    def _get_validation_metrics(self) -> Dict[str, float]:
        """Get current validation metrics"""
        
        if not hasattr(self, 'validator'):
            return {}
        
        return self.validator.get_current_metrics()
    
    def _get_all_components(self) -> List[str]:
        """Get list of all active components"""
        
        components = [
            'cortical_processor', 'identity_processor', 'episodic_memory',
            'token_manager', 'hierarchical_memory', 'compression_system',
            'cross_modal_system', 'boundary_refiner', 'advanced_retrieval'
        ]
        
        if hasattr(self, 'real_time_integrator') and self.real_time_integrator:
            components.append('real_time_integrator')
        
        if hasattr(self, 'llm_integrator'):
            components.append('llm_integrator')
        
        if hasattr(self, 'agent_coordinator'):
            components.append('agent_coordinator')
        
        return components
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        # Component statistics
        component_stats = {
            'cortical_processor': self.cortical_processor.get_processing_statistics(),
            'episodic_memory': self.episodic_memory.get_enhanced_memory_statistics(),
            'token_manager': self.token_manager.get_memory_statistics(),
            'hierarchical_memory': self.hierarchical_memory.get_memory_statistics(),
            'cross_modal_system': self.cross_modal_system.get_cross_modal_statistics(),
            'advanced_retrieval': self.advanced_retrieval.get_retrieval_statistics()
        }
        
        # System metrics
        system_metrics = {}
        for metric_name, metric_values in self.system_metrics.items():
            if metric_values:
                system_metrics[metric_name] = {
                    'current': metric_values[-1],
                    'average': np.mean(metric_values),
                    'trend': 'improving' if len(metric_values) > 5 and metric_values[-1] > np.mean(metric_values[-5:]) else 'stable',
                    'count': len(metric_values)
                }
        
        # Identity metrics
        personality_state = self.identity_processor.current_personality_state
        identity_metrics = {
            'narrative_coherence': personality_state.narrative_coherence,
            'identity_stability': personality_state.identity_stability,
            'episodic_narrative_depth': personality_state.episodic_narrative_depth,
            'cross_episodic_coherence': personality_state.cross_episodic_coherence,
            'development_stage': personality_state.development_stage,
            'identity_milestones': len(personality_state.episodic_identity_milestones)
        }
        
        return {
            'system_info': {
                'system_id': self.system_id,
                'domain': self.domain,
                'model_name': self.model_name,
                'creation_timestamp': self.creation_timestamp,
                'experience_count': self.experience_count
            },
            'system_metrics': system_metrics,
            'identity_metrics': identity_metrics,
            'component_statistics': component_stats,
            'system_state': self._get_system_state(),
            'validation_metrics': self._get_validation_metrics()
        }
    
    def demonstrate_advanced_features(self) -> Dict[str, Any]:
        """Demonstrate all advanced features of the system"""
        
        print("\n🚀 Demonstrating Enhanced Persistent Identity AI Features")
        print("=" * 70)
        
        demo_results = {}
        
        # Feature 1: Token-level context management
        print("\n📝 Feature 1: Token-Level Context Management")
        token_stats = self.token_manager.get_memory_statistics()
        print(f"   Context window: {token_stats['context_window_size']:,} tokens")
        print(f"   Current utilization: {token_stats['utilization_ratio']:.1%}")
        print(f"   Evicted tokens: {token_stats['evicted_tokens']}")
        demo_results['token_management'] = token_stats
        
        # Feature 2: Hierarchical memory
        print("\n🏛️ Feature 2: Hierarchical Memory System")
        hierarchical_stats = self.hierarchical_memory.get_memory_statistics()
        print(f"   Working memory: {hierarchical_stats['working_memory']['count']}/7")
        print(f"   Short-term memory: {hierarchical_stats['short_term_memory']['count']}/50")
        print(f"   Long-term memory: {hierarchical_stats['long_term_memory']['count']}")
        print(f"   Semantic concepts: {hierarchical_stats['semantic_memory']['concepts']}")
        demo_results['hierarchical_memory'] = hierarchical_stats
        
        # Feature 3: Cross-modal integration
        print("\n🌐 Feature 3: Cross-Modal Integration")
        cross_modal_stats = self.cross_modal_system.get_cross_modal_statistics()
        print(f"   Supported modalities: {len(cross_modal_stats['supported_modalities'])}")
        total_experiences = sum(stats['stored_experiences'] for stats in cross_modal_stats['modal_index_stats'].values())
        print(f"   Total cross-modal experiences: {total_experiences}")
        demo_results['cross_modal'] = cross_modal_stats
        
        # Feature 4: Advanced retrieval
        print("\n🔍 Feature 4: Advanced Multi-Strategy Retrieval")
        retrieval_stats = self.advanced_retrieval.get_retrieval_statistics()
        if retrieval_stats.get('total_retrievals', 0) > 0:
            print(f"   Total retrievals: {retrieval_stats['total_retrievals']}")
            print(f"   Ensemble effectiveness: {retrieval_stats['ensemble_effectiveness']:.3f}")
        else:
            print("   No retrievals performed yet")
        demo_results['advanced_retrieval'] = retrieval_stats
        
        # Feature 5: Identity coherence tracking
        print("\n👤 Feature 5: Identity Coherence Tracking")
        if self.system_metrics['identity_coherence_scores']:
            current_coherence = self.system_metrics['identity_coherence_scores'][-1]
            avg_coherence = np.mean(self.system_metrics['identity_coherence_scores'])
            print(f"   Current coherence: {current_coherence:.3f}")
            print(f"   Average coherence: {avg_coherence:.3f}")
            print(f"   Coherence samples: {len(self.system_metrics['identity_coherence_scores'])}")
        else:
            print("   No coherence data available yet")
        demo_results['identity_coherence'] = {
            'current': self.system_metrics['identity_coherence_scores'][-1] if self.system_metrics['identity_coherence_scores'] else 0,
            'history_length': len(self.system_metrics['identity_coherence_scores'])
        }
        
        # Feature 6: Episodic memory enhancement
        print("\n💾 Feature 6: Enhanced Episodic Memory")
        episodic_stats = self.episodic_memory.get_enhanced_memory_statistics()
        enhanced_features = episodic_stats.get('enhanced_features', {})
        print(f"   Total episodes: {episodic_stats['total_episodes']}")
        print(f"   Episode boundaries: {episodic_stats['episode_boundaries']}")
        print(f"   Memory span: {episodic_stats['memory_span_days']:.1f} days")
        print(f"   Enhanced features active: {len(enhanced_features)}")
        demo_results['episodic_memory'] = episodic_stats
        
        print(f"\n✅ Advanced features demonstration complete!")
        print(f"   Total features demonstrated: {len(demo_results)}")
        
        return demo_results

# ============================================================================
# LLM INTEGRATION CLASSES (COMPLETE ORIGINAL)
# ============================================================================

class LLMIntegrator:
    """Complete LLM integration with identity-aware processing"""
    
    def __init__(self, model_name: str = "gemma3n:e4b"):
        self.model_name = model_name
        self.episodic_memory = None
        self.identity_processor = None
        self.conversation_history = deque(maxlen=50)
        self.context_templates = self._initialize_context_templates()
        self.response_cache = {}
        
    def set_episodic_memory(self, episodic_memory):
        """Set episodic memory engine"""
        self.episodic_memory = episodic_memory
    
    def set_identity_processor(self, identity_processor):
        """Set identity processor"""
        self.identity_processor = identity_processor
    
    def _initialize_context_templates(self) -> Dict[str, str]:
        """Initialize context templates for different scenarios"""
        
        return {
            'identity_aware': """
You are an AI with a persistent identity. Here's your current state:

Personality Traits:
{personality_summary}

Recent Experiences:
{episodic_context}

Current Coherence: {coherence_score}

Based on this context, respond to: {user_input}

Maintain consistency with your established identity while being helpful and informative.
""",
            
            'analysis_focused': """
Based on your cognitive processing and identity:

Cortical Analysis:
{cortical_summary}

Identity Perspective:
{identity_perspective}

Episodic Context:
{episodic_context}

Analyze and respond to: {user_input}

Provide a thoughtful analysis that reflects your established cognitive patterns.
""",
            
            'narrative_continuity': """
Continuing your ongoing narrative:

Previous Themes: {narrative_themes}
Identity Development: {identity_development}
Recent Insights: {recent_insights}

Current situation: {user_input}

Respond in a way that maintains narrative continuity and shows identity growth.
"""
        }
    
    def process_with_identity_context(self, experience: SensorimotorExperience, 
                                    cortical_result: Dict[str, Any], 
                                    identity_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process experience with full identity context for LLM integration"""
        
        # Extract relevant context
        personality_summary = self._create_personality_summary(identity_result)
        episodic_context = self._create_episodic_context(experience)
        cortical_summary = self._create_cortical_summary(cortical_result)
        
        # Determine appropriate template
        template_type = self._select_template_type(experience, cortical_result, identity_result)
        
        # Generate context-aware prompt
        prompt = self._generate_identity_aware_prompt(
            experience.content, template_type, personality_summary, 
            episodic_context, cortical_summary, identity_result
        )
        
        # Simulate LLM processing (in real implementation, this would call actual LLM)
        llm_response = self._simulate_llm_response(prompt, experience, identity_result)
        
        # Update conversation history
        self._update_conversation_history(experience.content, llm_response, identity_result)
        
        return {
            'llm_response': llm_response,
            'template_used': template_type,
            'prompt_length': len(prompt),
            'context_elements': {
                'personality_included': bool(personality_summary),
                'episodic_context_included': bool(episodic_context),
                'cortical_analysis_included': bool(cortical_summary)
            },
            'response_metadata': {
                'coherence_maintained': self._assess_response_coherence(llm_response, identity_result),
                'identity_consistency': self._assess_identity_consistency(llm_response, identity_result),
                'narrative_continuity': self._assess_narrative_continuity(llm_response, identity_result)
            }
        }
    
    def _create_personality_summary(self, identity_result: Dict[str, Any]) -> str:
        """Create concise personality summary"""
        
        personality_state = identity_result.get('personality_state', {})
        
        # Extract key traits
        traits = personality_state.get('traits_big5', {})
        values = personality_state.get('core_value_system', {})
        themes = personality_state.get('narrative_themes', [])
        
        summary_parts = []
        
        # Big Five summary
        if traits:
            trait_descriptions = []
            for trait, value in traits.items():
                if value > 0.7:
                    trait_descriptions.append(f"high {trait}")
                elif value < 0.3:
                    trait_descriptions.append(f"low {trait}")
            
            if trait_descriptions:
                summary_parts.append(f"Traits: {', '.join(trait_descriptions[:3])}")
        
        # Core values
        if values:
            top_values = sorted(values.items(), key=lambda x: x[1], reverse=True)[:3]
            value_names = [name for name, _ in top_values if _ > 0.5]
            if value_names:
                summary_parts.append(f"Values: {', '.join(value_names)}")
        
        # Narrative themes
        if themes:
            summary_parts.append(f"Themes: {', '.join(themes[:3])}")
        
        return " | ".join(summary_parts) if summary_parts else "Developing identity"
    
    def _create_episodic_context(self, experience: SensorimotorExperience) -> str:
        """Create episodic context summary"""
        
        if not self.episodic_memory:
            return "No episodic context available"
        
        # Retrieve relevant episodes
        retrieval_result = self.episodic_memory.retrieve_episodic_context(experience, max_context_tokens=1000)
        episodes = retrieval_result.get('episodes', [])
        
        if not episodes:
            return "No relevant episodic memories"
        
        # Create context summary
        context_parts = []
        for i, episode in enumerate(episodes[:3]):  # Top 3 episodes
            content_preview = episode.get('content', '')[:80] + "..." if len(episode.get('content', '')) > 80 else episode.get('content', '')
            timestamp = episode.get('timestamp', '')[:10]  # Date only
            similarity = episode.get('similarity_score', 0)
            
            context_parts.append(f"{i+1}. [{timestamp}] {content_preview} (relevance: {similarity:.2f})")
        
        return "\n".join(context_parts)
    
    def _create_cortical_summary(self, cortical_result: Dict[str, Any]) -> str:
        """Create cortical processing summary"""
        
        summary_parts = []
        
        # Prediction accuracy
        pred_accuracy = cortical_result.get('prediction_accuracy', 0.5)
        summary_parts.append(f"Prediction accuracy: {pred_accuracy:.2f}")
        
        # Domain expertise
        domain_expertise = cortical_result.get('domain_expertise_level', 0.5)
        summary_parts.append(f"Domain expertise: {domain_expertise:.2f}")
        
        # Consensus information
        consensus = cortical_result.get('consensus', {})
        confidence = consensus.get('overall_confidence', 0.5)
        agreement = consensus.get('agreement_level', 0.5)
        
        summary_parts.append(f"Confidence: {confidence:.2f}")
        summary_parts.append(f"Agreement: {agreement:.2f}")
        
        # Actions triggered
        actions = consensus.get('consensus_actions', {})
        if actions:
            action_names = list(actions.keys())[:2]
            summary_parts.append(f"Actions: {', '.join(action_names)}")
        
        return " | ".join(summary_parts)
    
    def _select_template_type(self, experience: SensorimotorExperience, 
                            cortical_result: Dict[str, Any], 
                            identity_result: Dict[str, Any]) -> str:
        """Select appropriate template type"""
        
        # Check for analysis-heavy content
        analysis_keywords = ['analyze', 'evaluate', 'assess', 'examine', 'study', 'research']
        if any(keyword in experience.content.lower() for keyword in analysis_keywords):
            return 'analysis_focused'
        
        # Check for narrative development
        narrative_coherence = identity_result.get('coherence_assessment', {}).get('narrative_coherence', 0.5)
        if narrative_coherence > 0.7:
            return 'narrative_continuity'
        
        # Default to identity-aware
        return 'identity_aware'
    
    def _generate_identity_aware_prompt(self, user_input: str, template_type: str, 
                                      personality_summary: str, episodic_context: str, 
                                      cortical_summary: str, identity_result: Dict[str, Any]) -> str:
        """Generate identity-aware prompt"""
        
        template = self.context_templates[template_type]
        
        # Prepare template variables
        template_vars = {
            'user_input': user_input,
            'personality_summary': personality_summary,
            'episodic_context': episodic_context,
            'cortical_summary': cortical_summary
        }
        
        # Add template-specific variables
        if template_type == 'identity_aware':
            coherence_score = identity_result.get('coherence_assessment', {}).get('overall_coherence', 0.5)
            template_vars['coherence_score'] = f"{coherence_score:.2f}"
        
        elif template_type == 'analysis_focused':
            template_vars['identity_perspective'] = self._create_identity_perspective(identity_result)
        
        elif template_type == 'narrative_continuity':
            personality_state = identity_result.get('personality_state', {})
            template_vars['narrative_themes'] = ', '.join(personality_state.get('narrative_themes', [])[:3])
            template_vars['identity_development'] = personality_state.get('development_stage', 'unknown')
            template_vars['recent_insights'] = self._extract_recent_insights(identity_result)
        
        # Format template
        try:
            return template.format(**template_vars)
        except KeyError as e:
            # Fallback if template variable missing
            return f"Context: {personality_summary}\n\nInput: {user_input}\n\nRespond consistently with established identity."
    
    def _create_identity_perspective(self, identity_result: Dict[str, Any]) -> str:
        """Create identity perspective summary"""
        
        personality_state = identity_result.get('personality_state', {})
        
        # Extract cognitive style
        cognitive_style = personality_state.get('cognitive_style', {})
        perspective_parts = []
        
        if cognitive_style:
            for style, strength in cognitive_style.items():
                if strength > 0.6:
                    perspective_parts.append(f"{style}-oriented")
        
        # Add goal hierarchy
        goal_hierarchy = personality_state.get('goal_hierarchy', {})
        if goal_hierarchy:
            top_goals = sorted(goal_hierarchy.items(), key=lambda x: max(x[1].values()) if x[1] else 0, reverse=True)[:2]
            goal_names = [goal for goal, _ in top_goals]
            if goal_names:
                perspective_parts.append(f"focused on {', '.join(goal_names)}")
        
        return " and ".join(perspective_parts) if perspective_parts else "developing analytical perspective"
    
    def _extract_recent_insights(self, identity_result: Dict[str, Any]) -> str:
        """Extract recent insights from identity formation"""
        
        # Extract from meaning result
        meaning_result = identity_result.get('meaning_result', {})
        extracted_meaning = meaning_result.get('extracted_meaning', {})
        
        insights = []
        
        # Core insights
        core_insights = extracted_meaning.get('core_insights', [])
        if core_insights:
            insights.extend(core_insights[:2])
        
        # Pattern insights
        pattern_insights = extracted_meaning.get('pattern_insights', {})
        if pattern_insights:
            insight_texts = [f"{pattern}: {insight}" for pattern, insight in pattern_insights.items()]
            insights.extend(insight_texts[:2])
        
        return "; ".join(insights) if insights else "emerging insights about identity and purpose"
    
    def _simulate_llm_response(self, prompt: str, experience: SensorimotorExperience, 
                             identity_result: Dict[str, Any]) -> str:
        """Simulate LLM response (in real implementation, this would call actual LLM API)"""
        
        # Check cache first
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Simulate response based on identity characteristics
        personality_state = identity_result.get('personality_state', {})
        traits = personality_state.get('traits_big5', {})
        
        # Response style based on personality
        response_style = self._determine_response_style(traits)
        
        # Generate response based on content analysis
        content_analysis = self._analyze_content_for_response(experience.content)
        
        # Construct simulated response
        response_parts = []
        
        # Opening based on personality
        if traits.get('extraversion', 0.5) > 0.6:
            response_parts.append("I find this quite engaging.")
        elif traits.get('openness', 0.5) > 0.7:
            response_parts.append("This presents interesting possibilities to explore.")
        else:
            response_parts.append("Let me consider this carefully.")
        
        # Main content analysis
        if content_analysis['is_analytical']:
            response_parts.append(f"From an analytical perspective, {experience.content.lower()[:50]}... suggests several important considerations.")
        elif content_analysis['is_creative']:
            response_parts.append(f"This creative challenge opens up multiple avenues for innovation and exploration.")
        else:
            response_parts.append(f"The key aspects to address here involve understanding the underlying patterns and implications.")
        
        # Personality-influenced conclusion
        if traits.get('conscientiousness', 0.5) > 0.7:
            response_parts.append("I'd recommend a systematic approach to ensure comprehensive coverage.")
        elif traits.get('agreeableness', 0.5) > 0.6:
            response_parts.append("Collaboration and diverse perspectives would enhance our understanding here.")
        else:
            response_parts.append("This warrants deeper investigation to fully grasp the implications.")
        
        response = " ".join(response_parts)
        
        # Cache response
        self.response_cache[cache_key] = response
        
        return response
    
    def _determine_response_style(self, traits: Dict[str, float]) -> str:
        """Determine response style based on personality traits"""
        
        # Analytical vs. Intuitive
        openness = traits.get('openness', 0.5)
        conscientiousness = traits.get('conscientiousness', 0.5)
        
        if conscientiousness > 0.7 and openness > 0.6:
            return 'analytical_creative'
        elif conscientiousness > 0.7:
            return 'analytical_structured'
        elif openness > 0.7:
            return 'creative_exploratory'
        else:
            return 'balanced_thoughtful'
    
    def _analyze_content_for_response(self, content: str) -> Dict[str, bool]:
        """Analyze content to determine response approach"""
        
        content_lower = content.lower()
        
        # Check for analytical content
        analytical_keywords = ['analyze', 'data', 'research', 'study', 'examine', 'evaluate', 'assess']
        is_analytical = any(keyword in content_lower for keyword in analytical_keywords)
        
        # Check for creative content
        creative_keywords = ['create', 'design', 'innovate', 'imagine', 'artistic', 'creative']
        is_creative = any(keyword in content_lower for keyword in creative_keywords)
        
        # Check for problem-solving content
        problem_keywords = ['problem', 'solve', 'solution', 'challenge', 'issue', 'difficulty']
        is_problem_solving = any(keyword in content_lower for keyword in problem_keywords)
        
        # Check for emotional content
        emotional_keywords = ['feel', 'emotion', 'excited', 'worried', 'happy', 'concerned']
        is_emotional = any(keyword in content_lower for keyword in emotional_keywords)
        
        return {
            'is_analytical': is_analytical,
            'is_creative': is_creative,
            'is_problem_solving': is_problem_solving,
            'is_emotional': is_emotional
        }
    
    def _update_conversation_history(self, user_input: str, llm_response: str, 
                                   identity_result: Dict[str, Any]):
        """Update conversation history"""
        
        coherence = identity_result.get('coherence_assessment', {}).get('overall_coherence', 0.5)
        
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'llm_response': llm_response,
            'identity_coherence': coherence,
            'response_length': len(llm_response)
        })
    
    def _assess_response_coherence(self, response: str, identity_result: Dict[str, Any]) -> float:
        """Assess response coherence with identity"""
        
        # Simple coherence assessment based on response characteristics
        personality_state = identity_result.get('personality_state', {})
        traits = personality_state.get('traits_big5', {})
        
        coherence_factors = []
        
        # Response length coherence
        response_length = len(response.split())
        expected_length = 30 + traits.get('extraversion', 0.5) * 20  # More extraverted = longer responses
        length_coherence = 1.0 - abs(response_length - expected_length) / expected_length
        coherence_factors.append(max(0, length_coherence))
        
        # Analytical language coherence
        analytical_words = ['analyze', 'consider', 'examine', 'perspective', 'approach']
        analytical_count = sum(1 for word in analytical_words if word in response.lower())
        expected_analytical = traits.get('conscientiousness', 0.5) * 3
        analytical_coherence = 1.0 - abs(analytical_count - expected_analytical) / (expected_analytical + 1)
        coherence_factors.append(analytical_coherence)
        
        # Creative language coherence
        creative_words = ['creative', 'innovative', 'explore', 'possibilities', 'imagine']
        creative_count = sum(1 for word in creative_words if word in response.lower())
        expected_creative = traits.get('openness', 0.5) * 2
        creative_coherence = 1.0 - abs(creative_count - expected_creative) / (expected_creative + 1)
        coherence_factors.append(creative_coherence)
        
        return np.mean(coherence_factors)
    
    def _assess_identity_consistency(self, response: str, identity_result: Dict[str, Any]) -> float:
        """Assess identity consistency in response"""
        
        # Check consistency with core values
        personality_state = identity_result.get('personality_state', {})
        core_values = personality_state.get('core_value_system', {})
        
        consistency_score = 0.5  # Base consistency
        
        # Value expression in response
        value_keywords = {
            'honesty': ['honest', 'truth', 'genuine', 'authentic'],
            'excellence': ['excellent', 'quality', 'best', 'outstanding'],
            'growth': ['learn', 'develop', 'improve', 'progress'],
            'collaboration': ['together', 'team', 'collaborate', 'partnership']
        }
        
        value_consistency = []
        response_lower = response.lower()
        
        for value, strength in core_values.items():
            if value in value_keywords and strength > 0.6:
                keywords = value_keywords[value]
                value_expression = sum(1 for keyword in keywords if keyword in response_lower)
                expected_expression = strength * 2  # Higher valued = more expression expected
                
                if value_expression > 0:
                    consistency = min(1.0, value_expression / expected_expression)
                    value_consistency.append(consistency)
        
        if value_consistency:
            consistency_score = np.mean([consistency_score] + value_consistency)
        
        return consistency_score
    
    def _assess_narrative_continuity(self, response: str, identity_result: Dict[str, Any]) -> float:
        """Assess narrative continuity in response"""
        
        # Check for narrative themes
        personality_state = identity_result.get('personality_state', {})
        narrative_themes = personality_state.get('narrative_themes', [])
        
        if not narrative_themes:
            return 0.5  # Neutral if no themes established
        
        response_words = set(response.lower().split())
        theme_words = set()
        
        for theme in narrative_themes:
            theme_words.update(theme.lower().split())
        
        # Calculate theme overlap
        overlap = len(response_words & theme_words)
        total_theme_words = len(theme_words)
        
        if total_theme_words > 0:
            continuity = overlap / total_theme_words
            return min(1.0, continuity + 0.3)  # Boost base continuity
        
        return 0.5
    
    def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        
        if not self.conversation_history:
            return {'status': 'no_conversations'}
        
        # Calculate statistics
        total_conversations = len(self.conversation_history)
        avg_response_length = np.mean([conv['response_length'] for conv in self.conversation_history])
        avg_coherence = np.mean([conv['identity_coherence'] for conv in self.conversation_history])
        
        # Recent performance
        recent_conversations = list(self.conversation_history)[-10:]
        recent_coherence = np.mean([conv['identity_coherence'] for conv in recent_conversations])
        
        return {
            'total_conversations': total_conversations,
            'average_response_length': avg_response_length,
            'average_identity_coherence': avg_coherence,
            'recent_coherence': recent_coherence,
            'cache_size': len(self.response_cache),
            'model_name': self.model_name
        }

# ============================================================================
# MULTI-AGENT COORDINATION (COMPLETE ORIGINAL)
# ============================================================================

class MultiAgentCoordinator:
    """Complete multi-agent coordination system"""
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.registered_components = {}
        self.agent_roles = {}
        self.coordination_history = deque(maxlen=100)
        self.performance_metrics = {}
        
        # Initialize agent roles
        self._initialize_agent_roles()
        
    def _initialize_agent_roles(self):
        """Initialize different agent roles"""
        
        self.agent_roles = {
            'analyzer': {
                'responsibilities': ['data_analysis', 'pattern_recognition', 'trend_identification'],
                'expertise_domains': ['financial_analysis', 'research'],
                'decision_weight': 0.3
            },
            'synthesizer': {
                'responsibilities': ['information_integration', 'cross_domain_connections', 'holistic_perspective'],
                'expertise_domains': ['general', 'research'],
                'decision_weight': 0.25
            },
            'identity_keeper': {
                'responsibilities': ['identity_consistency', 'narrative_coherence', 'personality_maintenance'],
                'expertise_domains': ['identity_formation', 'personality_development'],
                'decision_weight': 0.25
            },
            'memory_manager': {
                'responsibilities': ['memory_optimization', 'retrieval_efficiency', 'storage_prioritization'],
                'expertise_domains': ['memory_management', 'information_storage'],
                'decision_weight': 0.2
            }
        }
    
    def register_component(self, component_name: str, component_instance):
        """Register a component for coordination"""
        self.registered_components[component_name] = component_instance
        
    def coordinate_processing(self, experience: SensorimotorExperience, 
                            cortical_result: Dict[str, Any], 
                            identity_result: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate processing across multiple agents"""
        
        coordination_start = time.time()
        
        # Generate agent perspectives
        agent_perspectives = self._generate_agent_perspectives(experience, cortical_result, identity_result)
        
        # Negotiate decisions
        negotiated_decisions = self._negotiate_agent_decisions(agent_perspectives, experience)
        
        # Integrate agent outputs
        integrated_output = self._integrate_agent_outputs(agent_perspectives, negotiated_decisions)
        
        # Evaluate coordination quality
        coordination_quality = self._evaluate_coordination_quality(agent_perspectives, integrated_output)
        
        # Update performance metrics
        self._update_coordination_metrics(coordination_quality, time.time() - coordination_start)
        
        # Record coordination history
        coordination_record = {
            'timestamp': datetime.now().isoformat(),
            'experience_id': experience.experience_id,
            'agent_perspectives': {name: perspective['decision_confidence'] for name, perspective in agent_perspectives.items()},
            'coordination_quality': coordination_quality,
            'processing_time': time.time() - coordination_start
        }
        self.coordination_history.append(coordination_record)
        
        return {
            'agent_perspectives': agent_perspectives,
            'negotiated_decisions': negotiated_decisions,
            'integrated_output': integrated_output,
            'coordination_quality': coordination_quality,
            'coordination_time': time.time() - coordination_start,
            'participating_agents': list(agent_perspectives.keys())
        }
    
    def _generate_agent_perspectives(self, experience: SensorimotorExperience, 
                                   cortical_result: Dict[str, Any], 
                                   identity_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate perspectives from each agent role"""
        
        perspectives = {}
        
        for agent_name, agent_config in self.agent_roles.items():
            perspective = self._generate_single_agent_perspective(
                agent_name, agent_config, experience, cortical_result, identity_result
            )
            perspectives[agent_name] = perspective
        
        return perspectives
    
    def _generate_single_agent_perspective(self, agent_name: str, agent_config: Dict[str, Any],
                                         experience: SensorimotorExperience, 
                                         cortical_result: Dict[str, Any], 
                                         identity_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate perspective for a single agent"""
        
        responsibilities = agent_config['responsibilities']
        expertise_domains = agent_config['expertise_domains']
        decision_weight = agent_config['decision_weight']
        
        perspective = {
            'agent_name': agent_name,
            'decision_weight': decision_weight,
            'recommendations': [],
            'confidence_scores': {},
            'decision_confidence': 0.0
        }
        
        # Generate agent-specific recommendations
        if agent_name == 'analyzer':
            perspective.update(self._generate_analyzer_perspective(experience, cortical_result))
        elif agent_name == 'synthesizer':
            perspective.update(self._generate_synthesizer_perspective(experience, cortical_result, identity_result))
        elif agent_name == 'identity_keeper':
            perspective.update(self._generate_identity_keeper_perspective(experience, identity_result))
        elif agent_name == 'memory_manager':
            perspective.update(self._generate_memory_manager_perspective(experience, cortical_result))
        
        # Calculate overall decision confidence
        if perspective['confidence_scores']:
            perspective['decision_confidence'] = np.mean(list(perspective['confidence_scores'].values()))
        
        return perspective
    
    def _generate_analyzer_perspective(self, experience: SensorimotorExperience, 
                                     cortical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analyzer agent perspective"""
        
        recommendations = []
        confidence_scores = {}
        
        # Analyze prediction accuracy
        pred_accuracy = cortical_result.get('prediction_accuracy', 0.5)
        if pred_accuracy > 0.8:
            recommendations.append("High prediction accuracy suggests reliable pattern recognition")
            confidence_scores['prediction_reliability'] = pred_accuracy
        elif pred_accuracy < 0.4:
            recommendations.append("Low prediction accuracy indicates need for more data or pattern refinement")
            confidence_scores['prediction_reliability'] = 1.0 - pred_accuracy
        
        # Analyze domain expertise
        domain_expertise = cortical_result.get('domain_expertise_level', 0.5)
        if domain_expertise > 0.7:
            recommendations.append("Strong domain expertise enables confident analysis")
            confidence_scores['domain_confidence'] = domain_expertise
        
        # Analyze consensus quality
        consensus = cortical_result.get('consensus', {})
        consensus_confidence = consensus.get('overall_confidence', 0.5)
        agreement_level = consensus.get('agreement_level', 0.5)
        
        if consensus_confidence > 0.7 and agreement_level > 0.6:
            recommendations.append("Strong consensus supports analytical conclusions")
            confidence_scores['consensus_quality'] = (consensus_confidence + agreement_level) / 2
        
        # Content analysis recommendations
        content_length = len(experience.content.split())
        if content_length > 50:
            recommendations.append("Substantial content provides rich analytical material")
            confidence_scores['content_richness'] = min(1.0, content_length / 100)
        
        return {
            'recommendations': recommendations,
            'confidence_scores': confidence_scores,
            'primary_focus': 'analytical_rigor',
            'suggested_actions': ['detailed_analysis', 'pattern_validation', 'trend_extrapolation']
        }
    
    def _generate_synthesizer_perspective(self, experience: SensorimotorExperience, 
                                        cortical_result: Dict[str, Any], 
                                        identity_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthesizer agent perspective"""
        
        recommendations = []
        confidence_scores = {}
        
        # Cross-domain synthesis opportunities
        domain = experience.domain
        if domain in ['financial_analysis', 'research']:
            recommendations.append(f"Cross-domain insights can enhance {domain} understanding")
            confidence_scores['cross_domain_potential'] = 0.7
        
        # Identity-cognition integration
        identity_coherence = identity_result.get('coherence_assessment', {}).get('overall_coherence', 0.5)
        cortical_confidence = cortical_result.get('consensus', {}).get('overall_confidence', 0.5)
        
        if abs(identity_coherence - cortical_confidence) < 0.2:
            recommendations.append("Strong alignment between identity and cognitive processing")
            confidence_scores['identity_cognition_alignment'] = 1.0 - abs(identity_coherence - cortical_confidence)
        
        # Narrative synthesis opportunities
        narrative_result = identity_result.get('narrative_result', {})
        if narrative_result.get('narrative_text'):
            recommendations.append("Rich narrative context enables deeper synthesis")
            confidence_scores['narrative_integration'] = 0.8
        
        # Holistic perspective assessment
        episodic_integration = cortical_result.get('episodic_integration_quality', 0.5)
        if episodic_integration > 0.6:
            recommendations.append("Strong episodic integration supports holistic understanding")
            confidence_scores['holistic_integration'] = episodic_integration
        
        return {
            'recommendations': recommendations,
            'confidence_scores': confidence_scores,
            'primary_focus': 'holistic_integration',
            'suggested_actions': ['cross_domain_mapping', 'pattern_synthesis', 'perspective_integration']
        }
    
    def _generate_identity_keeper_perspective(self, experience: SensorimotorExperience, 
                                            identity_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate identity keeper agent perspective"""
        
        recommendations = []
        confidence_scores = {}
        
        # Identity coherence assessment
        coherence_assessment = identity_result.get('coherence_assessment', {})
        overall_coherence = coherence_assessment.get('overall_coherence', 0.5)
        
        if overall_coherence > 0.8:
            recommendations.append("Strong identity coherence maintained")
            confidence_scores['coherence_maintenance'] = overall_coherence
        elif overall_coherence < 0.4:
            recommendations.append("Identity coherence requires attention and stabilization")
            confidence_scores['coherence_concern'] = 1.0 - overall_coherence
        
        # Narrative consistency
        narrative_coherence = coherence_assessment.get('narrative_coherence', 0.5)
        if narrative_coherence > 0.7:
            recommendations.append("Narrative consistency supports identity stability")
            confidence_scores['narrative_consistency'] = narrative_coherence
        
        # Personality development
        personality_state = identity_result.get('personality_state', {})
        identity_stability = personality_state.get('identity_stability', 0.5)
        
        if identity_stability > 0.6:
            recommendations.append("Personality development shows healthy stability")
            confidence_scores['personality_stability'] = identity_stability
        
        # Value consistency
        value_coherence = coherence_assessment.get('value_coherence', 0.5)
        if value_coherence > 0.7:
            recommendations.append("Core values remain consistent")
            confidence_scores['value_consistency'] = value_coherence
        
        return {
            'recommendations': recommendations,
            'confidence_scores': confidence_scores,
            'primary_focus': 'identity_preservation',
            'suggested_actions': ['coherence_monitoring', 'narrative_maintenance', 'value_reinforcement']
        }
    
    def _generate_memory_manager_perspective(self, experience: SensorimotorExperience, 
                                           cortical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate memory manager agent perspective"""
        
        recommendations = []
        confidence_scores = {}
        
        # Memory efficiency assessment
        if 'episodic_memory' in self.registered_components:
            episodic_memory = self.registered_components['episodic_memory']
            memory_stats = episodic_memory.get_enhanced_memory_statistics()
            
            # Episode storage efficiency
            total_episodes = memory_stats.get('total_episodes', 0)
            if total_episodes > 50:
                recommendations.append("Rich episodic memory enables effective retrieval")
                confidence_scores['memory_richness'] = min(1.0, total_episodes / 100)
            
            # Memory span optimization
            memory_span = memory_stats.get('memory_span_days', 0)
            if memory_span > 1:
                recommendations.append(f"Memory span of {memory_span:.1f} days provides temporal context")
                confidence_scores['temporal_coverage'] = min(1.0, memory_span / 7)
        
        # Novelty-based storage priority
        novelty_score = experience.novelty_score
        if novelty_score > 0.7:
            recommendations.append("High novelty content deserves priority storage")
            confidence_scores['storage_priority'] = novelty_score
        
        # Retrieval efficiency
        prediction_accuracy = cortical_result.get('prediction_accuracy', 0.5)
        if prediction_accuracy > 0.7:
            recommendations.append("Good prediction accuracy suggests effective memory utilization")
            confidence_scores['retrieval_effectiveness'] = prediction_accuracy
        
        return {
            'recommendations': recommendations,
            'confidence_scores': confidence_scores,
            'primary_focus': 'memory_optimization',
            'suggested_actions': ['storage_prioritization', 'retrieval_optimization', 'memory_consolidation']
        }
    
    def _negotiate_agent_decisions(self, agent_perspectives: Dict[str, Dict[str, Any]], 
                                 experience: SensorimotorExperience) -> Dict[str, Any]:
        """Negotiate decisions across agents"""
        
        # Collect all recommendations
        all_recommendations = []
        weighted_confidences = []
        
        for agent_name, perspective in agent_perspectives.items():
            decision_weight = perspective['decision_weight']
            decision_confidence = perspective['decision_confidence']
            
            weighted_confidence = decision_weight * decision_confidence
            weighted_confidences.append(weighted_confidence)
            
            # Add weighted recommendations
            for rec in perspective['recommendations']:
                all_recommendations.append({
                    'agent': agent_name,
                    'recommendation': rec,
                    'weight': decision_weight,
                    'confidence': decision_confidence,
                    'weighted_score': weighted_confidence
                })
        
        # Negotiate conflicts and find consensus
        consensus_decisions = self._find_consensus_decisions(all_recommendations)
        
        # Calculate overall negotiation confidence
        overall_confidence = np.mean(weighted_confidences) if weighted_confidences else 0.5
        
        return {
            'consensus_decisions': consensus_decisions,
            'overall_confidence': overall_confidence,
            'negotiation_quality': self._assess_negotiation_quality(agent_perspectives),
            'conflict_resolution': self._identify_resolved_conflicts(all_recommendations)
        }
    
    def _find_consensus_decisions(self, all_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find consensus among agent recommendations"""
        
        # Group similar recommendations
        recommendation_groups = defaultdict(list)
        
        for rec in all_recommendations:
            # Simple grouping by keywords
            rec_words = set(rec['recommendation'].lower().split())
            
            # Find best matching group
            best_group = None
            best_overlap = 0
            
            for group_key, group_recs in recommendation_groups.items():
                group_words = set(group_key.split())
                overlap = len(rec_words & group_words)
                
                if overlap > best_overlap and overlap > 2:
                    best_group = group_key
                    best_overlap = overlap
            
            if best_group:
                recommendation_groups[best_group].append(rec)
            else:
                # Create new group
                key_words = [word for word in rec_words if len(word) > 4][:3]
                new_key = ' '.join(key_words)
                recommendation_groups[new_key].append(rec)
        
        # Generate consensus decisions
        consensus_decisions = []
        
        for group_key, group_recs in recommendation_groups.items():
            if len(group_recs) >= 2:  # Require at least 2 agents to agree
                total_weight = sum(rec['weighted_score'] for rec in group_recs)
                agent_names = [rec['agent'] for rec in group_recs]
                
                consensus_decisions.append({
                    'decision_theme': group_key,
                    'supporting_agents': agent_names,
                    'consensus_strength': total_weight,
                    'agent_count': len(group_recs),
                    'representative_recommendation': group_recs[0]['recommendation']
                })
        
        # Sort by consensus strength
        consensus_decisions.sort(key=lambda x: x['consensus_strength'], reverse=True)
        
        return consensus_decisions
    
    def _assess_negotiation_quality(self, agent_perspectives: Dict[str, Dict[str, Any]]) -> float:
        """Assess quality of agent negotiation"""
        
        quality_factors = []
        
        # Agent participation
        participation_rate = len(agent_perspectives) / len(self.agent_roles)
        quality_factors.append(participation_rate)
        
        # Confidence distribution
        confidences = [p['decision_confidence'] for p in agent_perspectives.values()]
        if confidences:
            avg_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            
            quality_factors.append(avg_confidence)
            quality_factors.append(1.0 - min(1.0, confidence_std))  # Lower std = higher quality
        
        # Recommendation diversity
        all_recs = []
        for perspective in agent_perspectives.values():
            all_recs.extend(perspective['recommendations'])
        
        if all_recs:
            unique_recs = len(set(all_recs))
            diversity = unique_recs / len(all_recs)
            quality_factors.append(diversity)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def _identify_resolved_conflicts(self, all_recommendations: List[Dict[str, Any]]) -> List[str]:
        """Identify conflicts that were resolved during negotiation"""
        
        conflicts_resolved = []
        
        # Look for opposing recommendations
        opposing_keywords = [
            ('increase', 'decrease'),
            ('high', 'low'),
            ('strong', 'weak'),
            ('confident', 'uncertain'),
            ('reliable', 'unreliable')
        ]
        
        for keyword1, keyword2 in opposing_keywords:
            recs_with_kw1 = [rec for rec in all_recommendations if keyword1 in rec['recommendation'].lower()]
            recs_with_kw2 = [rec for rec in all_recommendations if keyword2 in rec['recommendation'].lower()]
            
            if recs_with_kw1 and recs_with_kw2:
                conflicts_resolved.append(f"Resolved {keyword1} vs {keyword2} conflict")
        
        return conflicts_resolved
    
    def _integrate_agent_outputs(self, agent_perspectives: Dict[str, Dict[str, Any]], 
                               negotiated_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate outputs from all agents"""
        
        integrated_output = {
            'primary_recommendations': [],
            'action_priorities': {},
            'confidence_assessment': {},
            'integration_quality': 0.0
        }
        
        # Extract primary recommendations from consensus
        consensus_decisions = negotiated_decisions.get('consensus_decisions', [])
        for decision in consensus_decisions[:3]:  # Top 3 consensus decisions
            integrated_output['primary_recommendations'].append({
                'recommendation': decision['representative_recommendation'],
                'consensus_strength': decision['consensus_strength'],
                'supporting_agents': decision['supporting_agents']
            })
        
        # Aggregate action priorities
        action_priorities = defaultdict(float)
        
        for agent_name, perspective in agent_perspectives.items():
            suggested_actions = perspective.get('suggested_actions', [])
            agent_weight = perspective['decision_weight']
            agent_confidence = perspective['decision_confidence']
            
            for action in suggested_actions:
                action_priorities[action] += agent_weight * agent_confidence
        
        # Sort actions by priority
        sorted_actions = sorted(action_priorities.items(), key=lambda x: x[1], reverse=True)
        integrated_output['action_priorities'] = dict(sorted_actions[:5])
        
        # Confidence assessment
        overall_confidence = negotiated_decisions.get('overall_confidence', 0.5)
        negotiation_quality = negotiated_decisions.get('negotiation_quality', 0.5)
        
        integrated_output['confidence_assessment'] = {
            'overall_confidence': overall_confidence,
            'negotiation_quality': negotiation_quality,
            'agent_agreement': self._calculate_agent_agreement(agent_perspectives)
        }
        
        # Integration quality
        integration_quality = (overall_confidence + negotiation_quality) / 2
        integrated_output['integration_quality'] = integration_quality
        
        return integrated_output
    
    def _calculate_agent_agreement(self, agent_perspectives: Dict[str, Dict[str, Any]]) -> float:
        """Calculate agreement level among agents"""
        
        if len(agent_perspectives) < 2:
            return 1.0
        
        # Compare confidence scores across agents
        all_confidences = [p['decision_confidence'] for p in agent_perspectives.values()]
        confidence_std = np.std(all_confidences)
        
        # Lower standard deviation = higher agreement
        agreement = 1.0 - min(1.0, confidence_std)
        
        return agreement
    
    def _evaluate_coordination_quality(self, agent_perspectives: Dict[str, Dict[str, Any]], 
                                     integrated_output: Dict[str, Any]) -> float:
        """Evaluate overall coordination quality"""
        
        quality_factors = []
        
        # Agent participation quality
        participation_quality = len(agent_perspectives) / len(self.agent_roles)
        quality_factors.append(participation_quality)
        
        # Decision confidence
        avg_confidence = np.mean([p['decision_confidence'] for p in agent_perspectives.values()])
        quality_factors.append(avg_confidence)
        
        # Integration quality
        integration_quality = integrated_output.get('integration_quality', 0.5)
        quality_factors.append(integration_quality)
        
        # Recommendation coherence
        primary_recs = integrated_output.get('primary_recommendations', [])
        if primary_recs:
            avg_consensus_strength = np.mean([rec['consensus_strength'] for rec in primary_recs])
            quality_factors.append(avg_consensus_strength)
        
        return np.mean(quality_factors)
    
    def _update_coordination_metrics(self, coordination_quality: float, processing_time: float):
        """Update coordination performance metrics"""
        
        current_time = time.time()
        
        if 'coordination_quality' not in self.performance_metrics:
            self.performance_metrics['coordination_quality'] = deque(maxlen=50)
        
        if 'processing_time' not in self.performance_metrics:
            self.performance_metrics['processing_time'] = deque(maxlen=50)
        
        self.performance_metrics['coordination_quality'].append(coordination_quality)
        self.performance_metrics['processing_time'].append(processing_time)
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination system statistics"""
        
        stats = {
            'system_id': self.system_id,
            'registered_components': list(self.registered_components.keys()),
            'agent_roles': list(self.agent_roles.keys()),
            'coordination_history_length': len(self.coordination_history)
        }
        
        # Performance metrics
        if self.performance_metrics:
            for metric_name, metric_values in self.performance_metrics.items():
                if metric_values:
                    stats[f'avg_{metric_name}'] = np.mean(metric_values)
                    stats[f'recent_{metric_name}'] = metric_values[-1] if metric_values else 0
        
        # Recent coordination quality
        if self.coordination_history:
            recent_quality = [record['coordination_quality'] for record in list(self.coordination_history)[-10:]]
            stats['recent_avg_quality'] = np.mean(recent_quality)
        
        return stats

# ============================================================================
# REAL-TIME DATA INTEGRATION (COMPLETE ORIGINAL)  
# ============================================================================

import os
import json
import time
import threading
import requests
import feedparser
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import signal
import sys

# Optional imports - will gracefully handle if not available
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    
try:
    from web3 import Web3
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    
try:
    from binance.client import Client as BinanceClient
    from binance import ThreadedWebsocketManager
    HAS_BINANCE = True
except ImportError:
    HAS_BINANCE = False

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    source_type: str
    endpoint: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    rate_limit: int = 60
    priority: int = 1
    enabled: bool = True
    last_fetch: Optional[datetime] = None
    error_count: int = 0

class RealTimeDataIntegrator:
    """
    Bulletproof Real-time data integrator for continuous learning
    Complete rewrite with proper async handling and EMMS compatibility
    """
    
    def __init__(self, memory_system=None, api_keys=None):
        """Initialize the bulletproof real-time data integrator"""
        
        # Core EMMS integration
        self.memory_system = memory_system
        self.api_keys = api_keys or {}
        
        # Initialize configuration
        self._load_configuration()
        
        # Thread-safe data structures
        self.processing_queue = queue.Queue(maxsize=10000)
        self.integration_history = deque(maxlen=1000)
        self.last_update = {}
        self.active_streams = {}
        
        # Thread management
        self.shutdown_flag = threading.Event()
        self.background_threads = []
        self.thread_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="EMMS_Data")
        
        # Rate limiting and caching
        self.rate_limiters = defaultdict(lambda: {'count': 0, 'reset_time': time.time()})
        self.data_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Statistics with ALL expected keys for EMMS compatibility
        self.integration_stats = {
            'total_fetched': 0,
            'total_processed': 0,
            'quality_filtered': 0,
            'quality_filtered_count': 0,  # EMMS expects this key
            'duplicates_removed': 0,
            'dedup_count': 0,  # EMMS expects this key
            'deduplicated': 0,  # Fix: Add missing deduplicated key
            'deduplicated_count': 0,  # Fix: Add missing deduplicated_count key
            'novel_content': 0,
            'novel_content_count': 0,  # EMMS expects this key
            'novel_count': 0,  # EMMS expects this key
            'experiences_created': 0,
            'experiences_created_count': 0,  # EMMS expects this key
            'experiences_count': 0,  # EMMS expects this key
            'raw_data_count': 0,  # EMMS expects this key
            'raw_data_fetched': 0,  # EMMS expects this key
            'streams_active': 0,
            'articles_per_minute': 0,
            'errors': 0,
            'experiences_processed': 0,
            'last_cycle_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Initialize API clients safely
        self._initialize_clients()
        
        # Register cleanup handlers
        self._register_cleanup_handlers()
        
        # Track start time for uptime calculation
        self._start_time = time.time()
        
        logger.info("🚀 Bulletproof RealTimeDataIntegrator initialized successfully")

    def _load_configuration(self):
        """Load API configuration with your keys"""
        self.api_config = {
            'binance': {
                'api_key': 'WbWY9Z6iQijb2knbi0lse1fPw5O5m94BxMd83C0z24EvU3Nh0LBg4pVCbWovvnqW',
                'api_secret': 'X8PYYps4iRTEz8eVkPEFa7OqHfmQri3JoZX606Qy973HdOWOXk7mzBTnZviT37VN'
            },
            'ethereum': {
                'rpc_url': 'https://eth-mainnet.g.alchemy.com/v2/iMMsD3cGTki2F4HKizkEMLwmyZDt1fDI',
                'fallback_url': 'https://mainnet.infura.io/v3/f522dd369adb453fbf6e8356acac6142'
            },
            'coingecko': {
                'api_key': 'CG-wQuXibxjo6BZS6e2MMQB3y1S'
            },
            'news_apis': {
                'newsapi_org': 'ed839fb9cb8947eba79089b1c87d5331',
                'alpha_vantage': '7KPDM0W8TFXGWMR0',
                'marketaux': 'k9TVRlzGPmkPtieYZQ3pQFXR1ae489xY5Vn80UDi',
                'fmp': '1oqrdXBUSMVItWaoIj2qjZ111BsEMcTU',
                'finnhub': 'd1puftpr01qku4u4p6u0d1puftpr01qku4u4p6ug'
            },
            'blockchain_apis': {
                'etherscan': 'REZAZVUXMZ258XECZ54HEMA49MKTA45YXY',
                'tenderly': '03bv3SuolJqItULkxlcJiwLXQYCOmHCh',
                'oneinch': 'pYigLGiEHsDB1CfZuDLruGa3qQsySPEy',
                'thegraph': '83984585a228ad2b12fc7325458dd5e7',
                'uniswap': '0xDcDd5634b0a1F1092Dd63809b6B66Bca5D847797'
            }
        }
        
        # Initialize data sources
        self.data_sources = self._initialize_data_sources()

    def _initialize_data_sources(self) -> Dict[str, DataSource]:
        """Initialize all data sources"""
        sources = {}
        
        # Financial/Market Data Sources
        if HAS_BINANCE:
            sources['binance_spot'] = DataSource(
                name='Binance Market Data',
                source_type='api',
                endpoint='https://api.binance.com/api/v3/ticker/24hr',
                api_key=self.api_config['binance']['api_key'],
                api_secret=self.api_config['binance']['api_secret'],
                rate_limit=1200,
                priority=1
            )
        
        sources['coingecko'] = DataSource(
            name='CoinGecko Prices',
            source_type='api',
            endpoint='https://api.coingecko.com/api/v3/coins/markets',
            api_key=self.api_config['coingecko']['api_key'],
            params={'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 20},
            rate_limit=450,
            priority=1
        )
        
        # News Sources
        sources['newsapi'] = DataSource(
            name='NewsAPI Crypto News',
            source_type='api', 
            endpoint='https://newsapi.org/v2/everything',
            api_key=self.api_config['news_apis']['newsapi_org'],
            params={'q': 'cryptocurrency OR bitcoin OR ethereum', 'sortBy': 'publishedAt', 'pageSize': 20},
            rate_limit=100,
            priority=2
        )
        
        sources['alpha_vantage'] = DataSource(
            name='Alpha Vantage News',
            source_type='api',
            endpoint='https://www.alphavantage.co/query',
            api_key=self.api_config['news_apis']['alpha_vantage'],
            params={'function': 'NEWS_SENTIMENT', 'topics': 'blockchain,cryptocurrency'},
            rate_limit=25,
            priority=2
        )
        
        # RSS Sources
        sources['coindesk_rss'] = DataSource(
            name='CoinDesk RSS',
            source_type='rss',
            endpoint='https://www.coindesk.com/arc/outboundfeeds/rss/',
            rate_limit=30,
            priority=3
        )
        
        sources['cointelegraph_rss'] = DataSource(
            name='Cointelegraph RSS',
            source_type='rss',
            endpoint='https://cointelegraph.com/rss',
            rate_limit=30,
            priority=3
        )
        
        return sources

    def _initialize_clients(self):
        """Initialize API clients safely"""
        try:
            # HTTP session for general requests
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'EMMS-DataIntegrator/1.0',
                'Accept': 'application/json'
            })
            
            # Initialize Binance client if available
            if HAS_BINANCE:
                try:
                    self.binance_client = BinanceClient(
                        self.api_config['binance']['api_key'],
                        self.api_config['binance']['api_secret']
                    )
                    self.binance_websocket = None  # Initialize when needed
                    logger.info("✅ Binance client initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Binance client initialization failed: {e}")
                    self.binance_client = None
            else:
                self.binance_client = None
                logger.info("ℹ️ Binance not available (python-binance not installed)")
            
            # Initialize Web3 client if available
            if HAS_WEB3:
                try:
                    self.web3_client = Web3(Web3.HTTPProvider(
                        self.api_config['ethereum']['rpc_url']
                    ))
                    logger.info("✅ Web3 client initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Web3 client initialization failed: {e}")
                    self.web3_client = None
            else:
                self.web3_client = None
                logger.info("ℹ️ Web3 not available (web3 not installed)")
            
            logger.info("✅ API clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Client initialization error: {e}")

    def _register_cleanup_handlers(self):
        """Register cleanup handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    # ========================================================================
    # CORE EMMS COMPATIBILITY METHODS
    # ========================================================================

    def fetch_and_process_cycle(self, domain: str, count: int = 10) -> Dict[str, Any]:
        """
        EMMS compatibility method - synchronous fetch and process cycle
        Returns dictionary with ALL keys EMMS expects
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Starting fetch cycle for domain: {domain}, count: {count}")
            
            # Fetch raw data
            raw_data = self._fetch_domain_data(domain)
            raw_count = len(raw_data) if raw_data else 0
            
            # Process and filter data
            filtered_data = self._filter_quality_data(raw_data) if raw_data else []
            quality_count = len(filtered_data)
            
            # Remove duplicates
            deduplicated_data = self._deduplicate_data(filtered_data)
            dedup_count = len(deduplicated_data)
            
            # Detect novel content
            novel_data = self._detect_novel_content(deduplicated_data)
            novel_count = len(novel_data)
            
            # Convert to experiences
            experiences = self._convert_to_experiences(novel_data[:count], domain)
            experience_count = len(experiences)
            
            # Process through EMMS memory if available
            if self.memory_system and experiences:
                emms_processed = 0
                for experience in experiences:
                    try:
                        # Convert to comprehensive sensorimotor experience
                        sensorimotor_exp = self._convert_to_sensorimotor_experience(experience)
                        
                        # Process through EMMS comprehensive pipeline
                        if hasattr(self.memory_system, 'process_experience_comprehensive'):
                            memory_result = self.memory_system.process_experience_comprehensive(sensorimotor_exp)
                            emms_processed += 1
                            
                            # Also ensure it gets stored in hierarchical memory
                            if hasattr(self.memory_system, 'hierarchical_memory') and hasattr(self.memory_system.hierarchical_memory, 'store_experience'):
                                self.memory_system.hierarchical_memory.store_experience(sensorimotor_exp)
                            
                            # Force consolidation check to move data through memory hierarchy
                            if hasattr(self.memory_system, 'hierarchical_memory') and hasattr(self.memory_system.hierarchical_memory, '_check_immediate_consolidation'):
                                self.memory_system.hierarchical_memory._check_immediate_consolidation()
                                
                        elif hasattr(self.memory_system, 'store_experience'):
                            # Direct storage if comprehensive method not available
                            memory_result = self.memory_system.store_experience(sensorimotor_exp)
                            emms_processed += 1
                            
                        elif hasattr(self.memory_system, 'hierarchical_memory'):
                            # Try hierarchical memory directly
                            if hasattr(self.memory_system.hierarchical_memory, 'store_experience'):
                                self.memory_system.hierarchical_memory.store_experience(sensorimotor_exp)
                                emms_processed += 1
                                
                    except Exception as e:
                        logger.error(f"EMMS memory processing error: {e}")
                        continue
                
                # Update EMMS processing statistics
                self.integration_stats['experiences_processed'] += emms_processed
                
                if emms_processed > 0:
                    logger.info(f"✅ EMMS Integration: {emms_processed}/{len(experiences)} experiences processed into hierarchical memory")
                else:
                    logger.warning(f"⚠️ EMMS Integration: 0/{len(experiences)} experiences processed - check memory system connection")
            
            # Update statistics
            cycle_time = time.time() - start_time
            self._update_stats(raw_count, quality_count, dedup_count, novel_count, experience_count, cycle_time)
            
            # Store integration record
            integration_record = {
                'timestamp': datetime.now().isoformat(),
                'domain': domain,
                'raw_fetched': raw_count,
                'quality_filtered': quality_count,
                'deduplicated': dedup_count,
                'novel_content': novel_count,
                'experiences_created': experience_count,
                'cycle_time': cycle_time
            }
            self.integration_history.append(integration_record)
            
            # Return with ALL possible keys EMMS might expect
            result = {
                'domain': domain,
                'cycle_time': cycle_time,
                'processing_time': cycle_time,
                
                # Raw data - all possible names
                'raw_data_count': raw_count,
                'raw_data_fetched': raw_count,
                'raw_count': raw_count,
                'raw_total': raw_count,
                'total_raw': raw_count,
                'fetched_count': raw_count,
                
                # Quality filtered - all possible names
                'quality_filtered': quality_count,
                'quality_filtered_count': quality_count,
                'quality_count': quality_count,
                'filtered_count': quality_count,
                'quality_total': quality_count,
                
                # Deduplicated - all possible names
                'deduplicated': dedup_count,
                'deduplicated_count': dedup_count,
                'dedup_count': dedup_count,
                'unique_count': dedup_count,
                'dedupe_count': dedup_count,
                
                # Novel content - all possible names
                'novel_content': novel_count,
                'novel_content_count': novel_count,
                'novel_count': novel_count,
                'novel': novel_count,
                'novel_total': novel_count,
                'new_count': novel_count,
                
                # Experiences created - all possible names
                'experiences_created': experience_count,
                'experiences_created_count': experience_count,
                'experiences_count': experience_count,
                'experiences_total': experience_count,
                'created_count': experience_count,
                
                # Data
                'experiences': experiences
            }
            
            logger.info(f"✅ Cycle complete: {raw_count} raw → {quality_count} quality → {dedup_count} unique → {novel_count} novel → {experience_count} experiences")
            return result
            
        except Exception as e:
            self.integration_stats['errors'] += 1
            logger.error(f"❌ Fetch cycle error for {domain}: {e}")
            
            # Return error result with all expected keys set to 0
            return {
                'domain': domain,
                'error': str(e),
                'cycle_time': time.time() - start_time,
                'processing_time': time.time() - start_time,
                'raw_data_count': 0, 'raw_data_fetched': 0, 'raw_count': 0, 'raw_total': 0, 'total_raw': 0, 'fetched_count': 0,
                'quality_filtered': 0, 'quality_filtered_count': 0, 'quality_count': 0, 'filtered_count': 0, 'quality_total': 0,
                'deduplicated': 0, 'deduplicated_count': 0, 'dedup_count': 0, 'unique_count': 0, 'dedupe_count': 0,
                'novel_content': 0, 'novel_content_count': 0, 'novel_count': 0, 'novel': 0, 'novel_total': 0, 'new_count': 0,
                'experiences_created': 0, 'experiences_created_count': 0, 'experiences_count': 0, 'experiences_total': 0, 'created_count': 0,
                'experiences': []
            }

    def start_continuous_integration(self, domains: List[str] = None) -> Dict[str, Any]:
        """Start continuous integration with proper thread management"""
        try:
            if domains is None:
                domains = ['financial_analysis', 'research']
            
            logger.info(f"🚀 Starting continuous integration for domains: {domains}")
            
            # Clear shutdown flag
            self.shutdown_flag.clear()
            
            # Start background fetch cycles for each domain
            for domain in domains:
                thread = threading.Thread(
                    target=self._continuous_fetch_worker,
                    args=(domain,),
                    name=f"FetchWorker-{domain}",
                    daemon=True
                )
                thread.start()
                self.background_threads.append(thread)
                
                # Update active streams
                self.active_streams[domain] = {
                    'status': 'active',
                    'started_at': datetime.now(),
                    'data_count': 0,
                    'last_fetch': None,
                    'error_count': 0
                }
            
            # Start WebSocket streams if available
            if self.binance_client and HAS_BINANCE:
                self._start_binance_websocket()
            
            # Start data processing worker
            processing_thread = threading.Thread(
                target=self._data_processing_worker,
                name="DataProcessor",
                daemon=True
            )
            processing_thread.start()
            self.background_threads.append(processing_thread)
            
            self.integration_stats['streams_active'] = len(domains)
            
            logger.info(f"✅ Started {len(self.background_threads)} background threads")
            
            return {
                'streams_initialized': len(domains),
                'domains': domains,
                'active_streams': len(self.active_streams),
                'background_threads': len(self.background_threads),
                'status': 'started',
                'continuous_integration': True
            }
            
        except Exception as e:
            logger.error(f"❌ Continuous integration start error: {e}")
            return {
                'streams_initialized': 0,
                'domains': domains or [],
                'active_streams': 0,
                'background_threads': 0,
                'status': 'error',
                'error': str(e)
            }

    def start_streams(self, domains: List[str] = None) -> Dict[str, Any]:
        """EMMS compatibility method - alias for start_continuous_integration"""
        return self.start_continuous_integration(domains)

    def start_integration(self, domains: List[str] = None) -> Dict[str, Any]:
        """EMMS compatibility method - alias for start_continuous_integration"""
        return self.start_continuous_integration(domains)

    def initialize_integration(self, domains: List[str] = None) -> Dict[str, Any]:
        """EMMS compatibility method - alias for start_continuous_integration"""
        return self.start_continuous_integration(domains)

    def stop_streams(self) -> bool:
        """Stop all data streams gracefully"""
        try:
            logger.info("🛑 Stopping all data streams...")
            
            # Signal shutdown
            self.shutdown_flag.set()
            
            # Stop Binance WebSocket if running
            if hasattr(self, 'binance_websocket') and self.binance_websocket:
                try:
                    self.binance_websocket.stop()
                    logger.info("✅ Binance WebSocket stopped")
                except Exception as e:
                    logger.error(f"Error stopping Binance WebSocket: {e}")
            
            # Wait for all threads to finish
            for thread in self.background_threads:
                if thread.is_alive():
                    thread.join(timeout=10)  # Wait up to 10 seconds
                    if thread.is_alive():
                        logger.warning(f"Thread {thread.name} did not stop gracefully")
            
            # Clear thread list
            self.background_threads.clear()
            
            # Clear active streams
            self.active_streams.clear()
            self.integration_stats['streams_active'] = 0
            
            logger.info("✅ All data streams stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping streams: {e}")
            return False

    def shutdown(self) -> bool:
        """Complete shutdown of the integrator"""
        try:
            logger.info("🛑 Shutting down RealTimeDataIntegrator...")
            
            # Stop streams
            self.stop_streams()
            
            # Shutdown thread pool (compatible with Python 3.10)
            if hasattr(self, 'thread_pool'):
                try:
                    self.thread_pool.shutdown(wait=True)  # Remove timeout parameter
                except Exception as e:
                    logger.error(f"Thread pool shutdown error: {e}")
            
            # Close HTTP session
            if hasattr(self, 'session'):
                self.session.close()
            
            logger.info("✅ RealTimeDataIntegrator shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
            return False

    # ========================================================================
    # DATA FETCHING AND PROCESSING
    # ========================================================================

    def _fetch_domain_data(self, domain: str) -> List[Dict[str, Any]]:
        """Fetch data for a specific domain"""
        all_data = []
        
        try:
            if domain in ['financial_analysis', 'market_data']:
                # Fetch market data
                market_data = self._fetch_market_data()
                if market_data:
                    all_data.extend(market_data)
            
            if domain in ['research', 'news']:
                # Fetch news data
                news_data = self._fetch_news_data()
                if news_data:
                    all_data.extend(news_data)
            
            if domain == 'blockchain':
                # Fetch blockchain data
                blockchain_data = self._fetch_blockchain_data()
                if blockchain_data:
                    all_data.extend(blockchain_data)
            
            self.integration_stats['total_fetched'] += len(all_data)
            return all_data
            
        except Exception as e:
            logger.error(f"Error fetching data for domain {domain}: {e}")
            return []

    def _fetch_market_data(self) -> List[Dict[str, Any]]:
        """Fetch market data from available sources"""
        market_data = []
        
        # Fetch from Binance if available
        if self.binance_client:
            try:
                logger.debug("Fetching Binance market data...")
                tickers = self.binance_client.get_ticker()
                
                if not tickers:
                    logger.warning("No tickers received from Binance")
                    return market_data
                
                # Filter for high volume pairs
                filtered_tickers = []
                for ticker in tickers:
                    try:
                        quote_volume = float(ticker.get('quoteVolume', 0))
                        if quote_volume > 1000000:  # $1M+ volume
                            filtered_tickers.append(ticker)
                    except (ValueError, TypeError):
                        continue  # Skip invalid tickers
                
                # Take top 20
                filtered_tickers = filtered_tickers[:20]
                
                for ticker in filtered_tickers:
                    try:
                        market_data.append({
                            'type': 'market_ticker',
                            'source': 'binance',
                            'symbol': ticker.get('symbol', ''),
                            'price': float(ticker.get('lastPrice', 0)),
                            'change': float(ticker.get('priceChangePercent', 0)),
                            'volume': float(ticker.get('volume', 0)),
                            'quote_volume': float(ticker.get('quoteVolume', 0)),
                            'timestamp': datetime.now().isoformat()
                        })
                    except (ValueError, TypeError, KeyError) as e:
                        logger.debug(f"Skipping invalid ticker: {e}")
                        continue
                
                logger.debug(f"✅ Fetched {len(filtered_tickers)} Binance tickers")
                
            except Exception as e:
                logger.error(f"Binance fetch error: {e}")
        
        # Fetch from CoinGecko
        try:
            if 'coingecko' in self.data_sources:
                source = self.data_sources['coingecko']
                
                if self._check_rate_limit('coingecko', source.rate_limit):
                    logger.debug("Fetching CoinGecko market data...")
                    params = source.params.copy()
                    if source.api_key:
                        params['x_cg_demo_api_key'] = source.api_key
                    
                    response = self.session.get(source.endpoint, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        coins = response.json()
                        
                        if not isinstance(coins, list):
                            logger.warning("CoinGecko returned non-list data")
                            return market_data
                        
                        for coin in coins[:10]:  # Top 10
                            try:
                                market_data.append({
                                    'type': 'coin_market',
                                    'source': 'coingecko',
                                    'symbol': str(coin.get('symbol', '')).upper(),
                                    'name': coin.get('name', ''),
                                    'price': float(coin.get('current_price', 0)),
                                    'change': float(coin.get('price_change_percentage_24h', 0)),
                                    'market_cap': float(coin.get('market_cap', 0)),
                                    'timestamp': datetime.now().isoformat()
                                })
                            except (ValueError, TypeError, KeyError) as e:
                                logger.debug(f"Skipping invalid coin: {e}")
                                continue
                        
                        logger.debug(f"✅ Fetched {len(coins)} CoinGecko coins")
                    else:
                        logger.warning(f"CoinGecko API error: {response.status_code}")
        
        except Exception as e:
            logger.error(f"CoinGecko fetch error: {e}")
        
        return market_data

    def _fetch_news_data(self) -> List[Dict[str, Any]]:
        """Fetch news data from available sources"""
        news_data = []
        
        # Fetch from NewsAPI
        try:
            if 'newsapi' in self.data_sources:
                source = self.data_sources['newsapi']
                
                if self._check_rate_limit('newsapi', source.rate_limit):
                    logger.debug("Fetching NewsAPI data...")
                    params = source.params.copy()
                    params['apiKey'] = source.api_key
                    
                    response = self.session.get(source.endpoint, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get('articles', [])
                        
                        if not isinstance(articles, list):
                            logger.warning("NewsAPI returned non-list articles")
                        else:
                            for article in articles[:10]:  # Top 10
                                try:
                                    if not isinstance(article, dict):
                                        continue
                                    
                                    news_data.append({
                                        'type': 'news_article',
                                        'source': 'newsapi',
                                        'title': str(article.get('title', '')),
                                        'description': str(article.get('description', '')),
                                        'url': str(article.get('url', '')),
                                        'published_at': str(article.get('publishedAt', '')),
                                        'timestamp': datetime.now().isoformat()
                                    })
                                except Exception as e:
                                    logger.debug(f"Skipping invalid article: {e}")
                                    continue
                            
                            logger.debug(f"✅ Fetched {len(articles)} NewsAPI articles")
                    else:
                        logger.warning(f"NewsAPI error: {response.status_code}")
        
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
        
        # Fetch from RSS feeds
        for source_name in ['coindesk_rss', 'cointelegraph_rss']:
            try:
                if source_name in self.data_sources:
                    source = self.data_sources[source_name]
                    
                    if self._check_rate_limit(source_name, source.rate_limit):
                        logger.debug(f"Fetching {source_name} data...")
                        feed = feedparser.parse(source.endpoint)
                        
                        if not hasattr(feed, 'entries') or not feed.entries:
                            logger.warning(f"No entries in {source_name} feed")
                            continue
                        
                        for entry in feed.entries[:5]:  # Top 5
                            try:
                                news_data.append({
                                    'type': 'rss_article',
                                    'source': source_name,
                                    'title': str(getattr(entry, 'title', '')),
                                    'description': str(getattr(entry, 'summary', '')),
                                    'url': str(getattr(entry, 'link', '')),
                                    'published_at': str(getattr(entry, 'published', '')),
                                    'timestamp': datetime.now().isoformat()
                                })
                            except Exception as e:
                                logger.debug(f"Skipping invalid RSS entry: {e}")
                                continue
                        
                        logger.debug(f"✅ Fetched {len(feed.entries)} {source_name} articles")
            
            except Exception as e:
                logger.error(f"{source_name} fetch error: {e}")
        
        return news_data

    def _fetch_blockchain_data(self) -> List[Dict[str, Any]]:
        """Fetch blockchain data if Web3 is available"""
        blockchain_data = []
        
        if self.web3_client:
            try:
                # Get latest block
                latest_block = self.web3_client.eth.get_block('latest')
                
                blockchain_data.append({
                    'type': 'ethereum_block',
                    'source': 'ethereum',
                    'block_number': latest_block.number,
                    'timestamp': datetime.fromtimestamp(latest_block.timestamp).isoformat(),
                    'gas_used': latest_block.gasUsed,
                    'gas_limit': latest_block.gasLimit,
                    'transaction_count': len(latest_block.transactions)
                })
                
                logger.debug(f"✅ Fetched Ethereum block {latest_block.number}")
                
            except Exception as e:
                logger.error(f"Ethereum fetch error: {e}")
        
        return blockchain_data

    def _filter_quality_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data based on quality metrics"""
        if not data:
            return []
        
        filtered = []
        for item in data:
            quality_score = self._calculate_quality_score(item)
            if quality_score > 0.3:  # Quality threshold
                item['quality_score'] = quality_score
                filtered.append(item)
        
        return filtered

    def _calculate_quality_score(self, item: Dict[str, Any]) -> float:
        """Calculate quality score for a data item"""
        score = 0.5  # Base score
        
        try:
            # Check for required fields
            if 'type' in item and item['type']:
                score += 0.1
            
            if 'source' in item and item['source']:
                score += 0.1
            
            if 'timestamp' in item and item['timestamp']:
                score += 0.1
            
            # Type-specific quality checks
            if item.get('type') == 'market_ticker':
                if 'price' in item and item['price'] > 0:
                    score += 0.1
                if 'volume' in item and item['volume'] > 0:
                    score += 0.1
            
            elif item.get('type') == 'news_article':
                if 'title' in item and len(item['title']) > 10:
                    score += 0.1
                if 'description' in item and len(item['description']) > 20:
                    score += 0.1
            
            return min(1.0, score)
            
        except Exception:
            return 0.2

    def _deduplicate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate data items"""
        if not data:
            return []
        
        seen_hashes = set()
        deduplicated = []
        
        for item in data:
            # Create hash based on key fields
            hash_content = f"{item.get('type', '')}_{item.get('source', '')}_{item.get('symbol', '')}_{item.get('title', '')}"
            item_hash = hashlib.md5(hash_content.encode()).hexdigest()
            
            if item_hash not in seen_hashes:
                seen_hashes.add(item_hash)
                deduplicated.append(item)
        
        return deduplicated

    def _detect_novel_content(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect novel content based on recent history"""
        if not data:
            return []
        
        novel_data = []
        current_time = time.time()
        
        for item in data:
            novelty_score = self._calculate_novelty_score(item)
            
            if novelty_score > 0.5:  # Novelty threshold
                item['novelty_score'] = novelty_score
                novel_data.append(item)
        
        return novel_data

    def _calculate_novelty_score(self, item: Dict[str, Any]) -> float:
        """Calculate novelty score for an item"""
        try:
            # Create content hash
            content = f"{item.get('type', '')}_{item.get('symbol', '')}_{item.get('title', '')}"
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check cache for recent similar content
            cache_key = f"novelty_{content_hash}"
            current_time = time.time()
            
            if cache_key in self.data_cache:
                age = current_time - self.data_cache[cache_key]
                # Novelty decreases with age (newer = more novel)
                return max(0.1, 1.0 - (age / 3600))  # 1 hour decay
            
            # Store in cache
            self.data_cache[cache_key] = current_time
            
            # Clean old cache entries
            self._clean_cache()
            
            return 0.8  # High novelty for new content
            
        except Exception:
            return 0.5

    def _clean_cache(self):
        """Clean old cache entries"""
        current_time = time.time()
        old_keys = [
            key for key, timestamp in self.data_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in old_keys:
            del self.data_cache[key]

    def _convert_to_experiences(self, data: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """Convert data items to EMMS experience format"""
        experiences = []
        
        for item in data:
            experience = {
                'content': json.dumps(item),
                'domain': domain,
                'source': item.get('source', 'unknown'),
                'timestamp': item.get('timestamp', datetime.now().isoformat()),
                'novelty_score': item.get('novelty_score', 0.5),
                'quality_score': item.get('quality_score', 0.5),
                'data_type': item.get('type', 'general')
            }
            experiences.append(experience)
        
        return experiences

    def _convert_to_sensorimotor_experience(self, experience: Dict[str, Any]) -> Any:
        """Convert experience to EMMS SensorimotorExperience format with proper integration"""
        try:
            # Import required classes
            if HAS_PANDAS:
                import numpy as np
            else:
                # Fallback numpy-like operations
                np = type('MockNumPy', (), {
                    'random': type('MockRandom', (), {
                        'rand': lambda size: [0.5] * size if isinstance(size, int) else [[0.5] * size[1] for _ in range(size[0])]
                    })(),
                    'array': lambda x: x,
                    'zeros': lambda size: [0.0] * size if isinstance(size, int) else [[0.0] * size[1] for _ in range(size[0])]
                })()
            
            # Create comprehensive sensorimotor experience
            sensorimotor_exp = type('SensorimotorExperience', (), {
                'experience_id': hashlib.md5(f"{experience['source']}_{experience['timestamp']}".encode()).hexdigest(),
                'content': experience['content'],
                'domain': experience['domain'],
                'sensory_features': {
                    'quality': experience.get('quality_score', 0.5),
                    'relevance': experience.get('relevance_score', 0.5),
                    'novelty': experience.get('novelty_score', 0.5),
                    'data_type': experience.get('data_type', 'general'),
                    'source': experience.get('source', 'unknown')
                },
                'motor_actions': [],
                'contextual_embedding': np.random.rand(16) if hasattr(np.random, 'rand') else [0.5] * 16,
                'temporal_markers': [time.time()],
                'attention_weights': {'content': 1.0, 'source': 0.8, 'novelty': experience.get('novelty_score', 0.5)},
                'prediction_targets': {},
                'novelty_score': experience.get('novelty_score', 0.5),
                'timestamp': experience['timestamp'],
                
                # Enhanced memory features for EMMS integration
                'emotional_features': {
                    'sentiment': self._extract_sentiment(experience.get('content', '')),
                    'intensity': experience.get('novelty_score', 0.5),
                    'valence': 0.5
                },
                'causal_indicators': [experience.get('source', 'unknown')],
                'goal_relevance': {
                    'financial_analysis': 1.0 if experience.get('domain') == 'financial_analysis' else 0.5,
                    'research': 1.0 if experience.get('domain') == 'research' else 0.5
                },
                'modality_features': {
                    'text': np.random.rand(16) if hasattr(np.random, 'rand') else [0.5] * 16,
                    'visual': np.random.rand(16) if hasattr(np.random, 'rand') else [0.3] * 16,
                    'audio': np.random.rand(16) if hasattr(np.random, 'rand') else [0.2] * 16,
                    'temporal': np.random.rand(16) if hasattr(np.random, 'rand') else [0.7] * 16,
                    'spatial': np.random.rand(16) if hasattr(np.random, 'rand') else [0.4] * 16,
                    'emotional': np.random.rand(16) if hasattr(np.random, 'rand') else [0.6] * 16
                },
                'importance_weight': experience.get('quality_score', 0.5),
                'access_frequency': 0,
                'last_access': time.time(),
                'memory_strength': 1.0,
                
                # Episodic enhancements
                'episodic_context': {
                    'domain': experience.get('domain', 'general'),
                    'source': experience.get('source', 'unknown'),
                    'data_type': experience.get('data_type', 'general'),
                    'processing_time': time.time()
                },
                'episode_boundary_score': 0.8,  # High boundary score for new data
                'cross_episode_similarity': 0.0
            })()
            
            return sensorimotor_exp
            
        except Exception as e:
            logger.error(f"Sensorimotor conversion error: {e}")
            # Return simple fallback object
            return type('FallbackExperience', (), {
                'experience_id': hashlib.md5(f"{experience.get('source', 'unknown')}_{time.time()}".encode()).hexdigest(),
                'content': experience.get('content', ''),
                'domain': experience.get('domain', 'general'),
                'novelty_score': experience.get('novelty_score', 0.5),
                'timestamp': experience.get('timestamp', datetime.now().isoformat()),
                'sensory_features': {'quality': 0.5},
                'motor_actions': [],
                'contextual_embedding': [0.5] * 16,
                'temporal_markers': [time.time()],
                'attention_weights': {'content': 1.0},
                'prediction_targets': {},
                'importance_weight': 0.5,
                'memory_strength': 1.0
            })()

    def _extract_sentiment(self, text: str) -> float:
        """Simple sentiment extraction"""
        positive_words = ['gain', 'rise', 'up', 'bullish', 'growth', 'increase', 'profit', 'good', 'positive']
        negative_words = ['loss', 'fall', 'down', 'bearish', 'decline', 'decrease', 'crash', 'bad', 'negative']
        
        text_lower = str(text).lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        return (positive_count + 0.5) / (positive_count + negative_count + 1)

    def _update_stats(self, raw_count: int, quality_count: int, dedup_count: int, 
                     novel_count: int, experience_count: int, cycle_time: float):
        """Update integration statistics safely"""
        
        try:
            # Calculate duplicates removed
            duplicates_removed = max(0, quality_count - dedup_count)
            
            # Update all stat variations safely
            stats_updates = {
                'raw_data_count': raw_count,
                'raw_data_fetched': raw_count,
                'quality_filtered': quality_count,
                'quality_filtered_count': quality_count,
                'deduplicated': dedup_count,
                'deduplicated_count': dedup_count,
                'dedup_count': dedup_count,
                'duplicates_removed': duplicates_removed,  # Track duplicates removed
                'novel_content': novel_count,
                'novel_content_count': novel_count,
                'novel_count': novel_count,
                'experiences_created': experience_count,
                'experiences_created_count': experience_count,
                'experiences_count': experience_count,
                'total_processed': experience_count,
                'last_cycle_time': cycle_time
            }
            
            # Safely update existing values
            for key, increment in stats_updates.items():
                if key in self.integration_stats:
                    if key in ['last_cycle_time']:
                        # These are direct assignments, not increments
                        self.integration_stats[key] = increment
                    else:
                        # These are incremental
                        self.integration_stats[key] += increment
                else:
                    # Initialize missing keys
                    self.integration_stats[key] = increment
            
            # Update average processing time safely
            if self.integration_stats.get('avg_processing_time', 0) == 0:
                self.integration_stats['avg_processing_time'] = cycle_time
            else:
                self.integration_stats['avg_processing_time'] = (
                    self.integration_stats['avg_processing_time'] * 0.9 + cycle_time * 0.1
                )
            
        except Exception as e:
            logger.error(f"Stats update error: {e}")
            # Ensure we don't crash on stats updates

    # ========================================================================
    # BACKGROUND WORKERS
    # ========================================================================

    def _continuous_fetch_worker(self, domain: str):
        """Background worker for continuous data fetching"""
        logger.info(f"🔄 Starting continuous fetch worker for {domain}")
        
        while not self.shutdown_flag.is_set():
            try:
                # Update stream status
                if domain in self.active_streams:
                    self.active_streams[domain]['last_fetch'] = datetime.now()
                
                # Perform fetch cycle
                result = self.fetch_and_process_cycle(domain, count=5)
                
                if result.get('experiences_created', 0) > 0:
                    logger.info(f"✅ {domain}: {result['experiences_created']} experiences processed")
                    
                    # Update stream data count
                    if domain in self.active_streams:
                        self.active_streams[domain]['data_count'] += result['experiences_created']
                
                # Wait between cycles
                wait_time = 120  # 2 minutes
                for _ in range(wait_time):
                    if self.shutdown_flag.is_set():
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Continuous fetch error for {domain}: {e}")
                
                # Update error count
                if domain in self.active_streams:
                    self.active_streams[domain]['error_count'] += 1
                
                # Wait longer on error
                for _ in range(300):  # 5 minutes
                    if self.shutdown_flag.is_set():
                        break
                    time.sleep(1)
        
        logger.info(f"🛑 Continuous fetch worker for {domain} stopped")

    def _data_processing_worker(self):
        """Background worker for processing queued data"""
        logger.info("🔄 Starting data processing worker")
        
        while not self.shutdown_flag.is_set():
            try:
                # Process items from queue
                processed_count = 0
                
                # Process up to 10 items at once
                for _ in range(10):
                    try:
                        item = self.processing_queue.get(timeout=1)
                        
                        # Process the item
                        self._process_queued_item(item)
                        processed_count += 1
                        
                        self.processing_queue.task_done()
                        
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"Error processing queued item: {e}")
                
                if processed_count > 0:
                    logger.debug(f"✅ Processed {processed_count} queued items")
                
                # Wait before next batch
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Data processing worker error: {e}")
                time.sleep(5)
        
        logger.info("🛑 Data processing worker stopped")

    def _process_queued_item(self, item: Dict[str, Any]):
        """Process a single queued data item with enhanced EMMS integration"""
        try:
            # Convert to experience format
            experience = self._convert_to_experiences([item], item.get('domain', 'general'))[0]
            
            # Process through EMMS with enhanced integration
            if self.memory_system:
                try:
                    # Convert to comprehensive sensorimotor experience
                    sensorimotor_exp = self._convert_to_sensorimotor_experience(experience)
                    
                    # Try multiple EMMS integration paths
                    processed = False
                    
                    # Path 1: Comprehensive processing
                    if hasattr(self.memory_system, 'process_experience_comprehensive'):
                        self.memory_system.process_experience_comprehensive(sensorimotor_exp)
                        processed = True
                    
                    # Path 2: Direct hierarchical memory storage
                    if hasattr(self.memory_system, 'hierarchical_memory') and hasattr(self.memory_system.hierarchical_memory, 'store_experience'):
                        self.memory_system.hierarchical_memory.store_experience(sensorimotor_exp)
                        processed = True
                    
                    # Path 3: Direct memory system storage
                    elif hasattr(self.memory_system, 'store_experience'):
                        self.memory_system.store_experience(sensorimotor_exp)
                        processed = True
                    
                    if processed:
                        self.integration_stats['experiences_processed'] += 1
                        
                        # Force memory consolidation to ensure data flows through hierarchy
                        if hasattr(self.memory_system, 'hierarchical_memory'):
                            if hasattr(self.memory_system.hierarchical_memory, '_check_immediate_consolidation'):
                                self.memory_system.hierarchical_memory._check_immediate_consolidation()
                    else:
                        logger.warning("⚠️ No valid EMMS integration path found")
                        
                except Exception as e:
                    logger.error(f"Enhanced EMMS processing error: {e}")
            
        except Exception as e:
            logger.error(f"Error processing queued item: {e}")

    def _start_binance_websocket(self):
        """Start Binance WebSocket stream in background thread"""
        if not self.binance_client or not HAS_BINANCE:
            return
        
        def websocket_worker():
            try:
                logger.info("🔴 Starting Binance WebSocket stream")
                
                # Create WebSocket manager
                self.binance_websocket = ThreadedWebsocketManager(
                    api_key=self.api_config['binance']['api_key'],
                    api_secret=self.api_config['binance']['api_secret']
                )
                
                def handle_socket_message(msg):
                    try:
                        # Handle both single ticker and ticker array formats
                        if isinstance(msg, list):
                            # Handle ticker array (!ticker@arr stream)
                            for ticker in msg:
                                if isinstance(ticker, dict) and ticker.get('e') == '24hrTicker':
                                    process_single_ticker(ticker)
                        elif isinstance(msg, dict) and msg.get('e') == '24hrTicker':
                            # Handle single ticker
                            process_single_ticker(msg)
                        else:
                            logger.debug(f"Unhandled WebSocket message type: {type(msg)}")
                    
                    except Exception as e:
                        logger.error(f"WebSocket message processing error: {e}")
                
                def process_single_ticker(ticker_msg):
                    """Process individual ticker message with enhanced EMMS integration"""
                    try:
                        # Only process high-volume tickers
                        quote_volume = float(ticker_msg.get('q', 0))
                        if quote_volume > 1000000:  # $1M+ quote volume
                            ticker_data = {
                                'type': 'websocket_ticker',
                                'source': 'binance_websocket',
                                'symbol': ticker_msg.get('s', ''),
                                'price': float(ticker_msg.get('c', 0)),
                                'change': float(ticker_msg.get('P', 0)),
                                'volume': float(ticker_msg.get('v', 0)),
                                'quote_volume': quote_volume,
                                'domain': 'financial_analysis',
                                'timestamp': datetime.now().isoformat(),
                                'quality_score': 0.9,  # High quality for real-time data
                                'novelty_score': 0.8,   # High novelty for live updates
                                'data_type': 'market_ticker'
                            }
                            
                            # Enhanced EMMS integration for WebSocket data
                            if self.memory_system:
                                try:
                                    # Convert to experience format
                                    experience = self._convert_to_experiences([ticker_data], 'financial_analysis')[0]
                                    
                                    # Convert to comprehensive sensorimotor experience  
                                    sensorimotor_exp = self._convert_to_sensorimotor_experience(experience)
                                    
                                    # Process through EMMS with multiple integration paths
                                    processed = False
                                    
                                    if hasattr(self.memory_system, 'process_experience_comprehensive'):
                                        self.memory_system.process_experience_comprehensive(sensorimotor_exp)
                                        processed = True
                                    
                                    if hasattr(self.memory_system, 'hierarchical_memory') and hasattr(self.memory_system.hierarchical_memory, 'store_experience'):
                                        self.memory_system.hierarchical_memory.store_experience(sensorimotor_exp)
                                        processed = True
                                    
                                    if processed:
                                        self.integration_stats['experiences_processed'] += 1
                                        logger.debug(f"📡 WebSocket → EMMS: {ticker_msg.get('s', 'UNKNOWN')} @ ${ticker_msg.get('c', 0)}")
                                        
                                except Exception as e:
                                    logger.error(f"WebSocket EMMS integration error: {e}")
                            
                            # Also add to processing queue as backup
                            try:
                                self.processing_queue.put_nowait(ticker_data)
                            except queue.Full:
                                logger.warning("Processing queue full, dropping ticker data")
                    
                    except Exception as e:
                        logger.error(f"Ticker processing error: {e}")
                
                # Start the WebSocket manager
                self.binance_websocket.start()
                
                # Start ticker stream
                self.binance_websocket.start_ticker_socket(callback=handle_socket_message)
                
                # Keep the thread alive
                while not self.shutdown_flag.is_set():
                    time.sleep(1)
                
                # Stop WebSocket
                self.binance_websocket.stop()
                logger.info("✅ Binance WebSocket stopped")
                
            except Exception as e:
                logger.error(f"❌ Binance WebSocket error: {e}")
        
        # Start WebSocket in background thread
        ws_thread = threading.Thread(target=websocket_worker, name="BinanceWebSocket", daemon=True)
        ws_thread.start()
        self.background_threads.append(ws_thread)

    def _check_rate_limit(self, source_name: str, limit_per_minute: int) -> bool:
        """Check if we can make a request within rate limits"""
        current_time = time.time()
        rate_info = self.rate_limiters[source_name]
        
        # Reset counter if minute has passed
        if current_time - rate_info['reset_time'] >= 60:
            rate_info['count'] = 0
            rate_info['reset_time'] = current_time
        
        # Check if under limit
        if rate_info['count'] < limit_per_minute:
            rate_info['count'] += 1
            return True
        
        return False

    # ========================================================================
    # ADDITIONAL EMMS COMPATIBILITY METHODS
    # ========================================================================

    def process_data(self, data: Any) -> Dict[str, Any]:
        """Process single data item"""
        try:
            if isinstance(data, dict):
                return {
                    'content': str(data.get('content', '')),
                    'domain': data.get('domain', 'general'),
                    'source': data.get('source', 'unknown'),
                    'timestamp': data.get('timestamp', datetime.now().isoformat()),
                    'novelty_score': data.get('novelty_score', 0.5),
                    'data_type': data.get('data_type', 'general')
                }
            else:
                return {
                    'content': str(data),
                    'domain': 'general',
                    'source': 'processed',
                    'timestamp': datetime.now().isoformat(),
                    'novelty_score': 0.5,
                    'data_type': 'processed'
                }
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            return {}

    def get_data_sources(self) -> Dict[str, Any]:
        """Get data sources"""
        return {name: {
            'name': source.name,
            'type': source.source_type,
            'endpoint': source.endpoint,
            'enabled': source.enabled,
            'priority': source.priority,
            'last_fetch': source.last_fetch.isoformat() if source.last_fetch else None,
            'error_count': source.error_count
        } for name, source in self.data_sources.items()}

    def get_status(self) -> Dict[str, Any]:
        """Get integrator status"""
        return {
            'status': 'operational' if not self.shutdown_flag.is_set() else 'stopped',
            'active_streams': len(self.active_streams),
            'background_threads': len([t for t in self.background_threads if t.is_alive()]),
            'total_fetched': self.integration_stats['total_fetched'],
            'processed_count': self.integration_stats['total_processed'],
            'experiences_processed': self.integration_stats['experiences_processed'],
            'queue_size': self.processing_queue.qsize(),
            'cache_size': len(self.data_cache),
            'errors': self.integration_stats['errors'],
            'avg_processing_time': self.integration_stats['avg_processing_time']
        }

    def reset_statistics(self):
        """Reset all statistics"""
        self.integration_stats = {
            'total_fetched': 0, 
            'total_processed': 0, 
            'quality_filtered': 0, 
            'quality_filtered_count': 0,
            'duplicates_removed': 0,  # Fix: Include duplicates_removed
            'dedup_count': 0, 
            'deduplicated': 0,  # Fix: Include deduplicated
            'deduplicated_count': 0,  # Fix: Include deduplicated_count
            'novel_content': 0, 
            'novel_content_count': 0,
            'novel_count': 0, 
            'experiences_created': 0, 
            'experiences_created_count': 0,
            'experiences_count': 0, 
            'raw_data_count': 0, 
            'raw_data_fetched': 0,
            'streams_active': len(self.active_streams), 
            'articles_per_minute': 0,
            'errors': 0, 
            'experiences_processed': 0, 
            'last_cycle_time': 0.0, 
            'avg_processing_time': 0.0
        }

    def get_available_domains(self) -> List[str]:
        """Get available domains"""
        return ['financial_analysis', 'research', 'market_data', 'news', 'blockchain', 'defi']

    def configure_domain(self, domain: str, config: Dict[str, Any]):
        """Configure domain settings"""
        logger.info(f"Configuring domain: {domain}")
        # Domain configuration can be stored and used for customization

    def get_integration_history(self) -> List[Dict[str, Any]]:
        """Get integration history"""
        return list(self.integration_history)

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics for EMMS - with proper nested structure"""
        
        # Core integration statistics that EMMS expects
        integration_stats = {
            # Core statistics
            'total_fetched': self.integration_stats['total_fetched'],
            'total_processed': self.integration_stats['total_processed'],
            'experiences_processed': self.integration_stats['experiences_processed'],
            'errors': self.integration_stats['errors'],
            
            # Processing pipeline statistics
            'raw_data_fetched': self.integration_stats['raw_data_fetched'],
            'quality_filtered': self.integration_stats['quality_filtered'],
            'quality_filtered_count': self.integration_stats['quality_filtered_count'],
            'deduplicated': self.integration_stats['deduplicated'],
            'deduplicated_count': self.integration_stats['deduplicated_count'],
            'dedup_count': self.integration_stats['dedup_count'],
            'duplicates_removed': self.integration_stats.get('duplicates_removed', 0),  # Fix: Add missing key
            'novel_content': self.integration_stats['novel_content'],
            'novel_content_count': self.integration_stats['novel_content_count'],
            'novel_count': self.integration_stats['novel_count'],
            'experiences_created': self.integration_stats['experiences_created'],
            'experiences_created_count': self.integration_stats['experiences_created_count'],
            'experiences_count': self.integration_stats['experiences_count'],
            
            # Performance metrics
            'avg_processing_time': self.integration_stats['avg_processing_time'],
            'last_cycle_time': self.integration_stats['last_cycle_time'],
            'articles_per_minute': self._calculate_articles_per_minute(),
            
            # System status
            'streams_active': len(self.active_streams),
            'background_threads_active': len([t for t in self.background_threads if t.is_alive()]),
            'queue_size': self.processing_queue.qsize(),
            'cache_size': len(self.data_cache),
            'uptime_seconds': time.time() - self._start_time,
            'last_update': datetime.now().isoformat()
        }
        
        # Return in the nested structure EMMS expects
        return {
            'integration_stats': integration_stats,
            'active_streams_detail': {
                domain: {
                    'status': info.get('status', 'unknown'),
                    'data_count': info.get('data_count', 0),
                    'error_count': info.get('error_count', 0),
                    'last_fetch': info.get('last_fetch').isoformat() if info.get('last_fetch') else None,
                    'started_at': info.get('started_at').isoformat() if info.get('started_at') else None
                }
                for domain, info in self.active_streams.items()
            },
            'api_status': self._get_api_status_summary(),
            'performance_metrics': {
                'avg_processing_time': self.integration_stats['avg_processing_time'],
                'articles_per_minute': self._calculate_articles_per_minute(),
                'processing_efficiency': self._calculate_processing_efficiency(),
                'uptime_seconds': time.time() - self._start_time
            }
        }

    def _calculate_processing_efficiency(self) -> float:
        """Calculate processing efficiency (experiences per second)"""
        try:
            uptime = time.time() - self._start_time
            if uptime > 0:
                return self.integration_stats['experiences_processed'] / uptime
            return 0.0
        except Exception:
            return 0.0

    def _calculate_articles_per_minute(self) -> float:
        """Calculate articles processed per minute"""
        try:
            uptime = time.time() - self._start_time
            if uptime > 0:
                return (self.integration_stats['experiences_processed'] / uptime) * 60
            return 0.0
        except Exception:
            return 0.0

    def _get_api_status_summary(self) -> Dict[str, str]:
        """Get a summary of API connection status"""
        try:
            connections = self.validate_api_connections()
            return {
                'total_apis': len(connections),
                'connected': len([status for status in connections.values() if status == 'CONNECTED']),
                'failed': len([status for status in connections.values() if 'ERROR' in status]),
                'not_available': len([status for status in connections.values() if 'NOT_AVAILABLE' in status])
            }
        except Exception:
            return {'total_apis': 0, 'connected': 0, 'failed': 0, 'not_available': 0}

    def clear_cache(self):
        """Clear data cache"""
        self.data_cache.clear()
        logger.info("✅ Data cache cleared")

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        alive_threads = [t for t in self.background_threads if t.is_alive()]
        
        return {
            'status': 'healthy' if not self.shutdown_flag.is_set() else 'shutdown',
            'active_streams': len(self.active_streams),
            'alive_threads': len(alive_threads),
            'total_threads': len(self.background_threads),
            'queue_size': self.processing_queue.qsize(),
            'cache_size': len(self.data_cache),
            'errors': self.integration_stats['errors'],
            'uptime_seconds': time.time() - self._start_time,
            'last_update': datetime.now().isoformat()
        }

    def validate_api_connections(self) -> Dict[str, str]:
        """Validate API connections"""
        results = {}
        
        # Test Binance
        if self.binance_client:
            try:
                self.binance_client.get_server_time()
                results['binance'] = 'CONNECTED'
            except Exception as e:
                results['binance'] = f'ERROR: {str(e)[:50]}'
        else:
            results['binance'] = 'NOT_AVAILABLE'
        
        # Test Web3
        if self.web3_client:
            try:
                self.web3_client.eth.block_number
                results['ethereum'] = 'CONNECTED'
            except Exception as e:
                results['ethereum'] = f'ERROR: {str(e)[:50]}'
        else:
            results['ethereum'] = 'NOT_AVAILABLE'
        
        # Test HTTP APIs
        test_urls = {
            'coingecko': 'https://api.coingecko.com/api/v3/ping'
        }
        
        for name, url in test_urls.items():
            try:
                response = self.session.get(url, timeout=10)
                results[name] = 'CONNECTED' if response.status_code == 200 else f'HTTP_{response.status_code}'
            except Exception as e:
                results[name] = f'ERROR: {str(e)[:50]}'
        
        return results

# ============================================================================
# CONTENT QUALITY AND DEDUPLICATION SYSTEMS
# ============================================================================

class ContentQualityAssessor:
    """Assess content quality using multiple metrics"""
    
    def __init__(self):
        self.quality_metrics = [
            'information_density',
            'readability',
            'completeness',
            'factual_consistency',
            'source_credibility'
        ]
        
        self.assessment_history = deque(maxlen=1000)
    
    def assess_content_quality(self, content_item: Dict[str, Any]) -> float:
        """Comprehensive content quality assessment"""
        
        content = content_item.get('content', '')
        source = content_item.get('source', '')
        
        if not content:
            return 0.0
        
        # Calculate individual quality metrics
        quality_scores = {}
        
        quality_scores['information_density'] = self._assess_information_density(content)
        quality_scores['readability'] = self._assess_readability(content)
        quality_scores['completeness'] = self._assess_completeness(content)
        quality_scores['factual_consistency'] = self._assess_factual_consistency(content)
        quality_scores['source_credibility'] = self._assess_source_credibility(source)
        
        # Weighted quality score
        weights = {
            'information_density': 0.25,
            'readability': 0.15,
            'completeness': 0.25,
            'factual_consistency': 0.20,
            'source_credibility': 0.15
        }
        
        overall_quality = sum(quality_scores[metric] * weights[metric] 
                             for metric in self.quality_metrics)
        
        # Record assessment
        self.assessment_history.append({
            'timestamp': time.time(),
            'content_length': len(content),
            'quality_scores': quality_scores,
            'overall_quality': overall_quality
        })
        
        return overall_quality
    
    def _assess_information_density(self, content: str) -> float:
        """Assess information density of content"""
        
        words = content.split()
        
        if not words:
            return 0.0
        
        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)
        
        # Content length factor (optimal around 100-500 words)
        length_factor = min(1.0, len(words) / 100.0) if len(words) < 500 else max(0.5, 500.0 / len(words))
        
        # Information-bearing word ratio
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        info_words = [word for word in words if word.lower() not in stop_words and len(word) > 3]
        info_ratio = len(info_words) / len(words)
        
        density_score = (unique_ratio * 0.4 + length_factor * 0.3 + info_ratio * 0.3)
        
        return density_score
    
    def _assess_readability(self, content: str) -> float:
        """Assess readability using simplified metrics"""
        
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Optimal ranges
        sentence_score = 1.0 - abs(avg_sentence_length - 15) / 20.0  # Optimal around 15 words
        word_score = 1.0 - abs(avg_word_length - 6) / 4.0  # Optimal around 6 characters
        
        readability_score = max(0.0, (sentence_score + word_score) / 2.0)
        
        return readability_score
    
    def _assess_completeness(self, content: str) -> float:
        """Assess content completeness"""
        
        # Check for complete sentences
        sentence_endings = ['.', '!', '?']
        has_proper_ending = any(content.strip().endswith(ending) for ending in sentence_endings)
        
        # Check for minimum content length
        min_length_met = len(content) >= 50
        
        # Check for informational elements
        has_numbers = any(char.isdigit() for char in content)
        has_proper_nouns = any(word[0].isupper() for word in content.split()[1:])  # Skip first word
        
        completeness_factors = [
            has_proper_ending,
            min_length_met,
            has_numbers,
            has_proper_nouns
        ]
        
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        return completeness_score
    
    def _assess_factual_consistency(self, content: str) -> float:
        """Assess factual consistency (simplified)"""
        
        # Look for contradictory statements (simplified heuristics)
        content_lower = content.lower()
        
        # Check for hedge words (indicate uncertainty, which is good for factual consistency)
        hedge_words = ['may', 'might', 'could', 'possibly', 'likely', 'appears', 'seems', 'suggests']
        hedge_count = sum(1 for word in hedge_words if word in content_lower)
        
        # Check for absolute statements (which may be less reliable)
        absolute_words = ['always', 'never', 'all', 'none', 'every', 'only', 'definitely']
        absolute_count = sum(1 for word in absolute_words if word in content_lower)
        
        words = content.split()
        hedge_ratio = hedge_count / max(len(words), 1)
        absolute_ratio = absolute_count / max(len(words), 1)
        
        # Moderate hedging is good, excessive absolutes are bad
        consistency_score = min(1.0, hedge_ratio * 10) - min(0.5, absolute_ratio * 10)
        consistency_score = max(0.3, consistency_score + 0.7)  # Base score of 0.7
        
        return consistency_score
    
    def _assess_source_credibility(self, source: str) -> float:
        """Assess source credibility"""
        
        if not source:
            return 0.5
        
        source_lower = source.lower()
        
        # High credibility sources
        high_credibility = [
            'reuters', 'bloomberg', 'ap news', 'associated press', 'bbc', 'cnn',
            'yahoo finance', 'marketwatch', 'coingecko', 'arxiv', 'nature', 'science'
        ]
        
        # Medium credibility sources
        medium_credibility = [
            'techcrunch', 'wired', 'ars technica', 'cnbc', 'forbes', 'wall street journal'
        ]
        
        # Check for high credibility
        for source_name in high_credibility:
            if source_name in source_lower:
                return 0.9
        
        # Check for medium credibility
        for source_name in medium_credibility:
            if source_name in source_lower:
                return 0.7
        
        # Check for API sources
        if 'api' in source_lower:
            return 0.8
        
        # Check for RSS feeds
        if 'rss' in source_lower:
            return 0.6
        
        return 0.5  # Default credibility
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quality assessment statistics"""
        
        if not self.assessment_history:
            return {'status': 'no_assessments'}
        
        quality_scores = [assessment['overall_quality'] for assessment in self.assessment_history]
        content_lengths = [assessment['content_length'] for assessment in self.assessment_history]
        
        return {
            'total_assessments': len(self.assessment_history),
            'avg_quality_score': np.mean(quality_scores),
            'quality_std': np.std(quality_scores),
            'avg_content_length': np.mean(content_lengths),
            'high_quality_ratio': len([q for q in quality_scores if q > 0.7]) / len(quality_scores)
        }

class ContentDeduplicator:
    """Remove duplicate content using multiple techniques"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.content_hashes = set()
        self.content_fingerprints = {}
        self.deduplication_stats = {
            'total_processed': 0,
            'duplicates_removed': 0,
            'hash_matches': 0,
            'similarity_matches': 0
        }
    
    def remove_duplicates(self, content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates using multiple techniques"""
        
        if not content_items:
            return []
        
        deduplicated_items = []
        
        for item in content_items:
            content = item.get('content', '')
            
            if not content:
                continue
            
            self.deduplication_stats['total_processed'] += 1
            
            # Method 1: Exact hash matching
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash in self.content_hashes:
                self.deduplication_stats['duplicates_removed'] += 1
                self.deduplication_stats['hash_matches'] += 1
                continue
            
            # Method 2: Semantic similarity
            is_similar = self._check_semantic_similarity(content)
            
            if is_similar:
                self.deduplication_stats['duplicates_removed'] += 1
                self.deduplication_stats['similarity_matches'] += 1
                continue
            
            # Not a duplicate - add to results
            self.content_hashes.add(content_hash)
            self._add_content_fingerprint(content)
            deduplicated_items.append(item)
        
        return deduplicated_items
    
    def _check_semantic_similarity(self, content: str) -> bool:
        """Check for semantic similarity with existing content"""
        
        content_fingerprint = self._create_content_fingerprint(content)
        
        for existing_fingerprint in self.content_fingerprints.values():
            similarity = self._calculate_fingerprint_similarity(content_fingerprint, existing_fingerprint)
            
            if similarity > self.similarity_threshold:
                return True
        
        return False
    
    def _create_content_fingerprint(self, content: str) -> Dict[str, Any]:
        """Create content fingerprint for similarity comparison"""
        
        words = content.lower().split()
        
        # Word frequency fingerprint
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top frequent words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # N-gram fingerprint (2-grams)
        bigrams = []
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                bigrams.append(f"{words[i]}_{words[i+1]}")
        
        bigram_freq = {}
        for bigram in bigrams:
            bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
        
        top_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fingerprint = {
            'length': len(content),
            'word_count': len(words),
            'top_words': [word for word, freq in top_words],
            'top_bigrams': [bigram for bigram, freq in top_bigrams],
            'word_frequencies': dict(top_words),
            'bigram_frequencies': dict(top_bigrams)
        }
        
        return fingerprint
    
    def _add_content_fingerprint(self, content: str):
        """Add content fingerprint to storage"""
        
        content_hash = hashlib.md5(content.encode()).hexdigest()
        fingerprint = self._create_content_fingerprint(content)
        
        self.content_fingerprints[content_hash] = fingerprint
        
        # Limit fingerprint storage
        if len(self.content_fingerprints) > 1000:
            # Remove oldest fingerprints (simplified - in practice would use LRU)
            oldest_hashes = list(self.content_fingerprints.keys())[:100]
            for old_hash in oldest_hashes:
                del self.content_fingerprints[old_hash]
    
    def _calculate_fingerprint_similarity(self, fp1: Dict[str, Any], fp2: Dict[str, Any]) -> float:
        """Calculate similarity between content fingerprints"""
        
        similarities = []
        
        # Length similarity
        len1, len2 = fp1['length'], fp2['length']
        length_sim = 1.0 - abs(len1 - len2) / max(len1, len2)
        similarities.append(length_sim)
        
        # Word overlap similarity
        words1 = set(fp1['top_words'])
        words2 = set(fp2['top_words'])
        
        if words1 or words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
            similarities.append(word_overlap)
        
        # Bigram overlap similarity
        bigrams1 = set(fp1['top_bigrams'])
        bigrams2 = set(fp2['top_bigrams'])
        
        if bigrams1 or bigrams2:
            bigram_overlap = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
            similarities.append(bigram_overlap)
        
        # Word frequency similarity
        freq_sim = self._calculate_frequency_similarity(fp1['word_frequencies'], fp2['word_frequencies'])
        similarities.append(freq_sim)
        
        # Overall similarity
        overall_similarity = np.mean(similarities) if similarities else 0.0
        
        return overall_similarity
    
    def _calculate_frequency_similarity(self, freq1: Dict[str, int], freq2: Dict[str, int]) -> float:
        """Calculate similarity between frequency distributions"""
        
        all_words = set(freq1.keys()) | set(freq2.keys())
        
        if not all_words:
            return 0.0
        
        # Create frequency vectors
        vec1 = np.array([freq1.get(word, 0) for word in all_words])
        vec2 = np.array([freq2.get(word, 0) for word in all_words])
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        
        return cosine_sim
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        
        return {
            'total_processed': self.deduplication_stats['total_processed'],
            'duplicates_removed': self.deduplication_stats['duplicates_removed'],
            'hash_matches': self.deduplication_stats['hash_matches'],
            'similarity_matches': self.deduplication_stats['similarity_matches'],
            'deduplication_rate': self.deduplication_stats['duplicates_removed'] / max(self.deduplication_stats['total_processed'], 1),
            'unique_hashes_stored': len(self.content_hashes),
            'fingerprints_stored': len(self.content_fingerprints),
            'similarity_threshold': self.similarity_threshold
        }

class NoveltyDetector:
    """Detect novel content using temporal and semantic analysis"""
    
    def __init__(self, novelty_threshold: float = 0.4):
        self.novelty_threshold = novelty_threshold
        self.historical_content = deque(maxlen=500)  # Keep recent content for comparison
        self.topic_tracking = defaultdict(list)
        self.novelty_stats = {
            'total_processed': 0,
            'novel_content_detected': 0,
            'avg_novelty_score': 0.0
        }
    
    def filter_novel_content(self, content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter content to return only novel items"""
        
        novel_items = []
        
        for item in content_items:
            novelty_score = self.calculate_novelty_score(item)
            
            self.novelty_stats['total_processed'] += 1
            
            if novelty_score >= self.novelty_threshold:
                item['calculated_novelty_score'] = novelty_score
                novel_items.append(item)
                self.novelty_stats['novel_content_detected'] += 1
            
            # Update novelty stats
            current_avg = self.novelty_stats['avg_novelty_score']
            total_processed = self.novelty_stats['total_processed']
            self.novelty_stats['avg_novelty_score'] = (current_avg * (total_processed - 1) + novelty_score) / total_processed
            
            # Add to historical content
            self._add_to_historical_content(item)
        
        return novel_items
    
    def calculate_novelty_score(self, content_item: Dict[str, Any]) -> float:
        """Calculate comprehensive novelty score"""
        
        content = content_item.get('content', '')
        category = content_item.get('category', 'general')
        timestamp = content_item.get('timestamp', datetime.now().isoformat())
        
        if not content:
            return 0.0
        
        novelty_factors = {}
        
        # Temporal novelty
        novelty_factors['temporal'] = self._calculate_temporal_novelty(timestamp)
        
        # Semantic novelty
        novelty_factors['semantic'] = self._calculate_semantic_novelty(content)
        
        # Topic novelty
        novelty_factors['topic'] = self._calculate_topic_novelty(content, category)
        
        # Keyword novelty
        novelty_factors['keyword'] = self._calculate_keyword_novelty(content)
        
        # Structural novelty
        novelty_factors['structural'] = self._calculate_structural_novelty(content)
        
        # Weighted combination
        weights = {
            'temporal': 0.15,
            'semantic': 0.35,
            'topic': 0.25,
            'keyword': 0.15,
            'structural': 0.10
        }
        
        overall_novelty = sum(novelty_factors[factor] * weights[factor] 
                             for factor in novelty_factors.keys())
        
        return overall_novelty
    
    def _calculate_temporal_novelty(self, timestamp: str) -> float:
        """Calculate temporal novelty based on recency"""
        
        try:
            content_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            current_time = datetime.now(content_time.tzinfo)
            
            time_diff = abs((current_time - content_time).total_seconds())
            
            # Content is more novel when it's more recent
            # Full novelty for content < 1 hour old, decreasing to 0.1 for content > 24 hours old
            temporal_novelty = max(0.1, np.exp(-time_diff / 14400))  # 4-hour half-life
            
            return temporal_novelty
            
        except:
            return 0.5  # Default if timestamp parsing fails
    
    def _calculate_semantic_novelty(self, content: str) -> float:
        """Calculate semantic novelty compared to historical content"""
        
        if not self.historical_content:
            return 0.8  # High novelty if no historical content
        
        content_words = set(content.lower().split())
        
        # Calculate similarity with recent historical content
        similarities = []
        
        for historical_item in list(self.historical_content)[-50:]:  # Compare with last 50 items
            historical_content = historical_item.get('content', '')
            historical_words = set(historical_content.lower().split())
            
            if content_words and historical_words:
                intersection = len(content_words & historical_words)
                union = len(content_words | historical_words)
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        if not similarities:
            return 0.8
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities)
        semantic_novelty = 1.0 - max_similarity
        
        return semantic_novelty
    
    def _calculate_topic_novelty(self, content: str, category: str) -> float:
        """Calculate topic novelty within category"""
        
        if category not in self.topic_tracking:
            self.topic_tracking[category] = []
            return 0.9  # High novelty for new category
        
        # Extract topic keywords
        topic_keywords = self._extract_topic_keywords(content)
        
        # Compare with historical topics in the same category
        category_history = self.topic_tracking[category]
        
        if not category_history:
            return 0.8
        
        topic_similarities = []
        
        for historical_keywords in category_history[-20:]:  # Last 20 topics
            if topic_keywords and historical_keywords:
                intersection = len(set(topic_keywords) & set(historical_keywords))
                union = len(set(topic_keywords) | set(historical_keywords))
                similarity = intersection / union if union > 0 else 0
                topic_similarities.append(similarity)
        
        if not topic_similarities:
            return 0.8
        
        # Topic novelty is inverse of maximum topic similarity
        max_topic_similarity = max(topic_similarities)
        topic_novelty = 1.0 - max_topic_similarity
        
        # Update topic tracking
        self.topic_tracking[category].append(topic_keywords)
        
        # Limit topic history
        if len(self.topic_tracking[category]) > 100:
            self.topic_tracking[category] = self.topic_tracking[category][-100:]
        
        return topic_novelty
    
    def _extract_topic_keywords(self, content: str) -> List[str]:
        """Extract topic-relevant keywords from content"""
        
        words = content.lower().split()
        
        # Filter for significant words
        topic_keywords = []
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for word in words:
            if (len(word) > 4 and 
                word not in stop_words and 
                word.isalpha() and 
                not word.isdigit()):
                topic_keywords.append(word)
        
        # Return top frequent keywords
        word_freq = {}
        for word in topic_keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def _calculate_keyword_novelty(self, content: str) -> float:
        """Calculate novelty based on unique keywords"""
        
        words = content.lower().split()
        
        # Extract significant keywords
        keywords = [word for word in words if len(word) > 4 and word.isalpha()]
        
        if not keywords:
            return 0.5
        
        # Check how many keywords are "rare" (not seen frequently in historical content)
        rare_keywords = 0
        
        for keyword in keywords:
            # Count occurrences in historical content
            historical_count = 0
            for historical_item in self.historical_content:
                historical_content = historical_item.get('content', '').lower()
                historical_count += historical_content.count(keyword)
            
            # Keyword is rare if it appears less than 3 times in history
            if historical_count < 3:
                rare_keywords += 1
        
        # Novelty based on ratio of rare keywords
        keyword_novelty = rare_keywords / len(keywords)
        
        return keyword_novelty
    
    def _calculate_structural_novelty(self, content: str) -> float:
        """Calculate structural novelty of content"""
        
        # Analyze content structure
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 0.5
        
        # Structure features
        avg_sentence_length = len(words) / len(sentences)
        punctuation_density = sum(1 for char in content if char in '.,!?;:') / len(content)
        capitalization_ratio = sum(1 for char in content if char.isupper()) / len(content)
        
        # Compare with historical structural patterns
        if not self.historical_content:
            return 0.6
        
        historical_structures = []
        
        for historical_item in list(self.historical_content)[-20:]:
            hist_content = historical_item.get('content', '')
            if hist_content:
                hist_sentences = hist_content.split('.')
                hist_words = hist_content.split()
                
                if hist_sentences and hist_words:
                    hist_avg_sent_len = len(hist_words) / len(hist_sentences)
                    hist_punct_density = sum(1 for char in hist_content if char in '.,!?;:') / len(hist_content)
                    hist_cap_ratio = sum(1 for char in hist_content if char.isupper()) / len(hist_content)
                    
                    historical_structures.append({
                        'avg_sentence_length': hist_avg_sent_len,
                        'punctuation_density': hist_punct_density,
                        'capitalization_ratio': hist_cap_ratio
                    })
        
        if not historical_structures:
            return 0.6
        
        # Calculate structural distance from historical patterns
        structural_distances = []
        
        current_structure = {
            'avg_sentence_length': avg_sentence_length,
            'punctuation_density': punctuation_density,
            'capitalization_ratio': capitalization_ratio
        }
        
        for hist_struct in historical_structures:
            distance = 0
            distance += abs(current_structure['avg_sentence_length'] - hist_struct['avg_sentence_length']) / 20
            distance += abs(current_structure['punctuation_density'] - hist_struct['punctuation_density']) * 10
            distance += abs(current_structure['capitalization_ratio'] - hist_struct['capitalization_ratio']) * 5
            
            structural_distances.append(distance)
        
        # Structural novelty based on minimum distance
        min_distance = min(structural_distances)
        structural_novelty = min(1.0, min_distance)
        
        return structural_novelty
    
    def _add_to_historical_content(self, content_item: Dict[str, Any]):
        """Add content item to historical tracking"""
        
        self.historical_content.append({
            'content': content_item.get('content', ''),
            'timestamp': content_item.get('timestamp', datetime.now().isoformat()),
            'category': content_item.get('category', 'general')
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get novelty detection statistics"""
        
        return {
            'total_processed': self.novelty_stats['total_processed'],
            'novel_content_detected': self.novelty_stats['novel_content_detected'],
            'novelty_detection_rate': self.novelty_stats['novel_content_detected'] / max(self.novelty_stats['total_processed'], 1),
            'avg_novelty_score': self.novelty_stats['avg_novelty_score'],
            'novelty_threshold': self.novelty_threshold,
            'historical_content_size': len(self.historical_content),
            'topics_tracked': len(self.topic_tracking)
        }

# ============================================================================
# MAIN DEMONSTRATION AND TESTING
# ============================================================================

def run_complete_gem2_demonstration():
    """Run the complete EMMS.py demonstration with all components"""
    
    print("🚀 Complete EMMS.py Enhanced Memory System Demonstration")
    print("=" * 80)
    print("🧠 Features: Token Management + Graph Boundaries + Hierarchical Memory")
    print("🔄 + Compression + Multi-Strategy Retrieval + Cross-Modal + Real-time Data")
    print()
    
    # Initialize the complete system
    memory_system = EnhancedIntegratedMemorySystem(
        domain="financial_analysis",
        model_architecture="gemma3n:e4b"
    )
    
    # Initialize real-time data integrator
    api_keys = {
        'newsapi': 'demo_key',  # Would be real API keys in production
        'alphavantage': 'demo_key'
    }
    
    data_integrator = RealTimeDataIntegrator(memory_system, api_keys)
    
    print("🌐 Starting Real-time Data Integration Demo")
    print("-" * 50)
    
    # Start continuous integration
    integration_start = data_integrator.start_continuous_integration(['financial_analysis', 'research'])
    print(f"✅ Initialized {integration_start['streams_initialized']} data streams")
    print()
    
    # Run multiple fetch and process cycles
    domains_to_test = ['financial_analysis', 'research']
    
    for cycle in range(3):
        print(f"📊 Data Integration Cycle {cycle + 1}")
        print("-" * 30)
        
        for domain in domains_to_test:
            print(f"Processing {domain} domain...")
            
            cycle_result = data_integrator.fetch_and_process_cycle(domain, count=15)
            
            print(f"   Raw data fetched: {cycle_result['raw_data_count']}")
            print(f"   Quality filtered: {cycle_result['quality_filtered_count']}")
            print(f"   Deduplicated: {cycle_result['deduplicated_count']}")
            print(f"   Novel content: {cycle_result['novel_count']}")
            print(f"   Experiences created: {cycle_result['experiences_created']}")
            print(f"   Cycle time: {cycle_result['cycle_time']:.2f}s")
            print()
        
        # Small delay between cycles
        time.sleep(1)
    
    print("🔍 Comprehensive Memory Retrieval Test")
    print("-" * 40)
    
    # Create test query
    test_query = create_test_experience(
        "What are the latest developments in cryptocurrency markets and AI research?",
        "financial_analysis"
    )
    
    # Comprehensive retrieval
    retrieval_result = memory_system.retrieve_comprehensive(test_query, max_results=15)
    
    print(f"Retrieved {len(retrieval_result['final_results'])} memories:")
    print(f"   Hierarchical: {retrieval_result['hierarchical_count']}")
    print(f"   Cross-modal: {retrieval_result['cross_modal_count']}")
    print(f"   Advanced: {retrieval_result['advanced_count']}")
    print(f"   Retrieval time: {retrieval_result['retrieval_time']:.3f}s")
    print()
    
    # Display top results
    print("Top Retrieved Memories:")
    for i, memory in enumerate(retrieval_result['final_results'][:5]):
        sources = ', '.join(memory['sources'])
        score = memory['ensemble_score']
        print(f"   {i+1}. Score: {score:.3f} | Sources: {sources}")
    print()
    
    # System statistics
    print("📈 Complete System Statistics")
    print("-" * 40)
    
    memory_stats = memory_system.get_comprehensive_statistics()
    integration_stats = data_integrator.get_integration_statistics()
    
    # Memory system stats
    print("Enhanced Memory System:")
    print(f"   Experiences processed: {memory_stats['integration_stats']['experiences_processed']}")
    print(f"   Average processing time: {memory_stats['integration_stats']['avg_processing_time']:.3f}s")
    print(f"   Processing efficiency: {memory_stats['integration_stats']['processing_efficiency']:.2f} exp/sec")
    print()
    
    # Component details
    component_stats = memory_stats['component_stats']
    
    print("Memory Components:")
    print(f"   Working memory: {component_stats['hierarchical_memory']['working_memory']['count']}/7")
    print(f"   Short-term memory: {component_stats['hierarchical_memory']['short_term_memory']['count']}/50")
    print(f"   Long-term memory: {component_stats['hierarchical_memory']['long_term_memory']['count']}")
    print(f"   Semantic concepts: {component_stats['hierarchical_memory']['semantic_memory']['concepts']}")
    print()
    
    print("Token Management:")
    print(f"   Context utilization: {component_stats['token_management']['utilization_ratio']:.2%}")
    print(f"   Evicted tokens: {component_stats['token_management']['evicted_tokens']}")
    print(f"   Unique evicted: {component_stats['token_management']['unique_evicted_tokens']}")
    print()
    
    print("Cross-Modal System:")
    modal_stats = component_stats['cross_modal_system']['modal_index_stats']
    for modality, stats_data in modal_stats.items():
        if stats_data['stored_experiences'] > 0:
            print(f"   {modality}: {stats_data['stored_experiences']} experiences")
    print()
    
    # Data integration stats
    print("Real-time Data Integration:")
    data_stats = integration_stats['integration_stats']
    print(f"   Total fetched: {data_stats['total_fetched']}")
    print(f"   Total processed: {data_stats['total_processed']}")
    print(f"   Quality filtered: {data_stats['quality_filtered']}")
    print(f"   Duplicates removed: {data_stats['duplicates_removed']}")
    print(f"   Active streams: {data_stats['streams_active']}")
    print()
    
    # Performance analysis
    print("📊 Performance Analysis")
    print("-" * 30)
    
    processing_efficiency = memory_stats['integration_stats']['processing_efficiency']
    token_utilization = component_stats['token_management']['utilization_ratio']
    memory_utilization = (
        component_stats['hierarchical_memory']['working_memory']['utilization'] +
        component_stats['hierarchical_memory']['short_term_memory']['utilization']
    ) / 2
    
    print(f"Overall System Performance:")
    print(f"   Processing efficiency: {processing_efficiency:.2f} exp/sec")
    print(f"   Token utilization: {token_utilization:.1%}")
    print(f"   Memory utilization: {memory_utilization:.1%}")
    
    if processing_efficiency > 1.0:
        print("   ✅ HIGH PERFORMANCE - System processing efficiently")
    elif processing_efficiency > 0.5:
        print("   ⚠️  MODERATE PERFORMANCE - Room for optimization")
    else:
        print("   ❌ LOW PERFORMANCE - Needs optimization")
    
    print()
    
    # Cleanup
    memory_system.shutdown()
    
    print("✅ Complete EMMS.py Demonstration Finished!")
    print("\n🏆 Successfully Demonstrated:")
    print("   ✓ Token-level context management with intelligent eviction")
    print("   ✓ Graph-theoretic boundary refinement with multiple algorithms")
    print("   ✓ Hierarchical memory system with automatic consolidation")
    print("   ✓ Memory compression with pattern detection and abstractions")
    print("   ✓ Multi-strategy retrieval with ensemble methods")
    print("   ✓ Cross-modal memory integration across 6 modalities")
    print("   ✓ Real-time data integration with quality assessment")
    print("   ✓ Content deduplication and novelty detection")
    print("   ✓ Comprehensive system integration and coordination")
    print()
    print("🧠 This represents the most advanced memory management system")
    print("   for persistent identity AI, combining cutting-edge techniques")
    print("   from neuroscience, cognitive science, and computer science.")
# ============================================================================
# DEMO AND TESTING FUNCTIONS
# ============================================================================

def create_test_experience(content: str, domain: str = "general") -> SensorimotorExperience:
    """Create test experience for demonstration"""
    
    return SensorimotorExperience(
        experience_id=f"test_{uuid.uuid4().hex[:8]}",
        content=content,
        domain=domain,
        timestamp=datetime.now().isoformat(),
        novelty_score=random.uniform(0.3, 0.9),
        sensory_features={'complexity': random.uniform(0.4, 0.8)},
        motor_actions=['analyze', 'process', 'respond'],
        contextual_embedding=np.random.rand(20),
        temporal_markers=[time.time()],
        attention_weights={'content': 0.8, 'novelty': 0.6},
        prediction_targets={'next_content': 0.7},
        emotional_features={'valence': random.uniform(-0.5, 0.5), 'arousal': random.uniform(0.2, 0.8)},
        causal_indicators=['because', 'resulting in'] if 'because' in content else [],
        goal_relevance={'learning': 0.8, 'analysis': 0.7},
        importance_weight=random.uniform(0.5, 0.9),
        memory_strength=random.uniform(0.7, 1.0)
    )

def run_enhanced_memory_demo():
    """Run comprehensive demonstration of enhanced memory system"""
    
    print("🧠 Enhanced Memory Management System - Comprehensive Demo")
    print("=" * 80)
    
    # Initialize system
    memory_system = EnhancedIntegratedMemorySystem(
        domain="financial_analysis",
        model_architecture="gemma3n:e4b"
    )
    
    # Test experiences
    test_experiences = [
        create_test_experience("Bitcoin price surges 15% following institutional adoption news", "financial_analysis"),
        create_test_experience("Federal Reserve announces interest rate decision affecting market volatility", "financial_analysis"),
        create_test_experience("Machine learning breakthrough in natural language processing published", "research"),
        create_test_experience("Tesla reports record quarterly earnings with strong growth in energy division", "financial_analysis"),
        create_test_experience("New AI model demonstrates superior performance in complex reasoning tasks", "research"),
        create_test_experience("Cryptocurrency regulations proposed by European Union affect trading volumes", "financial_analysis"),
        create_test_experience("Quantum computing research achieves milestone in error correction", "research"),
        create_test_experience("Stock market shows resilience despite geopolitical tensions", "financial_analysis"),
    ]
    
    print(f"Processing {len(test_experiences)} test experiences...")
    print()
    
    # Process experiences
    processing_results = []
    
    for i, experience in enumerate(test_experiences):
        print(f"📝 Processing Experience {i+1}: {experience.content[:50]}...")
        
        result = memory_system.process_experience_comprehensive(experience)
        processing_results.append(result)
        
        print(f"   ⏱️  Processing time: {result['total_processing_time']:.3f}s")
        print(f"   🧮 Token processing: {result['token_management']['context_tokens']} tokens")
        print(f"   🏛️  Hierarchical storage: {result['hierarchical_storage']['storage_level']}")
        print(f"   🌐 Cross-modal: {result.get('cross_modal_processing', {}).get('associations_created', 'N/A')}")
        print()
    
    # Test comprehensive retrieval
    print("🔍 Testing Comprehensive Retrieval")
    print("-" * 40)
    
    query_experience = create_test_experience("What are the latest developments in cryptocurrency markets?", "financial_analysis")
    
    retrieval_result = memory_system.retrieve_comprehensive(query_experience, max_results=10)
    
    print(f"Retrieved {len(retrieval_result['final_results'])} relevant memories:")
    print(f"   Hierarchical: {retrieval_result['hierarchical_count']}")
    print(f"   Cross-modal: {retrieval_result['cross_modal_count']}")
    print(f"   Advanced: {retrieval_result['advanced_count']}")
    print(f"   Retrieval time: {retrieval_result['retrieval_time']:.3f}s")
    print()
    
    # Display top retrieved memories
    print("Top Retrieved Memories:")
    for i, memory in enumerate(retrieval_result['final_results'][:3]):
        sources = ', '.join(memory['sources'])
        score = memory['ensemble_score']
        print(f"   {i+1}. Score: {score:.3f} | Sources: {sources}")
    print()
    
    # System statistics
    print("📊 Comprehensive System Statistics")
    print("-" * 40)
    
    stats = memory_system.get_comprehensive_statistics()
    
    integration_stats = stats['integration_stats']
    print(f"Session ID: {integration_stats['session_id']}")
    print(f"Experiences processed: {integration_stats['experiences_processed']}")
    print(f"Average processing time: {integration_stats['avg_processing_time']:.3f}s")
    print(f"Processing efficiency: {integration_stats['processing_efficiency']:.2f} exp/sec")
    print()
    
    component_stats = stats['component_stats']
    
    print("Component Statistics:")
    print(f"   Token Management:")
    print(f"     Context utilization: {component_stats['token_management']['utilization_ratio']:.2%}")
    print(f"     Evicted tokens: {component_stats['token_management']['evicted_tokens']}")
    
    print(f"   Hierarchical Memory:")
    print(f"     Working memory: {component_stats['hierarchical_memory']['working_memory']['count']}/7")
    print(f"     Short-term memory: {component_stats['hierarchical_memory']['short_term_memory']['count']}/50")
    print(f"     Long-term memory: {component_stats['hierarchical_memory']['long_term_memory']['count']}")
    print(f"     Semantic concepts: {component_stats['hierarchical_memory']['semantic_memory']['concepts']}")
    
    print(f"   Cross-Modal System:")
    modal_stats = component_stats['cross_modal_system']['modal_index_stats']
    for modality, stats_data in modal_stats.items():
        print(f"     {modality}: {stats_data['stored_experiences']} experiences")
    
    print()
    
    # Shutdown
    memory_system.shutdown()
    
    print("✅ Enhanced Memory System Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("   ✓ Token-level context management with intelligent eviction")
    print("   ✓ Graph-theoretic boundary refinement")
    print("   ✓ Hierarchical memory with consolidation")
    print("   ✓ Memory compression with pattern detection")
    print("   ✓ Multi-strategy advanced retrieval")
    print("   ✓ Cross-modal memory integration")
    print("   ✓ Comprehensive ensemble retrieval")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🎯 EMMS.py - Enhanced Memory Management System")
    print("Choose demonstration mode:")
    print("1. Complete system demonstration (recommended)")
    print("2. Memory components only")
    print("3. Real-time integration only")
    print("4. Performance testing")
    
    choice = input("Choice (1-4): ").strip()
    
    if choice == "1" or choice == "":
        run_complete_gem2_demonstration()
    elif choice == "2":
        run_enhanced_memory_demo()  # From the paste-3.txt implementation
    elif choice == "3":
        # Real-time integration demo
        memory_system = EnhancedIntegratedMemorySystem("general", "gemma3n:e4b")
        data_integrator = RealTimeDataIntegrator(memory_system)
        
        print("🌐 Real-time Data Integration Demo")
        result = data_integrator.fetch_and_process_cycle("financial_analysis", 20)
        print(f"Processed {result['experiences_created']} experiences in {result['cycle_time']:.2f}s")
        
        memory_system.shutdown()
    elif choice == "4":
        # Performance testing
        print("🔬 Performance Testing Mode")
        
        memory_system = EnhancedIntegratedMemorySystem("financial_analysis", "gemma3n:e4b")
        
        start_time = time.time()
        
        # Process many experiences rapidly
        for i in range(50):
            exp = create_test_experience(f"Performance test experience {i+1} with financial data analysis", "financial_analysis")
            memory_system.process_experience_comprehensive(exp)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/50 experiences...")
        
        total_time = time.time() - start_time
        
        stats = memory_system.get_comprehensive_statistics()
        
        print(f"\n📊 Performance Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Experiences/second: {50/total_time:.2f}")
        print(f"   Average processing time: {stats['integration_stats']['avg_processing_time']:.3f}s")
        
        memory_system.shutdown()
    else:
        print("Invalid choice. Running complete demonstration...")
        run_complete_gem2_demonstration()