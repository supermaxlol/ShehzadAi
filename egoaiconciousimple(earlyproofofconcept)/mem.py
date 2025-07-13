#!/usr/bin/env python3
"""
Enhanced Persistent Identity AI with Advanced Memory Management
Comprehensive implementation with token-level management, graph-theoretic refinement,
hierarchical memory, compression, and cross-modal integration.

Author: Advanced AI Research Team
Version: 2.0 - Enhanced Memory Architecture
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
# CORE DATA STRUCTURES AND INTERFACES
# ============================================================================

@dataclass
class EnhancedSensorimotorExperience:
    """Enhanced experience representation with multi-modal features"""
    experience_id: str
    content: str
    domain: str
    timestamp: str
    novelty_score: float
    
    # Multi-modal features
    sensory_features: Dict[str, Any]
    motor_actions: List[str]
    contextual_embedding: np.ndarray
    temporal_markers: List[float]
    attention_weights: Dict[str, float]
    prediction_targets: Dict[str, float]
    
    # Enhanced features
    emotional_features: Dict[str, float] = field(default_factory=dict)
    causal_indicators: List[str] = field(default_factory=list)
    goal_relevance: Dict[str, float] = field(default_factory=dict)
    modality_features: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Memory metadata
    importance_weight: float = 0.5
    access_frequency: int = 0
    last_access: float = field(default_factory=time.time)
    memory_strength: float = 1.0

@dataclass
class CompressedMemoryBlock:
    """Compressed memory representation"""
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
    """Represents a level in the memory hierarchy"""
    level_name: str
    capacity: int
    retention_policy: str
    consolidation_threshold: float
    access_frequency_weight: float
    importance_weight: float

class MemoryRetrievalStrategy(ABC):
    """Abstract base class for memory retrieval strategies"""
    
    @abstractmethod
    def retrieve(self, query_experience: EnhancedSensorimotorExperience, 
                memory_store: Dict, max_results: int = 20) -> List[Dict]:
        pass
    
    @abstractmethod
    def calculate_relevance_score(self, query: EnhancedSensorimotorExperience, 
                                candidate: Dict) -> float:
        pass

# ============================================================================
# TOKEN-LEVEL CONTEXT MANAGEMENT (EM-LLM STYLE)
# ============================================================================

class TokenLevelContextManager:
    """Advanced token-level context management inspired by EM-LLM"""
    
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
# ADVANCED GRAPH-THEORETIC BOUNDARY REFINEMENT
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
        """Calculate comprehensive boundary quality metrics"""
        
        metrics = {}
        
        try:
            # Modularity
            communities = self.episodes_to_communities(episodes)
            if len(communities) > 1:
                metrics['modularity'] = nx.algorithms.community.modularity(G, communities)
            else:
                metrics['modularity'] = 0.0
                
            # Average clustering coefficient
            metrics['clustering_coefficient'] = nx.average_clustering(G)
            
            # Number of boundaries
            boundary_count = sum(1 for ep in episodes if ep.get('is_boundary', False))
            metrics['boundary_count'] = boundary_count
            metrics['boundary_density'] = boundary_count / len(episodes) if episodes else 0
            
            # Graph connectivity metrics
            if nx.is_connected(G):
                metrics['diameter'] = nx.diameter(G)
                metrics['average_shortest_path'] = nx.average_shortest_path_length(G)
            else:
                metrics['diameter'] = float('inf')
                metrics['average_shortest_path'] = float('inf')
            
            # Edge density
            metrics['edge_density'] = nx.density(G)
            
        except Exception as e:
            logger.warning(f"Error calculating boundary metrics: {e}")
            metrics = {'error': str(e)}
        
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
# HIERARCHICAL MEMORY SYSTEM
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
    
    def store_experience(self, experience: EnhancedSensorimotorExperience) -> Dict[str, Any]:
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
    
    def retrieve_memories(self, query_experience: EnhancedSensorimotorExperience, 
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
    
    def _create_memory_item(self, experience: EnhancedSensorimotorExperience) -> Dict[str, Any]:
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
    
    def _search_memory_level(self, query_experience: EnhancedSensorimotorExperience, 
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
    
    def _search_semantic_memory(self, query_experience: EnhancedSensorimotorExperience) -> List[Dict[str, Any]]:
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
    
    def _calculate_memory_relevance(self, query: EnhancedSensorimotorExperience, 
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
    
    def _extract_concepts(self, experience: EnhancedSensorimotorExperience) -> List[str]:
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
# MEMORY COMPRESSION SYSTEM
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
    
    def decompress_memory_block(self, compressed_block: CompressedMemoryBlock) -> List[Dict]:
        """Decompress memory block back to episodes"""
        
        try:
            # Deserialize compressed data
            compressed_data = self._deserialize_compressed_data(compressed_block.compressed_content)
            
            # Reconstruct episodes using patterns and abstractions
            reconstructed_episodes = self._reconstruct_from_compressed_data(
                compressed_data, 
                compressed_block.compression_metadata,
                compressed_block.abstraction_levels
            )
            
            return reconstructed_episodes
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return []
    
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
    
    def _deserialize_compressed_data(self, compressed_bytes: bytes) -> Dict[str, Any]:
        """Deserialize compressed data from bytes"""
        
        try:
            decompressed_bytes = gzip.decompress(compressed_bytes)
            compressed_data = pickle.loads(decompressed_bytes)
            return compressed_data
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            return {}
    
    def _reconstruct_from_compressed_data(self, compressed_data: Dict[str, Any], 
                                        metadata: Dict[str, Any],
                                        abstractions: Dict[str, Any]) -> List[Dict]:
        """Reconstruct episodes from compressed data"""
        
        reconstructed_episodes = []
        
        pattern_substitutions = compressed_data.get('pattern_substitutions', {})
        compressed_episodes = compressed_data.get('compressed_episodes', [])
        
        # Create reverse substitution map
        reverse_substitutions = {symbol: pattern for pattern, symbol in pattern_substitutions.items()}
        
        for compressed_episode in compressed_episodes:
            # Reconstruct content
            content = compressed_episode.get('content', '')
            for symbol, pattern in reverse_substitutions.items():
                content = content.replace(symbol, pattern)
            
            # Reconstruct episode
            reconstructed_episode = {
                'content': content,
                'domain': compressed_episode.get('domain', ''),
                'novelty_score': compressed_episode.get('novelty_score', 0.5),
                'timestamp': compressed_episode.get('timestamp', ''),
                'representative_tokens': compressed_episode.get('tokens', []),
                'reconstructed': True
            }
            
            reconstructed_episodes.append(reconstructed_episode)
        
        return reconstructed_episodes
    
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
# ADVANCED RETRIEVAL STRATEGIES
# ============================================================================

class SemanticSimilarityStrategy(MemoryRetrievalStrategy):
    """Semantic similarity-based retrieval"""
    
    def retrieve(self, query_experience: EnhancedSensorimotorExperience, 
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
    
    def calculate_relevance_score(self, query: EnhancedSensorimotorExperience, 
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
    
    def retrieve(self, query_experience: EnhancedSensorimotorExperience, 
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
    
    def calculate_relevance_score(self, query: EnhancedSensorimotorExperience, 
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
    
    def retrieve(self, query_experience: EnhancedSensorimotorExperience, 
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
    
    def calculate_relevance_score(self, query: EnhancedSensorimotorExperience, 
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
    
    def _extract_emotional_features(self, experience: EnhancedSensorimotorExperience) -> Dict[str, float]:
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
    
    def retrieve(self, query_experience: EnhancedSensorimotorExperience, 
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
    
    def calculate_relevance_score(self, query: EnhancedSensorimotorExperience, 
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
    
    def multi_strategy_retrieval(self, query_experience: EnhancedSensorimotorExperience, 
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
                                 query_experience: EnhancedSensorimotorExperience) -> List[Dict]:
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
                                  query_experience: EnhancedSensorimotorExperience) -> List[Dict]:
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
    
    def _record_retrieval_statistics(self, query_experience: EnhancedSensorimotorExperience,
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
# CROSS-MODAL MEMORY INTEGRATION
# ============================================================================

class CrossModalMemorySystem:
    """Cross-modal memory integration system"""
    
    def __init__(self):
        self.modalities = ['text', 'visual', 'audio', 'temporal', 'spatial', 'emotional']
        self.cross_modal_graph = nx.MultiGraph()
        self.modal_indices = {modality: {} for modality in self.modalities}
        self.association_strength_threshold = 0.6
        
    def store_cross_modal_experience(self, experience: EnhancedSensorimotorExperience) -> Dict[str, Any]:
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
    
    def retrieve_cross_modal(self, query_experience: EnhancedSensorimotorExperience, 
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
    
    def _extract_modal_features(self, experience: EnhancedSensorimotorExperience, 
                               modality: str) -> Optional[np.ndarray]:
        """Extract features for specific modality"""
        
        if modality == 'text':
            return self._extract_text_features(experience)
        elif modality == 'temporal':
            return self._extract_temporal_features(experience)
        elif modality == 'spatial':
            return self._extract_spatial_features(experience)
        elif modality == 'emotional':
            return self._extract_emotional_features_array(experience)
        elif modality == 'visual':
            return self._extract_visual_features(experience)
        elif modality == 'audio':
            return self._extract_audio_features(experience)
        
        return None
    
    def _extract_text_features(self, experience: EnhancedSensorimotorExperience) -> np.ndarray:
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
    
    def _extract_temporal_features(self, experience: EnhancedSensorimotorExperience) -> np.ndarray:
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
    
    def _extract_spatial_features(self, experience: EnhancedSensorimotorExperience) -> np.ndarray:
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
    
    def _extract_emotional_features_array(self, experience: EnhancedSensorimotorExperience) -> np.ndarray:
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
    
    def _extract_visual_features(self, experience: EnhancedSensorimotorExperience) -> np.ndarray:
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
    
    def _extract_audio_features(self, experience: EnhancedSensorimotorExperience) -> np.ndarray:
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
    
    def _calculate_association_strength(self, features1: np.ndarray, 
                                      features2: np.ndarray) -> float:
        """Calculate association strength between modality features"""
        
        try:
            # Normalize features
            norm1 = features1 / (np.linalg.norm(features1) + 1e-10)
            norm2 = features2 / (np.linalg.norm(features2) + 1e-10)
            
            # Calculate correlation
            correlation = np.corrcoef(norm1, norm2)[0, 1]
            
            # Handle NaN
            if np.isnan(correlation):
                correlation = 0.0
            
            # Convert to positive association strength
            association_strength = abs(correlation)
            
            return association_strength
            
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
        
        for experience_id, stored_data in self.modal_indices[modality].items():
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
        
        # Check for strong cross-modal associations
        for edge in self.cross_modal_graph.edges(data=True):
            edge_data = edge[2]
            
            if (edge_data.get('type') == 'cross_modal_association' and 
                edge_data.get('experience') == experience_id):
                
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
    
    def process_experience_comprehensive(self, experience: EnhancedSensorimotorExperience) -> Dict[str, Any]:
        """Comprehensive experience processing with all memory enhancements"""
        
        start_time = time.time()
        processing_results = {}
        
        # Step 1: Token-level processing
        token_processing_start = time.time()
        experience_tokens = self._experience_to_tokens(experience)
        token_context = self.token_manager.process_tokens_with_memory(
            experience_tokens, 
            experience_context={'content': experience.content, 'domain': experience.domain}
        )
        
        processing_results['token_processing'] = {
            'input_tokens': len(experience_tokens),
            'context_tokens': len(token_context),
            'processing_time': time.time() - token_processing_start
        }
        
        # Step 2: Store in hierarchical memory
        hierarchical_start = time.time()
        storage_result = self.hierarchical_memory.store_experience(experience)
        
        processing_results['hierarchical_storage'] = storage_result
        processing_results['hierarchical_storage']['processing_time'] = time.time() - hierarchical_start
        
        # Step 3: Cross-modal storage
        cross_modal_start = time.time()
        cross_modal_result = self.cross_modal_system.store_cross_modal_experience(experience)
        
        processing_results['cross_modal_storage'] = cross_modal_result
        processing_results['cross_modal_storage']['processing_time'] = time.time() - cross_modal_start
        
        # Step 4: Comprehensive retrieval
        retrieval_start = time.time()
        
        # Hierarchical retrieval
        hierarchical_memories = self.hierarchical_memory.retrieve_memories(experience, max_memories=10)
        
        # Cross-modal retrieval
        cross_modal_memories = self.cross_modal_system.retrieve_cross_modal(experience, max_results=10)
        
        # Advanced multi-strategy retrieval (on hierarchical long-term memory)
        if self.hierarchical_memory.long_term_memory:
            advanced_memories = self.retrieval_system.multi_strategy_retrieval(
                experience, self.hierarchical_memory.long_term_memory, max_results=10
            )
        else:
            advanced_memories = []
        
        processing_results['memory_retrieval'] = {
            'hierarchical_memories': len(hierarchical_memories),
            'cross_modal_memories': len(cross_modal_memories),
            'advanced_memories': len(advanced_memories),
            'processing_time': time.time() - retrieval_start
        }
        
        # Step 5: Boundary refinement (if enough experiences)
        boundary_start = time.time()
        
        # Collect recent experiences for boundary analysis
        recent_experiences = self._get_recent_experiences_for_boundary_analysis()
        
        if len(recent_experiences) >= 5:
            refined_boundaries, boundary_metrics = self.boundary_refiner.refine_boundaries_with_graph_metrics(
                recent_experiences
            )
            
            processing_results['boundary_refinement'] = {
                'experiences_analyzed': len(recent_experiences),
                'boundary_metrics': boundary_metrics,
                'processing_time': time.time() - boundary_start
            }
        else:
            processing_results['boundary_refinement'] = {
                'status': 'insufficient_experiences',
                'processing_time': time.time() - boundary_start
            }
        
        # Step 6: Compression (periodic)
        compression_start = time.time()
        
        if len(recent_experiences) >= 10 and self.integration_stats['experiences_processed'] % 20 == 0:
            compression_result = self._periodic_compression(recent_experiences)
            processing_results['compression'] = compression_result
        else:
            processing_results['compression'] = {'status': 'not_triggered'}
        
        processing_results['compression']['processing_time'] = time.time() - compression_start
        
        # Step 7: Update integration statistics
        total_processing_time = time.time() - start_time
        self._update_integration_statistics(processing_results, total_processing_time)
        
        processing_results['total_processing_time'] = total_processing_time
        processing_results['experience_id'] = experience.experience_id
        
        return processing_results
    
    def retrieve_comprehensive(self, query_experience: EnhancedSensorimotorExperience, 
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
    
    def _experience_to_tokens(self, experience: EnhancedSensorimotorExperience) -> List[str]:
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
                              query_experience: EnhancedSensorimotorExperience) -> List[Dict]:
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
# DEMO AND TESTING FUNCTIONS
# ============================================================================

def create_test_experience(content: str, domain: str = "general") -> EnhancedSensorimotorExperience:
    """Create test experience for demonstration"""
    
    return EnhancedSensorimotorExperience(
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
        print(f"   🧮 Token processing: {result['token_processing']['context_tokens']} tokens")
        print(f"   🏛️  Hierarchical storage: {result['hierarchical_storage']['storage_level']}")
        print(f"   🌐 Cross-modal: {result['cross_modal_storage']['modalities_stored']}")
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
    # Run the comprehensive demonstration
    run_enhanced_memory_demo()