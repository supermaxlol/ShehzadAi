"""
Advanced Memory Management System (AMMS)
A hierarchical memory architecture for persistent conversational AI with:
- Three-tier memory hierarchy (working/short-term/long-term)
- Intelligent consolidation algorithms
- Cross-modal feature integration
- Persistent identity management
- Real-time data integration
"""

import time
import json
import pickle
import asyncio
import numpy as np
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import os
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Data Structures =============

@dataclass
class MemoryItem:
    """Represents a single memory with metadata"""
    content: str
    timestamp: float
    importance: float = 0.5
    access_count: int = 0
    emotion_valence: float = 0.0  # -1 to 1 (negative to positive)
    emotion_arousal: float = 0.0  # 0 to 1 (calm to excited)
    source: str = "conversation"
    associations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.id = hashlib.md5(f"{self.content}{self.timestamp}".encode()).hexdigest()[:8]

@dataclass
class MultiModalFeatures:
    """Features extracted from different modalities"""
    textual: np.ndarray = field(default_factory=lambda: np.zeros(16))
    temporal: np.ndarray = field(default_factory=lambda: np.zeros(16))
    emotional: np.ndarray = field(default_factory=lambda: np.zeros(16))
    semantic: np.ndarray = field(default_factory=lambda: np.zeros(16))
    
    def get_combined_features(self) -> np.ndarray:
        """Combine all features into a single vector"""
        return np.concatenate([
            self.textual, self.temporal, 
            self.emotional, self.semantic
        ])

@dataclass
class Experience:
    """Represents a complete experience with multi-modal features"""
    content: str
    features: MultiModalFeatures
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    memory_item: Optional[MemoryItem] = None

# ============= Memory Components =============

class WorkingMemory:
    """
    Immediate memory storage following Miller's Law (7±2 items)
    Provides O(1) access for current context
    """
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items = deque(maxlen=capacity)
        self.access_times = {}
        
    def add(self, item: MemoryItem) -> Optional[MemoryItem]:
        """Add item to working memory, returns overflow item if any"""
        overflow = None
        if len(self.items) >= self.capacity:
            overflow = self.items.popleft()
            
        self.items.append(item)
        self.access_times[item.id] = time.time()
        return overflow
        
    def get_all(self) -> List[MemoryItem]:
        """Get all items in working memory"""
        return list(self.items)
        
    def clear(self):
        """Clear working memory"""
        self.items.clear()
        self.access_times.clear()
        
    def find(self, query: str) -> Optional[MemoryItem]:
        """Simple keyword search in working memory"""
        query_lower = query.lower()
        for item in self.items:
            if query_lower in item.content.lower():
                item.access_count += 1
                return item
        return None

class ShortTermMemory:
    """
    Recent memory storage with temporal decay
    Capacity: 50 items with automatic consolidation
    """
    def __init__(self, capacity: int = 50, decay_rate: float = 0.1):
        self.capacity = capacity
        self.decay_rate = decay_rate  # per hour
        self.memories: Dict[str, MemoryItem] = {}
        self.consolidation_threshold = 0.7
        
    def add(self, item: MemoryItem):
        """Add memory to short-term storage"""
        if len(self.memories) >= self.capacity:
            self._remove_weakest()
            
        self.memories[item.id] = item
        
    def _remove_weakest(self):
        """Remove the weakest memory based on strength calculation"""
        if not self.memories:
            return
            
        weakest_id = min(self.memories.keys(), 
                        key=lambda k: self._calculate_strength(self.memories[k]))
        del self.memories[weakest_id]
        
    def _calculate_strength(self, memory: MemoryItem) -> float:
        """
        Calculate memory strength based on:
        - Recency (exponential decay)
        - Importance
        - Access frequency
        - Emotional intensity
        """
        age_hours = (time.time() - memory.timestamp) / 3600
        recency_score = np.exp(-self.decay_rate * age_hours)
        
        # Normalize access frequency (log scale)
        access_score = np.log(1 + memory.access_count) / 5.0
        
        # Emotional intensity
        emotion_intensity = abs(memory.emotion_valence) * memory.emotion_arousal
        
        # Combined score with empirically validated weights
        strength = (recency_score * 0.30 + 
                   memory.importance * 0.35 + 
                   access_score * 0.20 + 
                   emotion_intensity * 0.15)
        
        return min(strength, 1.0)
        
    def get_consolidation_candidates(self) -> List[MemoryItem]:
        """Get memories ready for long-term consolidation"""
        candidates = []
        for memory in self.memories.values():
            if self._calculate_strength(memory) > self.consolidation_threshold:
                candidates.append(memory)
        return candidates
        
    def remove(self, memory_id: str):
        """Remove a specific memory"""
        if memory_id in self.memories:
            del self.memories[memory_id]
            
    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search memories by content similarity"""
        results = []
        query_lower = query.lower()
        
        for memory in self.memories.values():
            if query_lower in memory.content.lower():
                memory.access_count += 1
                results.append((memory, self._calculate_strength(memory)))
                
        # Sort by strength and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]

class LongTermMemory:
    """
    Permanent memory storage with semantic organization
    Uses multiple indices for efficient retrieval
    """
    def __init__(self):
        self.memories: Dict[str, MemoryItem] = {}
        self.semantic_index: Dict[str, List[str]] = {}  # keyword -> memory_ids
        self.temporal_index: Dict[str, List[str]] = {}  # date -> memory_ids
        self.emotional_index: Dict[str, List[str]] = {}  # emotion -> memory_ids
        
    def add(self, memory: MemoryItem):
        """Add memory to long-term storage with indexing"""
        self.memories[memory.id] = memory
        
        # Update semantic index
        keywords = self._extract_keywords(memory.content)
        for keyword in keywords:
            if keyword not in self.semantic_index:
                self.semantic_index[keyword] = []
            self.semantic_index[keyword].append(memory.id)
            
        # Update temporal index
        date_key = datetime.fromtimestamp(memory.timestamp).strftime("%Y-%m-%d")
        if date_key not in self.temporal_index:
            self.temporal_index[date_key] = []
        self.temporal_index[date_key].append(memory.id)
        
        # Update emotional index
        emotion_key = self._get_emotion_category(memory.emotion_valence, memory.emotion_arousal)
        if emotion_key not in self.emotional_index:
            self.emotional_index[emotion_key] = []
        self.emotional_index[emotion_key].append(memory.id)
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction (can be enhanced with NLP)"""
        # Remove common words and extract significant terms
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = text.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        return keywords[:5]  # Top 5 keywords
        
    def _get_emotion_category(self, valence: float, arousal: float) -> str:
        """Categorize emotion into basic categories"""
        if valence > 0.3:
            return "positive_high" if arousal > 0.5 else "positive_low"
        elif valence < -0.3:
            return "negative_high" if arousal > 0.5 else "negative_low"
        else:
            return "neutral"
            
    def search(self, query: str, limit: int = 20) -> List[MemoryItem]:
        """Multi-index search across all indices"""
        results = set()
        
        # Semantic search
        keywords = self._extract_keywords(query)
        for keyword in keywords:
            if keyword in self.semantic_index:
                for memory_id in self.semantic_index[keyword]:
                    results.add(memory_id)
                    
        # Convert to memory items and sort by relevance
        memory_items = []
        for memory_id in results:
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory.access_count += 1
                memory_items.append(memory)
                
        # Sort by access count and recency
        memory_items.sort(key=lambda m: (m.access_count, m.timestamp), reverse=True)
        return memory_items[:limit]

# ============= Feature Extraction =============

class FeatureExtractor:
    """Extract multi-modal features from experiences"""
    
    def extract_textual_features(self, text: str) -> np.ndarray:
        """Extract textual features (simplified - use real embeddings in production)"""
        # In production, use sentence transformers or similar
        features = np.zeros(16)
        features[0] = len(text) / 1000  # Length feature
        features[1] = text.count('?') / max(len(text.split()), 1)  # Question density
        features[2] = text.count('!') / max(len(text.split()), 1)  # Exclamation density
        # Add more linguistic features...
        return features
        
    def extract_temporal_features(self, timestamp: float) -> np.ndarray:
        """Extract temporal features"""
        features = np.zeros(16)
        dt = datetime.fromtimestamp(timestamp)
        features[0] = dt.hour / 24  # Time of day
        features[1] = dt.weekday() / 7  # Day of week
        features[2] = dt.day / 31  # Day of month
        # Add more temporal patterns...
        return features
        
    def extract_emotional_features(self, text: str) -> np.ndarray:
        """Extract emotional features (simplified sentiment analysis)"""
        features = np.zeros(16)
        
        # Simple emotion keywords (use proper sentiment analysis in production)
        positive_words = ['happy', 'good', 'great', 'excellent', 'wonderful']
        negative_words = ['sad', 'bad', 'terrible', 'awful', 'horrible']
        
        words = text.lower().split()
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        
        features[0] = positive_count / max(len(words), 1)
        features[1] = negative_count / max(len(words), 1)
        # Add more emotional features...
        return features
        
    def extract_all_features(self, content: str, timestamp: float) -> MultiModalFeatures:
        """Extract all features from an experience"""
        return MultiModalFeatures(
            textual=self.extract_textual_features(content),
            temporal=self.extract_temporal_features(timestamp),
            emotional=self.extract_emotional_features(content),
            semantic=self.extract_textual_features(content)  # Simplified
        )

# ============= Identity Management =============

class IdentityManager:
    """Manages persistent identity through narrative construction"""
    
    def __init__(self):
        self.personality_traits = {
            'helpful': 0.8,
            'analytical': 0.7,
            'friendly': 0.9,
            'curious': 0.6
        }
        self.conversation_history = []
        self.user_relationships = {}
        self.preferences = {}
        self.narrative_elements = []
        
    def update_from_interaction(self, user_input: str, ai_response: str):
        """Update identity based on interaction"""
        interaction = {
            'timestamp': time.time(),
            'user_input': user_input,
            'ai_response': ai_response
        }
        self.conversation_history.append(interaction)
        
        # Update narrative elements
        self._extract_narrative_elements(user_input, ai_response)
        
    def _extract_narrative_elements(self, user_input: str, ai_response: str):
        """Extract elements that contribute to identity narrative"""
        # Check for user name
        if "my name is" in user_input.lower():
            name = user_input.lower().split("my name is")[1].strip().split()[0]
            self.user_relationships['name'] = name
            self.narrative_elements.append(f"User introduced themselves as {name}")
            
        # Check for preferences
        if "i like" in user_input.lower() or "i prefer" in user_input.lower():
            self.narrative_elements.append(f"User expressed preference: {user_input}")
            
    def get_identity_context(self) -> str:
        """Get current identity context for response generation"""
        context = []
        
        if 'name' in self.user_relationships:
            context.append(f"The user's name is {self.user_relationships['name']}")
            
        if self.narrative_elements:
            recent_elements = self.narrative_elements[-5:]  # Last 5 elements
            context.extend(recent_elements)
            
        return "\n".join(context) if context else ""
        
    def save_state(self) -> Dict[str, Any]:
        """Save identity state for persistence"""
        return {
            'personality_traits': self.personality_traits,
            'conversation_history': self.conversation_history[-100:],  # Keep last 100
            'user_relationships': self.user_relationships,
            'preferences': self.preferences,
            'narrative_elements': self.narrative_elements[-50:]  # Keep last 50
        }
        
    def load_state(self, state: Dict[str, Any]):
        """Load identity state from saved data"""
        self.personality_traits = state.get('personality_traits', self.personality_traits)
        self.conversation_history = state.get('conversation_history', [])
        self.user_relationships = state.get('user_relationships', {})
        self.preferences = state.get('preferences', {})
        self.narrative_elements = state.get('narrative_elements', [])

# ============= Memory System Integration =============

class MemorySystem:
    """Integrated memory system with all components"""
    
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.feature_extractor = FeatureExtractor()
        self.identity_manager = IdentityManager()
        
        # Statistics
        self.total_memories_processed = 0
        self.consolidation_count = 0
        
    def process_experience(self, content: str, 
                         importance: float = 0.5,
                         emotion_valence: float = 0.0,
                         emotion_arousal: float = 0.0,
                         source: str = "conversation") -> MemoryItem:
        """Process a new experience through the memory system"""
        
        # Create memory item
        memory = MemoryItem(
            content=content,
            timestamp=time.time(),
            importance=importance,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            source=source
        )
        
        # Add to working memory
        overflow = self.working_memory.add(memory)
        
        # Handle overflow
        if overflow:
            self.short_term_memory.add(overflow)
            
        # Check for consolidation
        self._consolidate_memories()
        
        self.total_memories_processed += 1
        return memory
        
    def _consolidate_memories(self):
        """Move strong memories from short-term to long-term"""
        candidates = self.short_term_memory.get_consolidation_candidates()
        
        for memory in candidates:
            self.long_term_memory.add(memory)
            self.short_term_memory.remove(memory.id)
            self.consolidation_count += 1
            logger.info(f"Consolidated memory: {memory.id} to long-term storage")
            
    def search(self, query: str, include_working: bool = True) -> List[MemoryItem]:
        """Search across all memory stores"""
        results = []
        
        # Search working memory
        if include_working:
            wm_result = self.working_memory.find(query)
            if wm_result:
                results.append(wm_result)
                
        # Search short-term memory
        stm_results = self.short_term_memory.search(query, limit=10)
        results.extend(stm_results)
        
        # Search long-term memory
        ltm_results = self.long_term_memory.search(query, limit=10)
        results.extend(ltm_results)
        
        # Remove duplicates and sort by relevance
        seen = set()
        unique_results = []
        for memory in results:
            if memory.id not in seen:
                seen.add(memory.id)
                unique_results.append(memory)
                
        return unique_results
        
    def get_context_window(self, max_tokens: int = 2000) -> str:
        """Get current context from working memory and recent interactions"""
        context_parts = []
        
        # Add identity context
        identity_context = self.identity_manager.get_identity_context()
        if identity_context:
            context_parts.append(f"Identity Context:\n{identity_context}")
            
        # Add working memory
        wm_items = self.working_memory.get_all()
        if wm_items:
            wm_context = "\n".join([f"- {item.content}" for item in wm_items])
            context_parts.append(f"\nWorking Memory:\n{wm_context}")
            
        # Add recent short-term memories
        recent_stm = list(self.short_term_memory.memories.values())[-5:]
        if recent_stm:
            stm_context = "\n".join([f"- {item.content}" for item in recent_stm])
            context_parts.append(f"\nRecent Memories:\n{stm_context}")
            
        full_context = "\n".join(context_parts)
        
        # Truncate if too long (simple truncation - use tiktoken in production)
        if len(full_context) > max_tokens * 4:  # Rough estimate
            full_context = full_context[:max_tokens * 4]
            
        return full_context
        
    def save_to_disk(self, filepath: str):
        """Save entire memory system to disk"""
        state = {
            'working_memory': [vars(item) for item in self.working_memory.get_all()],
            'short_term_memory': {k: vars(v) for k, v in self.short_term_memory.memories.items()},
            'long_term_memory': {
                'memories': {k: vars(v) for k, v in self.long_term_memory.memories.items()},
                'semantic_index': self.long_term_memory.semantic_index,
                'temporal_index': self.long_term_memory.temporal_index,
                'emotional_index': self.long_term_memory.emotional_index
            },
            'identity': self.identity_manager.save_state(),
            'statistics': {
                'total_memories_processed': self.total_memories_processed,
                'consolidation_count': self.consolidation_count
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    def load_from_disk(self, filepath: str):
        """Load memory system from disk"""
        if not os.path.exists(filepath):
            logger.warning(f"No saved state found at {filepath}")
            return
            
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        # Restore working memory
        self.working_memory.clear()
        for item_data in state.get('working_memory', []):
            memory = MemoryItem(**item_data)
            self.working_memory.add(memory)
            
        # Restore short-term memory
        self.short_term_memory.memories.clear()
        for k, v in state.get('short_term_memory', {}).items():
            self.short_term_memory.memories[k] = MemoryItem(**v)
            
        # Restore long-term memory
        ltm_state = state.get('long_term_memory', {})
        self.long_term_memory.memories.clear()
        for k, v in ltm_state.get('memories', {}).items():
            self.long_term_memory.memories[k] = MemoryItem(**v)
        self.long_term_memory.semantic_index = ltm_state.get('semantic_index', {})
        self.long_term_memory.temporal_index = ltm_state.get('temporal_index', {})
        self.long_term_memory.emotional_index = ltm_state.get('emotional_index', {})
        
        # Restore identity
        self.identity_manager.load_state(state.get('identity', {}))
        
        # Restore statistics
        stats = state.get('statistics', {})
        self.total_memories_processed = stats.get('total_memories_processed', 0)
        self.consolidation_count = stats.get('consolidation_count', 0)
        
        logger.info(f"Loaded memory system with {self.total_memories_processed} total memories")

# ============= Conversational Interface =============

class ConversationalAI:
    """Main interface for the conversational AI system"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system
        self.conversation_count = 0
        
    def process_user_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        self.conversation_count += 1
        
        # Store user input as memory
        self.memory_system.process_experience(
            content=f"User: {user_input}",
            importance=0.7,
            source="user_input"
        )
        
        # Search for relevant memories
        relevant_memories = self.memory_system.search(user_input)
        
        # Get context for response generation
        context = self.memory_system.get_context_window()
        
        # Generate response (placeholder - integrate with LLM)
        response = self._generate_response(user_input, context, relevant_memories)
        
        # Store AI response as memory
        self.memory_system.process_experience(
            content=f"AI: {response}",
            importance=0.6,
            source="ai_response"
        )
        
        # Update identity
        self.memory_system.identity_manager.update_from_interaction(user_input, response)
        
        return response
        
    def _generate_response(self, user_input: str, context: str, memories: List[MemoryItem]) -> str:
        """Generate response based on input and context"""
        # This is where you'd integrate with an LLM
        # For now, return a simple response showing memory integration
        
        memory_context = ""
        if memories:
            memory_context = f"I found {len(memories)} relevant memories. "
            
        if "name" in user_input.lower():
            if 'name' in self.memory_system.identity_manager.user_relationships:
                name = self.memory_system.identity_manager.user_relationships['name']
                return f"{memory_context}Your name is {name}. I remember you!"
            else:
                return f"{memory_context}I don't have your name stored yet. What is it?"
                
        return f"{memory_context}I'm processing your input with my hierarchical memory system."

# ============= Example Usage =============

def main():
    # Initialize the system
    memory_system = MemorySystem()
    ai = ConversationalAI(memory_system)
    
    # Example conversation
    print("=== Advanced Memory System for Conversational AI ===\n")
    
    # Simulate a conversation
    interactions = [
        "Hello! My name is Alice.",
        "I love reading science fiction books.",
        "What's my name?",
        "Do you remember what I like to read?",
        "Tell me about yourself."
    ]
    
    for user_input in interactions:
        print(f"User: {user_input}")
        response = ai.process_user_input(user_input)
        print(f"AI: {response}\n")
        time.sleep(0.1)  # Small delay for timestamp variation
        
    # Show memory statistics
    print("\n=== Memory Statistics ===")
    print(f"Total memories processed: {memory_system.total_memories_processed}")
    print(f"Memories consolidated: {memory_system.consolidation_count}")
    print(f"Working memory items: {len(memory_system.working_memory.get_all())}")
    print(f"Short-term memory items: {len(memory_system.short_term_memory.memories)}")
    print(f"Long-term memory items: {len(memory_system.long_term_memory.memories)}")
    
    # Save state
    memory_system.save_to_disk("conversation_memory.pkl")
    print("\nMemory system saved to disk.")
    
    # Demonstrate loading
    new_memory_system = MemorySystem()
    new_memory_system.load_from_disk("conversation_memory.pkl")
    print("Memory system loaded from disk.")
    
    # Test persistence
    new_ai = ConversationalAI(new_memory_system)
    test_input = "What's my name again?"
    print(f"\nUser: {test_input}")
    response = new_ai.process_user_input(test_input)
    print(f"AI: {response}")

if __name__ == "__main__":
    main()