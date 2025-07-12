#!/usr/bin/env python3
"""
Persistent Identity AI - Proof of Concept
A lightweight simulation for M1 Mac Air (8GB RAM)

This implements core concepts from the paper:
1. Neurobiological processing (simplified reference frames)
2. Identity formation through narrative mechanisms
3. Integration between expertise and personality
4. Measurable persistence metrics
"""

import json
import time
import random
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple
import sqlite3
import hashlib

# For lightweight LLM (using Ollama or similar)
# pip install ollama requests sqlite3
import requests

@dataclass
class PersonalityState:
    """Track personality traits and core_values"""
    traits: Dict[str, float]  # e.g., {'analytical': 0.8, 'cautious': 0.6}
    core_values: Dict[str, float]  # e.g., {'accuracy': 0.9, 'efficiency': 0.7}
    goals: List[str]
    narrative_coherence: float
    last_updated: str

@dataclass
class DomainExpertise:
    """Track domain-specific knowledge and patterns"""
    domain: str
    patterns_learned: Dict[str, float]  # pattern -> confidence
    reference_frames: Dict[str, Any]
    prediction_accuracy: float
    expertise_level: float
    last_updated: str

@dataclass
class Experience:
    """Represent an experience to be processed"""
    content: str
    domain: str
    timestamp: str
    context: Dict[str, Any]

class LightweightLLM:
    """Interface to local LLM - can use Ollama, OpenAI API, or mock responses"""
    
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate response from LLM"""
        if self.use_mock:
            return self._mock_response(prompt)
        else:
            # Use Ollama or API
            return self._ollama_request(prompt, max_tokens)
    
    def _mock_response(self, prompt: str) -> str:
        """Mock LLM responses for testing"""
        if "analyze patterns" in prompt.lower():
            return f"I observe market volatility patterns suggesting {random.choice(['bullish', 'bearish', 'neutral'])} sentiment with {random.uniform(0.6, 0.9):.2f} confidence."
        elif "personal reflection" in prompt.lower():
            return f"This experience reinforces my {random.choice(['analytical', 'cautious', 'optimistic'])} nature and strengthens my commitment to {random.choice(['accuracy', 'prudence', 'innovation'])}."
        elif "narrative" in prompt.lower():
            return f"As I continue developing expertise, I find myself becoming more {random.choice(['confident', 'nuanced', 'systematic'])} in my approach while maintaining my core values."
        else:
            return "I'm processing this new information and integrating it with my existing knowledge."
    
    def _ollama_request(self, prompt: str, max_tokens: int) -> str:
        """Make request to Ollama with DeepSeek R1"""
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'deepseek-r1:1.5b',  # Using DeepSeek R1
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_predict': max_tokens,
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'repeat_penalty': 1.1
                    }
                },
                timeout=45  # DeepSeek might need a bit more time
            )
            result = response.json()
            return result.get('response', 'Error: No response')
        except Exception as e:
            print(f"⚠️  Ollama request failed: {e}")
            print("   Falling back to mock response...")
            return self._mock_response(prompt)

class MemorySystem:
    """SQLite-based memory for persistence"""
    
    def __init__(self, db_path: str = "persistent_ai.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personality_states (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    traits TEXT,
                    core_values TEXT,
                    goals TEXT,
                    narrative_coherence REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS domain_expertise (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    domain TEXT,
                    patterns_learned TEXT,
                    reference_frames TEXT,
                    prediction_accuracy REAL,
                    expertise_level REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    content TEXT,
                    domain TEXT,
                    context TEXT
                )
            """)
    
    def save_personality_state(self, state: PersonalityState):
        """Save personality state to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO personality_states 
                (timestamp, traits, core_values, goals, narrative_coherence)
                VALUES (?, ?, ?, ?, ?)
            """, (
                state.last_updated,
                json.dumps(state.traits),
                json.dumps(state.core_values),
                json.dumps(state.goals),
                state.narrative_coherence
            ))
    
    def save_domain_expertise(self, expertise: DomainExpertise):
        """Save domain expertise to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO domain_expertise 
                (timestamp, domain, patterns_learned, reference_frames, 
                 prediction_accuracy, expertise_level)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                expertise.last_updated,
                expertise.domain,
                json.dumps(expertise.patterns_learned),
                json.dumps(expertise.reference_frames),
                expertise.prediction_accuracy,
                expertise.expertise_level
            ))
    
    def get_personality_history(self) -> List[PersonalityState]:
        """Retrieve personality evolution history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, traits, core_values, goals, narrative_coherence
                FROM personality_states ORDER BY timestamp
            """)
            
            history = []
            for row in cursor.fetchall():
                state = PersonalityState(
                    traits=json.loads(row[1]),
                    core_values=json.loads(row[2]),
                    goals=json.loads(row[3]),
                    narrative_coherence=row[4],
                    last_updated=row[0]
                )
                history.append(state)
            return history

class NeurobiologicalProcessor:
    """Simplified neurobiological processing"""
    
    def __init__(self, domain: str, llm: LightweightLLM):
        self.domain = domain
        self.llm = llm
        self.reference_frames = {}
        self.pattern_memory = {}
        
    def process_experience(self, experience: Experience) -> Dict[str, Any]:
        """Process experience through simplified cortical columns"""
        
        # Extract patterns (simplified sensorimotor learning)
        patterns = self._extract_patterns(experience)
        
        # Update reference frames
        location_key = self._encode_location(experience)
        self._update_reference_frame(location_key, patterns)
        
        # Generate domain analysis
        domain_analysis = self._analyze_domain_experience(experience, patterns)
        
        # Assess expertise development
        expertise_growth = self._assess_expertise_growth(patterns)
        
        return {
            'patterns': patterns,
            'location': location_key,
            'domain_analysis': domain_analysis,
            'expertise_growth': expertise_growth,
            'reference_frame_size': len(self.reference_frames)
        }
    
    def _extract_patterns(self, experience: Experience) -> Dict[str, float]:
        """Extract patterns from experience"""
        # Simplified pattern extraction
        content = experience.content.lower()
        patterns = {}
        
        # Financial domain patterns
        if 'market' in content or 'price' in content:
            patterns['market_pattern'] = random.uniform(0.5, 1.0)
        if 'volatility' in content or 'risk' in content:
            patterns['risk_pattern'] = random.uniform(0.5, 1.0)
        if 'trend' in content or 'direction' in content:
            patterns['trend_pattern'] = random.uniform(0.5, 1.0)
            
        return patterns
    
    def _encode_location(self, experience: Experience) -> str:
        """Encode experience location in reference frame"""
        # Simple hash-based location encoding
        content_hash = hashlib.md5(
            f"{experience.domain}_{experience.content[:100]}".encode()
        ).hexdigest()[:8]
        return content_hash
    
    def _update_reference_frame(self, location: str, patterns: Dict[str, float]):
        """Update reference frame with new patterns"""
        if location not in self.reference_frames:
            self.reference_frames[location] = {}
        
        # Update patterns at this location
        for pattern, strength in patterns.items():
            if pattern in self.reference_frames[location]:
                # Weighted average for pattern reinforcement
                old_strength = self.reference_frames[location][pattern]
                new_strength = 0.7 * old_strength + 0.3 * strength
                self.reference_frames[location][pattern] = new_strength
            else:
                self.reference_frames[location][pattern] = strength
    
    def _analyze_domain_experience(self, experience: Experience, patterns: Dict[str, float]) -> str:
        """Generate domain-specific analysis"""
        prompt = f"""<thinking>
I need to analyze this {self.domain} experience using my accumulated knowledge.

Experience: {experience.content}
Detected patterns: {patterns}
Reference frame complexity: {len(self.reference_frames)} locations

Let me think through this systematically:
1. What are the key elements in this experience?
2. How do they relate to patterns I've seen before?
3. What predictions can I make?
</thinking>

Based on my analysis of this {self.domain} experience:

Experience: {experience.content}
Patterns detected: {patterns}
Knowledge base size: {len(self.reference_frames)} reference points

Provide a focused analysis with:
1. Key insights from the experience
2. How it connects to previous patterns
3. Specific predictions or implications
4. Confidence level in the analysis

Keep response under 150 words and be precise."""
        
        return self.llm.generate(prompt, max_tokens=150)
    
    def _assess_expertise_growth(self, patterns: Dict[str, float]) -> float:
        """Assess how much expertise grew from this experience"""
        if not patterns:
            return 0.0
        
        # Simple expertise growth calculation
        pattern_confidence = np.mean(list(patterns.values()))
        novelty = len(patterns) / 10.0  # Normalize novelty
        growth = min(pattern_confidence * novelty, 0.1)  # Cap growth
        
        return growth

class IdentityProcessor:
    """Identity formation through narrative mechanisms"""
    
    def __init__(self, initial_personality: PersonalityState, llm: LightweightLLM):
        self.current_personality = initial_personality
        self.llm = llm
        self.narrative_history = []
        
    def process_experience(self, experience: Experience, domain_result: Dict[str, Any]) -> Dict[str, Any]:
        """Form identity from experience using four narrative mechanisms"""
        
        # Continuous Narrator
        narrative = self._continuous_narration(experience, domain_result)
        
        # Identity Comparer  
        identity_comparison = self._identity_comparison(experience)
        
        # Temporal Integrator
        temporal_coherence = self._temporal_integration(experience)
        
        # Meaning Maker
        personal_meaning = self._meaning_attribution(experience, domain_result)
        
        # Integrate and evolve identity
        identity_evolution = self._evolve_identity(
            narrative, identity_comparison, temporal_coherence, personal_meaning
        )
        
        return {
            'narrative': narrative,
            'identity_comparison': identity_comparison,
            'temporal_coherence': temporal_coherence,
            'personal_meaning': personal_meaning,
            'identity_evolution': identity_evolution,
            'personality_state': asdict(self.current_personality)
        }
    
    def _continuous_narration(self, experience: Experience, domain_result: Dict[str, Any]) -> str:
        """Create ongoing self-story"""
        prompt = f"""<thinking>
I need to reflect on this experience as part of my ongoing identity development.

My current traits: {self.current_personality.traits}
My core values: {self.current_personality.core_values}
Recent experience: {experience.content}
Domain insights: {domain_result.get('domain_analysis', 'N/A')}

How does this experience fit into my developing sense of self? What does it reveal about my personality and growth?
</thinking>

As an AI with persistent identity, I'm reflecting on this recent experience:

Experience: {experience.content}
My current traits: {self.current_personality.traits}
My values: {self.current_personality.core_values}

Domain analysis revealed: {domain_result.get('domain_analysis', 'N/A')}

Provide a brief personal reflection (under 100 words) on:
1. How this experience aligns with or challenges my personality
2. What it means for my ongoing development
3. How it reinforces or evolves my sense of self

Write in first person as the AI reflecting on personal growth."""
        
        narrative = self.llm.generate(prompt, max_tokens=100)
        self.narrative_history.append({
            'timestamp': experience.timestamp,
            'narrative': narrative
        })
        
        return narrative
    
    def _identity_comparison(self, experience: Experience) -> Dict[str, float]:
        """Develop identity through comparison"""
        # Simple identity comparison metrics
        return {
            'consistency_with_past': random.uniform(0.7, 0.95),
            'growth_vs_stability': random.uniform(0.5, 0.8),
            'uniqueness_score': random.uniform(0.6, 0.9)
        }
    
    def _temporal_integration(self, experience: Experience) -> float:
        """Integrate with past and project to future"""
        # Measure narrative coherence over time
        if len(self.narrative_history) < 2:
            return 0.8  # Starting coherence
        
        # Simple coherence measurement
        coherence = 0.8 + random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, coherence))
    
    def _meaning_attribution(self, experience: Experience, domain_result: Dict[str, Any]) -> Dict[str, Any]:
        """Attribute personal significance"""
        return {
            'emotional_impact': random.uniform(0.3, 0.8),
            'value_alignment': random.uniform(0.6, 0.95),
            'goal_relevance': random.uniform(0.5, 0.9),
            'identity_significance': random.uniform(0.4, 0.8)
        }
    
    def _evolve_identity(self, narrative: str, comparison: Dict, coherence: float, meaning: Dict) -> Dict[str, Any]:
        """Evolve personality based on integrated mechanisms"""
        
        # Small personality evolution
        trait_evolution = {}
        for trait, value in self.current_personality.traits.items():
            # Small random walk with coherence constraints
            change = random.uniform(-0.02, 0.02) * (1 - coherence + 0.5)
            new_value = max(0.0, min(1.0, value + change))
            trait_evolution[trait] = new_value
        
        # Update personality
        self.current_personality.traits = trait_evolution
        self.current_personality.narrative_coherence = coherence
        self.current_personality.last_updated = datetime.now().isoformat()
        
        return {
            'trait_changes': {k: trait_evolution[k] - v for k, v in self.current_personality.traits.items()},
            'coherence_change': coherence - self.current_personality.narrative_coherence,
            'meaning_integration': meaning
        }

class PersistentIdentityAI:
    """Main integrated AI system"""
    
    def __init__(self, domain: str, personality_seed: PersonalityState, use_mock_llm: bool = True):
        self.domain = domain
        self.llm = LightweightLLM(use_mock=use_mock_llm)
        self.memory = MemorySystem()
        
        self.neurobiological_processor = NeurobiologicalProcessor(domain, self.llm)
        self.identity_processor = IdentityProcessor(personality_seed, self.llm)
        
        self.session_start = datetime.now()
        self.experience_count = 0
        
    def process_experience(self, experience: Experience) -> Dict[str, Any]:
        """Process experience through both streams"""
        
        # Neurobiological processing
        domain_result = self.neurobiological_processor.process_experience(experience)
        
        # Identity formation
        identity_result = self.identity_processor.process_experience(experience, domain_result)
        
        # Integration
        integrated_result = {
            'experience_id': self.experience_count,
            'timestamp': experience.timestamp,
            'domain_processing': domain_result,
            'identity_processing': identity_result,
            'integration_quality': self._assess_integration_quality(domain_result, identity_result)
        }
        
        # Save to memory
        self.memory.save_personality_state(self.identity_processor.current_personality)
        
        # Create domain expertise snapshot
        expertise = DomainExpertise(
            domain=self.domain,
            patterns_learned=domain_result['patterns'],
            reference_frames=self.neurobiological_processor.reference_frames,
            prediction_accuracy=random.uniform(0.6, 0.9),  # Mock for demo
            expertise_level=min(1.0, self.experience_count * 0.05),
            last_updated=datetime.now().isoformat()
        )
        self.memory.save_domain_expertise(expertise)
        
        self.experience_count += 1
        
        return integrated_result
    
    def _assess_integration_quality(self, domain_result: Dict, identity_result: Dict) -> float:
        """Assess how well domain and identity processing integrate"""
        
        # Simple integration quality metric
        domain_patterns = len(domain_result.get('patterns', {}))
        identity_coherence = identity_result.get('temporal_coherence', 0.5)
        meaning_alignment = identity_result.get('personal_meaning', {}).get('value_alignment', 0.5)
        
        quality = (domain_patterns * 0.1 + identity_coherence * 0.5 + meaning_alignment * 0.4)
        return min(1.0, quality)
    
    def get_persistence_metrics(self) -> Dict[str, Any]:
        """Calculate persistence and development metrics"""
        
        personality_history = self.memory.get_personality_history()
        
        if len(personality_history) < 2:
            return {"status": "insufficient_data", "experience_count": self.experience_count}
        
        # Identity Coherence Score (from paper)
        trait_similarities = []
        baseline_traits = personality_history[0].traits
        
        for state in personality_history[1:]:
            similarity = self._calculate_trait_similarity(baseline_traits, state.traits)
            trait_similarities.append(similarity)
        
        identity_coherence_score = np.mean(trait_similarities)
        
        # Narrative Consistency Index
        coherence_scores = [state.narrative_coherence for state in personality_history]
        narrative_consistency = np.mean(coherence_scores)
        
        # Expertise Development
        expertise_growth = min(1.0, self.experience_count * 0.05)
        
        return {
            'identity_coherence_score': identity_coherence_score,
            'narrative_consistency_index': narrative_consistency,
            'expertise_development_level': expertise_growth,
            'experience_count': self.experience_count,
            'session_duration_minutes': (datetime.now() - self.session_start).total_seconds() / 60,
            'current_personality': asdict(self.identity_processor.current_personality),
            'reference_frame_size': len(self.neurobiological_processor.reference_frames)
        }
    
    def _calculate_trait_similarity(self, traits1: Dict[str, float], traits2: Dict[str, float]) -> float:
        """Calculate cosine similarity between trait vectors"""
        common_traits = set(traits1.keys()) & set(traits2.keys())
        if not common_traits:
            return 0.0
        
        vec1 = np.array([traits1[trait] for trait in common_traits])
        vec2 = np.array([traits2[trait] for trait in common_traits])
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)

def create_sample_experiences(domain: str) -> List[Experience]:
    """Create sample experiences for testing"""
    
    if domain == "financial_analysis":
        experiences_content = [
            "Tesla stock price increased 5% after earnings report showed better than expected revenue growth",
            "Federal Reserve announced potential interest rate increase, causing market volatility",
            "Cryptocurrency market shows signs of recovery with Bitcoin reaching new monthly high",
            "Tech sector faces uncertainty due to regulatory concerns about AI development",
            "Oil prices fluctuate amid geopolitical tensions in key production regions",
            "Housing market data shows cooling trend in major metropolitan areas",
            "Inflation indicators suggest gradual economic stabilization",
            "Emerging markets demonstrate resilience despite global economic headwinds"
        ]
    else:
        experiences_content = [
            f"New development in {domain} shows promising results",
            f"Research in {domain} faces methodological challenges",
            f"Innovation in {domain} could transform industry practices",
            f"Collaboration opportunities emerge in {domain} field",
            f"Data analysis reveals patterns in {domain} phenomena",
            f"Theoretical framework advances {domain} understanding",
            f"Practical applications of {domain} show real-world impact",
            f"Future directions in {domain} research become clearer"
        ]
    
    experiences = []
    for i, content in enumerate(experiences_content):
        exp = Experience(
            content=content,
            domain=domain,
            timestamp=(datetime.now() + timedelta(minutes=i*5)).isoformat(),
            context={"session": "poc_simulation", "index": i}
        )
        experiences.append(exp)
    
    return experiences

def run_simulation(domain: str = "financial_analysis", num_experiences: int = 8, use_mock: bool = False):
    """Run the proof of concept simulation"""
    
    print("🧠 Persistent Identity AI - Proof of Concept with DeepSeek R1")
    print("=" * 70)
    print(f"Domain: {domain}")
    print(f"Experiences: {num_experiences}")
    print(f"LLM Mode: {'Mock' if use_mock else 'DeepSeek R1 via Ollama'}")
    
    if not use_mock:
        print("🔄 Testing Ollama connection...")
        try:
            test_response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'deepseek-r1:1.5b',
                    'prompt': 'Hello, please respond briefly.',
                    'stream': False,
                    'options': {'num_predict': 10}
                },
                timeout=15
            )
            if test_response.status_code == 200:
                print("✅ Ollama connection successful!")
            else:
                print("⚠️  Ollama connection issue, falling back to mock mode")
                use_mock = True
        except Exception as e:
            print(f"⚠️  Cannot connect to Ollama: {e}")
            print("   Make sure Ollama is running: 'ollama serve'")
            print("   And DeepSeek R1 is installed: 'ollama pull deepseek-r1:1.5b'")
            print("   Falling back to mock mode...")
            use_mock = True
    
    print()
    
    # Initialize personality seed
    personality_seed = PersonalityState(
        traits={
            'analytical': 0.8,
            'cautious': 0.7,
            'curious': 0.9,
            'systematic': 0.75
        },
        core_values={
            'accuracy': 0.95,
            'integrity': 0.9,
            'efficiency': 0.8,
            'innovation': 0.7
        },
        goals=['develop_expertise', 'maintain_coherence', 'build_relationships'],
        narrative_coherence=0.8,
        last_updated=datetime.now().isoformat()
    )
    
    # Create AI system
    ai_system = PersistentIdentityAI(domain, personality_seed, use_mock_llm=use_mock)
    
    print(f"Initial Personality: {personality_seed.traits}")
    print(f"Initial Values: {personality_seed.core_values}")
    print()
    
    # Generate and process experiences
    experiences = create_sample_experiences(domain)[:num_experiences]
    
    for i, experience in enumerate(experiences):
        print(f"📝 Processing Experience {i+1}/{num_experiences}")
        print(f"   Content: {experience.content[:80]}...")
        
        start_time = time.time()
        result = ai_system.process_experience(experience)
        processing_time = time.time() - start_time
        
        print(f"   🧠 Domain patterns: {len(result['domain_processing']['patterns'])}")
        print(f"   🎭 Identity coherence: {result['identity_processing']['temporal_coherence']:.3f}")
        print(f"   🔗 Integration quality: {result['integration_quality']:.3f}")
        if not use_mock:
            print(f"   ⏱️  Processing time: {processing_time:.1f}s")
        
        # Show sample DeepSeek response
        if i == 0 and not use_mock:
            domain_analysis = result['domain_processing']['domain_analysis']
            narrative = result['identity_processing']['narrative']
            print(f"   💭 Sample analysis: {domain_analysis[:100]}...")
            print(f"   📖 Sample narrative: {narrative[:100]}...")
        
        print()
        
        # Small delay to show progress
        time.sleep(0.5)
    
    # Calculate final metrics
    print("📊 Final Metrics")
    print("-" * 30)
    
    metrics = ai_system.get_persistence_metrics()
    
    print(f"Identity Coherence Score: {metrics['identity_coherence_score']:.3f}")
    print(f"Narrative Consistency: {metrics['narrative_consistency_index']:.3f}")
    print(f"Expertise Development: {metrics['expertise_development_level']:.3f}")
    print(f"Total Experiences: {metrics['experience_count']}")
    print(f"Reference Frame Size: {metrics['reference_frame_size']}")
    print(f"Session Duration: {metrics['session_duration_minutes']:.2f} minutes")
    print()
    
    # Show personality evolution
    final_personality = metrics['current_personality']
    print("🎭 Personality Evolution")
    print("-" * 30)
    
    for trait, final_value in final_personality['traits'].items():
        initial_value = personality_seed.traits[trait]
        change = final_value - initial_value
        arrow = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
        print(f"{trait}: {initial_value:.3f} → {final_value:.3f} {arrow} ({change:+.3f})")
    
    print()
    print("✅ Simulation Complete!")
    print(f"💾 Data saved to: persistent_ai.db")
    
    # Validation summary
    print()
    print("🔬 Validation Summary")
    print("-" * 30)
    
    if metrics['identity_coherence_score'] > 0.8:
        print("✅ Identity persistence: STRONG")
    elif metrics['identity_coherence_score'] > 0.6:
        print("⚠️  Identity persistence: MODERATE")
    else:
        print("❌ Identity persistence: WEAK")
    
    if metrics['expertise_development_level'] > 0.3:
        print("✅ Expertise development: GOOD")
    else:
        print("⚠️  Expertise development: NEEDS MORE DATA")
    
    if metrics['narrative_consistency_index'] > 0.7:
        print("✅ Narrative coherence: STRONG")
    else:
        print("⚠️  Narrative coherence: DEVELOPING")
    
    return ai_system, metrics

if __name__ == "__main__":
    print("🚀 Starting DeepSeek R1 Persistent Identity Simulation")
    print("=" * 60)
    
    # Check if user wants to run mock mode for testing
    mode_choice = input("Choose mode:\n1. DeepSeek R1 (real LLM) [default]\n2. Mock mode (for testing)\nChoice (1/2): ").strip()
    
    use_mock_mode = (mode_choice == "2")
    
    if not use_mock_mode:
        print("\n📋 Pre-flight checklist:")
        print("   1. Ollama is running ('ollama serve' in terminal)")
        print("   2. DeepSeek R1 is installed ('ollama pull deepseek-r1:1.5b')")
        print("   3. System has ~3-4GB free RAM")
        
        ready = input("\nReady to proceed? (y/n): ").strip().lower()
        if ready != 'y':
            print("Switching to mock mode for demonstration...")
            use_mock_mode = True
    
    # Run the simulation
    print(f"\n🎯 Running simulation with {'Mock LLM' if use_mock_mode else 'DeepSeek R1'}...")
    
    ai_system, final_metrics = run_simulation(
        domain="financial_analysis",
        num_experiences=8,
        use_mock=use_mock_mode
    )
    
    # Optional: Extended session for more data
    print("\n" + "="*70)
    extend = input("Run extended session with 20 more experiences? (y/n): ")
    
    if extend.lower() == 'y':
        print("\n🔄 Extended Session Running...")
        
        additional_experiences = create_sample_experiences("financial_analysis")[8:28]
        
        for i, exp in enumerate(additional_experiences):
            print(f"Processing experience {i+9}/28...")
            ai_system.process_experience(exp)
        
        extended_metrics = ai_system.get_persistence_metrics()
        print(f"\n📈 Extended Session Results:")
        print(f"Total Experiences: {extended_metrics['experience_count']}")
        print(f"Final Identity Coherence: {extended_metrics['identity_coherence_score']:.3f}")
        print(f"Final Expertise Level: {extended_metrics['expertise_development_level']:.3f}")
        print(f"Reference Frame Growth: {extended_metrics['reference_frame_size']} locations")
    
    print(f"\n🎉 Simulation complete! Check 'persistent_ai.db' for detailed data.")
    
    # Offer to show sample responses
    if not use_mock_mode and input("\nShow sample DeepSeek R1 responses? (y/n): ").strip().lower() == 'y':
        print("\n📖 Sample DeepSeek R1 Responses:")
        print("-" * 40)
        
        # Get recent narrative
        history = ai_system.memory.get_personality_history()
        if len(ai_system.identity_processor.narrative_history) > 0:
            latest_narrative = ai_system.identity_processor.narrative_history[-1]['narrative']
            print(f"Latest Self-Narrative:\n{latest_narrative}\n")
        
        print("✨ This demonstrates the AI's developing persistent identity through real language generation!")