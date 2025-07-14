# ============================================================================
# BULLETPROOF REAL-TIME DATA INTEGRATION - COMPLETE REWRITE
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
# TESTING AND VALIDATION
# ============================================================================

def test_bulletproof_integrator():
    """Test the bulletproof integrator"""
    print("🧪 Testing Bulletproof RealTimeDataIntegrator")
    print("=" * 60)
    
    try:
        # Initialize integrator
        print("1. Initializing integrator...")
        integrator = RealTimeDataIntegrator()
        print("   ✅ Integrator initialized")
        
        # Test API connections
        print("2. Testing API connections...")
        connections = integrator.validate_api_connections()
        for api, status in connections.items():
            status_emoji = "✅" if status == 'CONNECTED' else "⚠️" if 'NOT_AVAILABLE' in status else "❌"
            print(f"   {status_emoji} {api}: {status}")
        
        # Test fetch cycle
        print("3. Testing fetch and process cycle...")
        result = integrator.fetch_and_process_cycle('financial_analysis', count=3)
        print(f"   ✅ Cycle result: {result['experiences_created']} experiences created")
        print(f"      Raw: {result['raw_data_count']} → Quality: {result['quality_filtered_count']} → Novel: {result['novel_count']}")
        
        # Test continuous integration
        print("4. Testing continuous integration...")
        start_result = integrator.start_continuous_integration(['financial_analysis'])
        print(f"   ✅ Integration started: {start_result['streams_initialized']} streams")
        
        # Let it run briefly
        print("5. Running for 30 seconds...")
        
        # Check status multiple times during the run
        for i in range(6):  # Check every 5 seconds
            time.sleep(5)
            status = integrator.get_status()
            if i == 0 or i == 5:  # Print at start and end
                print(f"   📊 Status: {status['status']} | Processed: {status['experiences_processed']} | Threads: {status['background_threads']}")
        
        # Final status check
        status = integrator.get_status()
        print(f"   📊 Final Status: {status['status']}")
        print(f"   📈 Total Processed: {status['experiences_processed']} experiences")
        print(f"   🧵 Active Threads: {status['background_threads']}")
        print(f"   📊 Queue Size: {status['queue_size']}")
        print(f"   📦 Cache Size: {status['cache_size']}")
        
        # Health check
        print("6. Health check...")
        health = integrator.health_check()
        print(f"   ❤️ Health: {health['status']}")
        print(f"   ⏱️ Uptime: {health['uptime_seconds']:.1f}s")
        
        # EMMS Memory diagnostics
        print("7. EMMS Memory Integration Check...")
        emms_status = integrator.get_emms_memory_status()
        print(f"   🧠 EMMS Connected: {emms_status.get('emms_connected', False)}")
        print(f"   🔗 Integration Paths: {len(emms_status.get('integration_paths', []))}")
        print(f"   📊 EMMS Experiences: {emms_status.get('total_emms_experiences', 0)}")
        
        if emms_status.get('hierarchical_memory_available'):
            print(f"   📋 Working Memory: {emms_status.get('working_memory_count', 0)}")
            print(f"   📚 Short-term Memory: {emms_status.get('short_term_memory_count', 0)}")
            print(f"   🗄️ Long-term Memory: {emms_status.get('long_term_memory_count', 0)}")
        
        # Force memory consolidation
        print("8. Forcing memory consolidation...")
        consolidation = integrator.force_memory_consolidation()
        print(f"   🔄 Working→Short-term: {consolidation.get('working_to_short_term', 0)}")
        print(f"   🔄 Short-term→Long-term: {consolidation.get('short_term_to_long_term', 0)}")
        
        # Check EMMS status after consolidation
        emms_status_after = integrator.get_emms_memory_status()
        print(f"   📊 EMMS Experiences After: {emms_status_after.get('total_emms_experiences', 0)}")
        
        # Shutdown
        print("9. Shutting down...")
        shutdown_success = integrator.shutdown()
        print(f"   ✅ Shutdown: {'Success' if shutdown_success else 'Failed'}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bulletproof_integrator()