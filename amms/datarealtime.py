"""
Real-Time Data Integration Module
Integrates live data streams into the memory system
"""

import aiohttp
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import json

class DataIntegrator:
    """Integrates real-time data from various sources"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system
        self.active_sources = {}
        self.update_interval = 30  # seconds
        
    async def add_data_source(self, name: str, url: str, parser_func):
        """Add a new data source"""
        self.active_sources[name] = {
            'url': url,
            'parser': parser_func,
            'last_update': None
        }
        
    async def fetch_data(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Fetch data from a specific source"""
        source = self.active_sources.get(source_name)
        if not source:
            return None
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source['url']) as response:
                    if response.status == 200:
                        data = await response.json()
                        parsed_data = source['parser'](data)
                        return parsed_data
        except Exception as e:
            logger.error(f"Error fetching data from {source_name}: {e}")
            return None
            
    async def update_all_sources(self):
        """Update all active data sources"""
        tasks = []
        for source_name in self.active_sources:
            task = self.process_source_update(source_name)
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
    async def process_source_update(self, source_name: str):
        """Process update for a single source"""
        data = await self.fetch_data(source_name)
        if data:
            # Create memory from data
            content = f"Data from {source_name}: {json.dumps(data, indent=2)}"
            
            # Calculate importance based on data changes
            importance = self.calculate_data_importance(source_name, data)
            
            # Store in memory system
            self.memory_system.process_experience(
                content=content,
                importance=importance,
                source=f"data_{source_name}"
            )
            
            self.active_sources[source_name]['last_update'] = datetime.now()
            
    def calculate_data_importance(self, source_name: str, data: Dict[str, Any]) -> float:
        """Calculate importance of data update"""
        # Simple heuristic - can be enhanced
        last_data = self.active_sources[source_name].get('last_data')
        if not last_data:
            self.active_sources[source_name]['last_data'] = data
            return 0.8  # First data point is important
            
        # Calculate change magnitude
        change = 0.5  # Default
        # Add logic to compare data and calculate actual change
        
        self.active_sources[source_name]['last_data'] = data
        return change
        
    async def start_monitoring(self):
        """Start continuous monitoring of data sources"""
        while True:
            await self.update_all_sources()
            await asyncio.sleep(self.update_interval)

# Example parsers for different data types
def parse_financial_data(data: Dict) -> Dict[str, Any]:
    """Parse financial market data"""
    return {
        'price': data.get('price'),
        'volume': data.get('volume'),
        'change_24h': data.get('change_24h'),
        'timestamp': datetime.now().isoformat()
    }

def parse_news_feed(data: Dict) -> Dict[str, Any]:
    """Parse news feed data"""
    articles = data.get('articles', [])
    return {
        'headlines': [a.get('title') for a in articles[:5]],
        'count': len(articles),
        'timestamp': datetime.now().isoformat()
    }