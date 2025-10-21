"""
Test script for worker pool concurrent conversion functionality.
Tests that multiple conversions can run independently.
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_FILES = ["graph_r1.pdf", "test3.pdf"]  # Adjust based on your test files


class ConversionTester:
    """Test concurrent conversions"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.conversions: Dict[str, dict] = {}

    async def start_conversion(self, filename: str, prompt_mode: str = "prompt_layout_all_en") -> str:
        """Start a conversion and return conversion_id"""
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field("filename", filename)
            data.add_field("prompt_mode", prompt_mode)

            try:
                async with session.post(f"{self.base_url}/convert", data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        conversion_id = result.get("conversion_id")
                        logger.info(f"✓ Started conversion for {filename}: {conversion_id}")
                        logger.info(f"  Queue size: {result.get('queue_size')}, Active tasks: {result.get('active_tasks')}")
                        self.conversions[conversion_id] = {
                            "filename": filename,
                            "status": "pending",
                            "progress": 0,
                        }
                        return conversion_id
                    else:
                        logger.error(f"✗ Failed to start conversion for {filename}: {resp.status}")
                        return None
            except Exception as e:
                logger.error(f"✗ Error starting conversion: {str(e)}")
                return None

    async def monitor_conversion(self, conversion_id: str, timeout: int = 3600):
        """Monitor a conversion via WebSocket"""
        ws_url = f"{self.base_url.replace('http', 'ws')}/ws/conversion/{conversion_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    logger.info(f"✓ Connected to WebSocket for {conversion_id}")
                    
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        try:
                            msg = await asyncio.wait_for(ws.receive_json(), timeout=5)
                            
                            status = msg.get("status", "unknown")
                            progress = msg.get("progress", 0)
                            message = msg.get("message", "")
                            
                            self.conversions[conversion_id]["status"] = status
                            self.conversions[conversion_id]["progress"] = progress
                            
                            logger.info(f"  [{conversion_id[:8]}...] {status}: {progress}% - {message}")
                            
                            if status in ["completed", "error"]:
                                logger.info(f"✓ Conversion {conversion_id[:8]}... finished with status: {status}")
                                return status
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error(f"Error receiving message: {str(e)}")
                            break
                    
                    logger.warning(f"✗ Conversion {conversion_id[:8]}... timed out")
                    return "timeout"
        except Exception as e:
            logger.error(f"✗ WebSocket error for {conversion_id}: {str(e)}")
            return "error"

    async def get_worker_pool_status(self) -> dict:
        """Get worker pool status"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/worker-pool-status") as resp:
                    if resp.status == 200:
                        return await resp.json()
            except Exception as e:
                logger.error(f"Error getting worker pool status: {str(e)}")
        return {}

    async def test_concurrent_conversions(self, files: List[str]):
        """Test concurrent conversions"""
        logger.info("=" * 60)
        logger.info("Testing Concurrent Conversions")
        logger.info("=" * 60)

        # Start all conversions
        conversion_ids = []
        for filename in files:
            conv_id = await self.start_conversion(filename)
            if conv_id:
                conversion_ids.append(conv_id)
            await asyncio.sleep(0.5)  # Small delay between submissions

        if not conversion_ids:
            logger.error("No conversions started!")
            return

        logger.info(f"\nStarted {len(conversion_ids)} conversions")
        logger.info("Monitoring progress...\n")

        # Monitor all conversions concurrently
        tasks = [self.monitor_conversion(conv_id) for conv_id in conversion_ids]
        results = await asyncio.gather(*tasks)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        
        for i, (conv_id, result) in enumerate(zip(conversion_ids, results)):
            filename = self.conversions[conv_id]["filename"]
            status = self.conversions[conv_id]["status"]
            progress = self.conversions[conv_id]["progress"]
            logger.info(f"{i+1}. {filename}: {status} ({progress}%)")

        # Get final worker pool status
        pool_status = await self.get_worker_pool_status()
        logger.info(f"\nFinal Worker Pool Status:")
        logger.info(f"  Queue size: {pool_status.get('queue_size', 'N/A')}")
        logger.info(f"  Active tasks: {pool_status.get('active_tasks', 'N/A')}")
        logger.info(f"  Num workers: {pool_status.get('num_workers', 'N/A')}")


async def main():
    """Main test function"""
    tester = ConversionTester()
    
    # Check initial pool status
    logger.info("Initial Worker Pool Status:")
    status = await tester.get_worker_pool_status()
    logger.info(f"  Queue size: {status.get('queue_size', 'N/A')}")
    logger.info(f"  Active tasks: {status.get('active_tasks', 'N/A')}")
    logger.info(f"  Num workers: {status.get('num_workers', 'N/A')}\n")
    
    # Run concurrent conversion test
    await tester.test_concurrent_conversions(TEST_FILES)


if __name__ == "__main__":
    asyncio.run(main())

