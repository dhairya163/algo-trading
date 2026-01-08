"""
Direct Groww API client without the growwapi package.

Uses the Groww Trading API directly via HTTP requests.
"""

import httpx
from datetime import datetime
from typing import Optional
from loguru import logger


class GrowwDirectClient:
    """Direct HTTP client for Groww Trading API."""

    BASE_URL = "https://api.groww.in/v1/api"

    def __init__(self, api_key: str, api_secret: str = None):
        """
        Initialize Groww client.

        Args:
            api_key: Groww API key (JWT token)
            api_secret: Groww API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = httpx.Client(timeout=30.0)

    def _get_headers(self) -> dict:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def get_holdings(self) -> list[dict]:
        """
        Fetch all holdings from Groww.

        Returns:
            List of holding dictionaries
        """
        try:
            # Try the holdings endpoint
            response = self._client.get(
                f"{self.BASE_URL}/stocks/holdings",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("holdings", data.get("data", []))

            # Try alternative endpoint
            response = self._client.get(
                f"{self.BASE_URL}/user/holdings",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("holdings", data.get("data", []))

            logger.error(f"Holdings API returned {response.status_code}: {response.text[:500]}")
            return []

        except Exception as e:
            logger.error(f"Failed to fetch holdings: {e}")
            return []

    def get_positions(self) -> list[dict]:
        """Fetch current positions."""
        try:
            response = self._client.get(
                f"{self.BASE_URL}/stocks/positions",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("positions", data.get("data", []))

            return []

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def get_portfolio(self) -> dict:
        """Fetch complete portfolio summary."""
        try:
            response = self._client.get(
                f"{self.BASE_URL}/stocks/portfolio",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                return response.json()

            return {}

        except Exception as e:
            logger.error(f"Failed to fetch portfolio: {e}")
            return {}

    def get_quote(self, symbol: str) -> Optional[dict]:
        """Get current quote for a symbol."""
        try:
            response = self._client.get(
                f"{self.BASE_URL}/stocks/quote/{symbol}",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self._client.close()


async def test_groww_connection(api_key: str, api_secret: str = None):
    """Test Groww API connection and fetch holdings."""

    print("\n" + "=" * 60)
    print("🔗 Testing Groww API Connection")
    print("=" * 60)

    client = GrowwDirectClient(api_key, api_secret)

    try:
        # Try to fetch holdings
        print("\n📊 Fetching holdings...")
        holdings = client.get_holdings()

        if holdings:
            print(f"✅ Found {len(holdings)} holdings!\n")
            return holdings
        else:
            print("⚠️  No holdings found or API returned empty response")

            # Try portfolio endpoint
            print("\n📊 Trying portfolio endpoint...")
            portfolio = client.get_portfolio()
            if portfolio:
                print(f"Portfolio data: {portfolio}")
                return portfolio.get("holdings", [])

            # Try positions endpoint
            print("\n📊 Trying positions endpoint...")
            positions = client.get_positions()
            if positions:
                print(f"✅ Found {len(positions)} positions!")
                return positions

        return []

    finally:
        client.close()


if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("GROWW_API_KEY")
    api_secret = os.getenv("GROWW_API_SECRET")

    if not api_key:
        print("❌ GROWW_API_KEY not found in environment")
    else:
        holdings = asyncio.run(test_groww_connection(api_key, api_secret))
        print(f"\nHoldings: {holdings}")
