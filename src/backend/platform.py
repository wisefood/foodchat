"""WiseFood Client connection pool for efficient client instance management."""
import os
import logging
from typing import Optional
from threading import Lock
from contextlib import contextmanager

from wisefood import Client, Credentials

logger = logging.getLogger(__name__)

# Environment variable configuration
WISEFOOD_API_URL = os.getenv("WISEFOOD_API_URL")
WISEFOOD_USERNAME = os.getenv("WISEFOOD_USERNAME")
WISEFOOD_PASSWORD = os.getenv("WISEFOOD_PASSWORD")
WISEFOOD_CLIENT_ID = os.getenv("WISEFOOD_CLIENT_ID")
WISEFOOD_CLIENT_SECRET = os.getenv("WISEFOOD_CLIENT_SECRET")
WISEFOOD_POOL_SIZE = int(os.getenv("WISEFOOD_POOL_SIZE", "5"))


class WiseFoodPool:
    """
    Connection pool for WiseFood Client instances.

    Maintains reusable Client instances to avoid creating new connections
    for every request. Supports both username/password and client credentials auth.

    Environment variables:
        WISEFOOD_API_URL: The WiseFood API base URL (required)
        WISEFOOD_USERNAME: Username for password auth
        WISEFOOD_PASSWORD: Password for password auth
        WISEFOOD_CLIENT_ID: Client ID for Keycloak auth
        WISEFOOD_CLIENT_SECRET: Client secret for Keycloak auth
        WISEFOOD_POOL_SIZE: Number of connections to maintain (default: 5)
    """

    def __init__(self, pool_size: int = None, lazy_init: bool = True):
        """
        Initialize the connection pool.

        Args:
            pool_size: Number of connections to maintain (default from env: 5)
            lazy_init: If True, don't create connections until first use
        """
        self._pool: list[Client] = []
        self._available: list[Client] = []
        self._lock = Lock()
        self._pool_size = pool_size or WISEFOOD_POOL_SIZE
        self._initialized = False

        if not lazy_init:
            self._initialize_pool()

    def _get_credentials(self) -> Credentials:
        """
        Build credentials from environment variables.

        Supports two auth methods:
        1. Username/password auth (WISEFOOD_USERNAME, WISEFOOD_PASSWORD)
        2. Client credentials auth (WISEFOOD_CLIENT_ID, WISEFOOD_CLIENT_SECRET)
        """
        # Prefer client credentials if available
        if WISEFOOD_CLIENT_ID and WISEFOOD_CLIENT_SECRET:
            return Credentials(
                client_id=WISEFOOD_CLIENT_ID,
                client_secret=WISEFOOD_CLIENT_SECRET
            )

        # Fall back to username/password
        if WISEFOOD_USERNAME and WISEFOOD_PASSWORD:
            return Credentials(
                username=WISEFOOD_USERNAME,
                password=WISEFOOD_PASSWORD
            )

        raise ValueError(
            "Missing WiseFood credentials. Set either:\n"
            "  - WISEFOOD_USERNAME and WISEFOOD_PASSWORD, or\n"
            "  - WISEFOOD_CLIENT_ID and WISEFOOD_CLIENT_SECRET"
        )

    def _create_client(self) -> Client:
        """Create a new WiseFood client instance."""
        if not WISEFOOD_API_URL:
            raise ValueError("WISEFOOD_API_URL environment variable is required")

        credentials = self._get_credentials()
        return Client(WISEFOOD_API_URL, credentials)

    def _initialize_pool(self):
        """Initialize the pool with connections."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                for _ in range(self._pool_size):
                    client = self._create_client()
                    self._pool.append(client)
                    self._available.append(client)

                self._initialized = True
                logger.info(f"Initialized WiseFood pool with {self._pool_size} connections")

            except Exception as e:
                logger.error(f"Failed to initialize WiseFood pool: {e}")
                raise

    def get_client(self) -> Client:
        """
        Get an available Client from the pool.

        Returns:
            Client instance from the pool
        """
        if not self._initialized:
            self._initialize_pool()

        with self._lock:
            if self._available:
                client = self._available.pop()
                logger.debug(f"Got client from pool ({len(self._available)} remaining)")
                return client
            else:
                logger.warning("No available clients in pool, reusing existing")
                return self._pool[0] if self._pool else self._create_client()

    def return_client(self, client: Client):
        """
        Return a client to the pool.

        Args:
            client: Client instance to return
        """
        with self._lock:
            if client in self._pool and client not in self._available:
                self._available.append(client)
                logger.debug(f"Returned client to pool ({len(self._available)} available)")

    @contextmanager
    def client(self):
        """
        Context manager for automatic client checkout/return.

        Usage:
            with WISEFOOD.client() as client:
                member = client.members.get(member_id)
        """
        client = self.get_client()
        try:
            yield client
        finally:
            self.return_client(client)

    def get_available_count(self) -> int:
        """Get the number of available clients."""
        with self._lock:
            return len(self._available)

    def get_pool_size(self) -> int:
        """Get the total pool size."""
        return self._pool_size

    def clear_pool(self):
        """Clear all connections from the pool."""
        with self._lock:
            self._pool.clear()
            self._available.clear()
            self._initialized = False
            logger.info("Cleared WiseFood pool")

    def refresh_pool(self):
        """Refresh all connections in the pool."""
        self.clear_pool()
        self._initialize_pool()


# Global connection pool instance (lazy initialization)
WISEFOOD = WiseFoodPool(lazy_init=True)