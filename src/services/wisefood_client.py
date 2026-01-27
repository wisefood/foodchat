import os
from typing import Optional

from wisefood import Client, Credentials


class WiseFoodClientPool:
    """Connection pool for WiseFood API clients.

    Initializes the client from environment variables:
    - WISEFOOD_API_URL: The WiseFood API base URL
    - WISEFOOD_USERNAME: API username
    - WISEFOOD_PASSWORD: API password
    """

    def __init__(self):
        self._client: Optional[Client] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the WiseFood client from environment variables."""
        api_url = os.getenv("WISEFOOD_API_URL")
        username = os.getenv("WISEFOOD_USERNAME")
        password = os.getenv("WISEFOOD_PASSWORD")

        if not all([api_url, username, password]):
            raise ValueError(
                "Missing required environment variables: "
                "WISEFOOD_API_URL, WISEFOOD_USERNAME, WISEFOOD_PASSWORD"
            )

        creds = Credentials(username=username, password=password)
        self._client = Client(api_url, creds)

    def get_client(self) -> Client:
        """Get the WiseFood client instance."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def refresh_client(self) -> Client:
        """Force re-initialization of the client."""
        self._initialize_client()
        return self._client
