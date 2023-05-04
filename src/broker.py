import logging
import requests


logger = logging.getLogger("broker")


class Broker:
    """Class for interacting with the broker server."""

    def __init__(self, url, username, password, shadow_mode=False):
        logger.info(f"Init: {url}, shadow mode: {shadow_mode}")
        self.url = url.strip("/")
        self.username = username
        self.password = password
        self.shadow_mode = shadow_mode
        self.auth_header = None

    def ping(self):
        """Ping the broker server."""
        res = requests.get(f"{self.url}/docs")
        if res.ok:
            logger.info(f"Ping: {res.status_code} {res.reason}")
        else:
            logger.warning(f"Ping: {res.status_code} {res.reason}")
        return res.ok

    def authenticate(self):
        """Authenticate with the broker server and get auth token."""
        res = requests.post(
            f"{self.url}/user_management/authenticate/",
            data={"username": self.username, "password": self.password, "grant_type": "password"},
            headers={"content-type": "application/x-www-form-urlencoded"}
        )
        logger.info(f"Authenticate: {res.status_code} {res.reason}")
        assert res.ok, f"Response not OK: {res.status_code} {res.reason}"
        if res.ok:
            access_token = res.json()["access_token"]
            self.auth_header = {"Authorization": f"Bearer {access_token}"}
        return res.ok

    def get_capabilities(self, available=True):
        """Get capabilities."""
        assert self.auth_header is not None, "Not authenticated."
        res = requests.get(
            self.url + "/get/capabilities",
            params={"currently_available": available},
            headers=self.auth_header,
        )
        if res.ok:
            logger.info(f"Get capabilities: {res.status_code}, {res.reason}")
            capabilities = res.json()
            return capabilities
        else:
            logger.warning(f"Get capabilities: {res.status_code}, {res.reason}")
            return []

    def get_results(self, quantity=None, method=None):
        """Get results."""
        assert self.auth_header is not None, "Not authenticated."
        res = requests.get(
            self.url + "/get/all_results",
            params={"quantity": quantity, "method": method},
            headers=self.auth_header,
        )
        if res.ok:
            logger.info(f"Get results: {res.status_code}, {res.reason}")
            results = res.json()
            # TODO: validate results
            return results
        else:
            logger.warning(f"Get results: {res.status_code}, {res.reason}")
            return []

    def post_result(self, result):
        """Post result."""
        assert self.auth_header is not None, "Not authenticated."
        # TODO: validate result
        if self.shadow_mode:
            logger.info("Post result: shadow mode")
            return "result_shadow_id"
        res = requests.post(
            self.url + "/post/result",
            json=result,
            headers=self.auth_header,
        )
        if res.ok:
            logger.info(f"Post result: {res.status_code}, {res.reason}, {res.json()}")
            result_id = res.json()
            return result_id
        else:
            logger.warning(f"Post result: {res.status_code}, {res.reason}")
            return None

    def get_pending_requests(self):
        """Get pending requests."""
        assert self.auth_header is not None, "Not authenticated."
        res = requests.get(
            self.url + "/get/pending_requests",
            headers=self.auth_header,
        )
        if res.ok:
            logger.info(f"Get pending requests: {res.status_code}, {res.reason}")
            requests_ = res.json()
            # TODO: validate requests_
            return requests_
        else:
            logger.warning(f"Get pending requests: {res.status_code}, {res.reason}")
            return []

    def post_request(self, request):
        """Post request."""
        assert self.auth_header is not None, "Not authenticated."
        # TODO: validate request
        if self.shadow_mode:
            logger.info("Post request: shadow mode")
            return "request_shadow_id"
        res = requests.post(
            self.url + "/post/request",
            json=request,
            headers=self.auth_header,
        )
        if res.ok:
            logger.info(f"Post request: {res.status_code}, {res.reason}, {res.json()}")
            request_id = res.json()
            return request_id
        else:
            logger.warning(f"Post request: {res.status_code}, {res.reason}")
            return None
