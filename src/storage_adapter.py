"""
Storage Adapter - Abstract storage layer for local and Cloud Storage
Allows seamless switching between local development and cloud deployment
"""

import json
import os
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class StorageAdapter:
    """
    Abstract storage - works with both local filesystem and Google Cloud Storage
    """

    def __init__(self, use_cloud: bool = False, bucket_name: str = None):
        """
        Initialize storage adapter

        Args:
            use_cloud: If True, use Google Cloud Storage. If False, use local filesystem.
            bucket_name: Name of GCS bucket (required if use_cloud=True)
        """
        self.use_cloud = use_cloud

        if use_cloud:
            try:
                from google.cloud import storage
                self.client = storage.Client()
                self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME', '')
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"Cloud Storage initialized: {self.bucket_name}")
            except ImportError:
                logger.error("google-cloud-storage not installed. Run: pip install google-cloud-storage")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Cloud Storage: {e}")
                raise
        else:
            # Local mode - use paths from config
            from config import ROOT_DIR
            self.root_dir = ROOT_DIR
            logger.info("Local storage initialized")

    def read_json(self, path: str) -> dict:
        """
        Read JSON file from local or cloud storage

        Args:
            path: Relative path to file (e.g., 'topics/philosophy.json')

        Returns:
            Parsed JSON as dictionary
        """
        if self.use_cloud:
            blob = self.bucket.blob(path)
            if not blob.exists():
                logger.warning(f"File not found in Cloud Storage: {path}")
                return None
            content = blob.download_as_text()
            return json.loads(content)
        else:
            # Local filesystem
            full_path = self.root_dir / path
            if not full_path.exists():
                logger.warning(f"File not found locally: {full_path}")
                return None
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)

    def write_json(self, path: str, data: dict, indent: int = 2):
        """
        Write JSON file to local or cloud storage

        Args:
            path: Relative path to file (e.g., 'topics/philosophy.json')
            data: Dictionary to write as JSON
            indent: JSON indentation (default: 2)
        """
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)

        if self.use_cloud:
            blob = self.bucket.blob(path)
            blob.upload_from_string(json_str, content_type='application/json')
            logger.debug(f"Wrote to Cloud Storage: {path}")
        else:
            # Local filesystem
            full_path = self.root_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.debug(f"Wrote to local storage: {full_path}")

    def read_text(self, path: str) -> str:
        """
        Read text file from local or cloud storage

        Args:
            path: Relative path to file (e.g., 'prompts/meta_system_prompt.txt')

        Returns:
            File content as string
        """
        if self.use_cloud:
            blob = self.bucket.blob(path)
            if not blob.exists():
                logger.warning(f"File not found in Cloud Storage: {path}")
                return ""
            return blob.download_as_text()
        else:
            # Local filesystem
            full_path = self.root_dir / path
            if not full_path.exists():
                logger.warning(f"File not found locally: {full_path}")
                return ""
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()

    def write_text(self, path: str, content: str):
        """
        Write text file to local or cloud storage

        Args:
            path: Relative path to file (e.g., 'prompts/meta_system_prompt.txt')
            content: Text content to write
        """
        if self.use_cloud:
            blob = self.bucket.blob(path)
            blob.upload_from_string(content, content_type='text/plain')
            logger.debug(f"Wrote to Cloud Storage: {path}")
        else:
            # Local filesystem
            full_path = self.root_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.debug(f"Wrote to local storage: {full_path}")

    def write_file(self, path: str, data: bytes, content_type: str = 'application/octet-stream'):
        """
        Write binary file to local or cloud storage

        Args:
            path: Relative path to file (e.g., 'data/pdfs/mybook.pdf')
            data: Binary data to write
            content_type: MIME type (default: 'application/octet-stream')
        """
        if self.use_cloud:
            blob = self.bucket.blob(path)
            blob.upload_from_string(data, content_type=content_type)
            logger.debug(f"Wrote binary file to Cloud Storage: {path}")
        else:
            # Local filesystem
            full_path = self.root_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'wb') as f:
                f.write(data)
            logger.debug(f"Wrote binary file to local storage: {full_path}")

    def list_files(self, prefix: str) -> list[str]:
        """
        List files in directory (or with prefix in cloud)

        Args:
            prefix: Directory path or prefix (e.g., 'topics/')

        Returns:
            List of file paths
        """
        if self.use_cloud:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        else:
            # Local filesystem
            full_path = self.root_dir / prefix
            if not full_path.exists():
                return []
            if full_path.is_file():
                return [prefix]
            return [str(p.relative_to(self.root_dir)) for p in full_path.rglob('*') if p.is_file()]

    def exists(self, path: str) -> bool:
        """
        Check if file exists

        Args:
            path: Relative path to file

        Returns:
            True if file exists, False otherwise
        """
        if self.use_cloud:
            blob = self.bucket.blob(path)
            return blob.exists()
        else:
            full_path = self.root_dir / path
            return full_path.exists()

    def delete_file(self, path: str):
        """
        Delete file from local or cloud storage

        Args:
            path: Relative path to file
        """
        if self.use_cloud:
            blob = self.bucket.blob(path)
            blob.delete()
            logger.info(f"Deleted file from Cloud Storage: {path}")
        else:
            full_path = self.root_dir / path
            if full_path.exists():
                full_path.unlink()
                logger.info(f"Deleted file from local storage: {full_path}")
            else:
                logger.warning(f"File not found for deletion: {full_path}")

    def generate_upload_url(self, path: str, content_type: str = 'application/pdf', expiration_minutes: int = 30) -> dict:
        """
        Generate a signed URL for direct upload to GCS (bypasses web server)

        Args:
            path: Relative path where file will be stored (e.g., 'data/pdfs/mybook.pdf')
            content_type: MIME type of the file
            expiration_minutes: How long the URL is valid

        Returns:
            dict with 'upload_url' and 'gcs_path', or None if not using cloud storage
        """
        if not self.use_cloud:
            logger.warning("Signed URLs only available in cloud mode")
            return None

        from datetime import timedelta
        from google.auth import default
        from google.auth.transport import requests as auth_requests

        blob = self.bucket.blob(path)

        # Get default credentials (works on Cloud Run)
        credentials, project = default()

        # Refresh credentials to get access token
        auth_request = auth_requests.Request()
        credentials.refresh(auth_request)

        # Generate signed URL using IAM signBlob API (works with compute engine credentials)
        upload_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method="PUT",
            content_type=content_type,
            service_account_email=credentials.service_account_email,
            access_token=credentials.token,
        )

        logger.info(f"Generated signed upload URL for: {path}")
        return {
            'upload_url': upload_url,
            'gcs_path': path
        }


# Global storage adapter instance
# Initialize based on environment variable
_storage_adapter = None


def get_storage() -> StorageAdapter:
    """
    Get global storage adapter instance

    Returns:
        StorageAdapter configured for current environment
    """
    global _storage_adapter

    if _storage_adapter is None:
        use_cloud = os.getenv('USE_CLOUD_STORAGE', 'false').lower() == 'true'
        bucket_name = os.getenv('GCS_BUCKET_NAME', '')
        _storage_adapter = StorageAdapter(use_cloud=use_cloud, bucket_name=bucket_name)

    return _storage_adapter
