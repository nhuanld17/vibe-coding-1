"""
Vector Database service using Qdrant for Missing Person AI system.

This module provides vector database operations for storing and searching
face embeddings with metadata using Qdrant vector database.
"""

import uuid
import numpy as np
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)
from loguru import logger


class VectorDatabaseService:
    """
    Vector database service for storing and searching face embeddings.
    
    This class provides methods for:
    - Connecting to Qdrant vector database
    - Creating collections for missing and found persons
    - Inserting embeddings with metadata
    - Searching for similar faces with filters
    - Managing database operations (CRUD)
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 6333, 
        api_key: Optional[str] = None,
        timeout: float = 60.0
    ) -> None:
        """
        Initialize the vector database service.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            
        Raises:
            RuntimeError: If connection to Qdrant fails
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.timeout = timeout
        
        # Collection names
        self.missing_collection = "missing_persons"
        self.found_collection = "found_persons"
        
        try:
            # Initialize Qdrant client
            # Use https=False for Docker internal connections
            self.client = QdrantClient(
                host=host,
                port=port,
                api_key=api_key,
                timeout=timeout,
                https=False,
                prefer_grpc=False
            )
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant at {host}:{port}")
            logger.info(f"Existing collections: {[c.name for c in collections.collections]}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise RuntimeError(f"Qdrant connection failed: {str(e)}")
    
    def initialize_collections(self, vector_size: int = 512) -> None:
        """
        Initialize collections for missing and found persons if they don't exist.
        
        Args:
            vector_size: Dimension of face embedding vectors
            
        Raises:
            RuntimeError: If collection creation fails
        """
        try:
            # Get existing collections
            collections = self.client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            # Create missing persons collection
            if self.missing_collection not in existing_names:
                self.client.create_collection(
                    collection_name=self.missing_collection,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.missing_collection}")
            else:
                logger.info(f"Collection already exists: {self.missing_collection}")
            
            # Create found persons collection
            if self.found_collection not in existing_names:
                self.client.create_collection(
                    collection_name=self.found_collection,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.found_collection}")
            else:
                logger.info(f"Collection already exists: {self.found_collection}")
                
        except Exception as e:
            logger.error(f"Failed to initialize collections: {str(e)}")
            raise RuntimeError(f"Collection initialization failed: {str(e)}")
    
    def insert_missing_person(
        self, 
        embedding: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> str:
        """
        Insert a missing person's embedding and metadata.
        
        Args:
            embedding: Face embedding vector (512-dim)
            metadata: Metadata dictionary with person information
            
        Returns:
            Point ID of the inserted record
            
        Raises:
            ValueError: If embedding or metadata is invalid
            RuntimeError: If insertion fails
            
        Example:
            >>> metadata = {
            ...     'case_id': 'MISS_2023_001',
            ...     'name': 'John Doe',
            ...     'age_at_disappearance': 25,
            ...     'year_disappeared': 2020,
            ...     'gender': 'male',
            ...     'location_last_seen': 'New York, NY'
            ... }
            >>> point_id = db.insert_missing_person(embedding, metadata)
        """
        try:
            # Validate embedding
            if embedding is None or len(embedding) == 0:
                raise ValueError("Embedding cannot be None or empty")
            
            if len(embedding) != 512:
                raise ValueError(f"Embedding must be 512-dimensional, got {len(embedding)}")
            
            # Validate required metadata fields
            required_fields = ['case_id', 'name', 'age_at_disappearance', 'year_disappeared', 'gender']
            for field in required_fields:
                if field not in metadata:
                    raise ValueError(f"Missing required metadata field: {field}")
            
            # Generate point ID
            point_id = str(uuid.uuid4())
            
            # Add system metadata
            current_time = datetime.utcnow().isoformat()
            enhanced_metadata = metadata.copy()
            enhanced_metadata.update({
                'collection_type': 'missing',
                'upload_timestamp': current_time,
                'point_id': point_id
            })
            
            # Calculate estimated current age
            current_year = datetime.now().year
            years_missing = current_year - metadata['year_disappeared']
            enhanced_metadata['estimated_current_age'] = metadata['age_at_disappearance'] + years_missing
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=enhanced_metadata
            )
            
            # Insert into collection
            self.client.upsert(
                collection_name=self.missing_collection,
                points=[point]
            )
            
            logger.info(f"Inserted missing person: {metadata.get('name', 'Unknown')} (ID: {point_id})")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to insert missing person: {str(e)}")
            raise RuntimeError(f"Insertion failed: {str(e)}")
    
    def insert_found_person(
        self, 
        embedding: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> str:
        """
        Insert a found person's embedding and metadata.
        
        Args:
            embedding: Face embedding vector (512-dim)
            metadata: Metadata dictionary with person information
            
        Returns:
            Point ID of the inserted record
            
        Raises:
            ValueError: If embedding or metadata is invalid
            RuntimeError: If insertion fails
            
        Example:
            >>> metadata = {
            ...     'found_id': 'FOUND_2023_001',
            ...     'current_age_estimate': 30,
            ...     'gender': 'male',
            ...     'current_location': 'Los Angeles, CA',
            ...     'finder_contact': 'finder@email.com'
            ... }
            >>> point_id = db.insert_found_person(embedding, metadata)
        """
        try:
            # Validate embedding
            if embedding is None or len(embedding) == 0:
                raise ValueError("Embedding cannot be None or empty")
            
            if len(embedding) != 512:
                raise ValueError(f"Embedding must be 512-dimensional, got {len(embedding)}")
            
            # Validate required metadata fields
            required_fields = ['found_id', 'current_age_estimate', 'gender', 'finder_contact']
            for field in required_fields:
                if field not in metadata:
                    raise ValueError(f"Missing required metadata field: {field}")
            
            # Generate point ID
            point_id = str(uuid.uuid4())
            
            # Add system metadata
            current_time = datetime.utcnow().isoformat()
            enhanced_metadata = metadata.copy()
            enhanced_metadata.update({
                'collection_type': 'found',
                'upload_timestamp': current_time,
                'point_id': point_id
            })
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=enhanced_metadata
            )
            
            # Insert into collection
            self.client.upsert(
                collection_name=self.found_collection,
                points=[point]
            )
            
            logger.info(f"Inserted found person: {metadata.get('found_id', 'Unknown')} (ID: {point_id})")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to insert found person: {str(e)}")
            raise RuntimeError(f"Insertion failed: {str(e)}")
    
    def search_similar_faces(
        self,
        query_embedding: np.ndarray,
        collection_name: str,
        limit: int = 10,
        score_threshold: float = 0.65,
        filters: Optional[Dict[str, Any]] = None,
        with_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar faces in the specified collection.
        
        Args:
            query_embedding: Query face embedding vector (512-dim)
            collection_name: Name of collection to search ('missing_persons' or 'found_persons')
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1)
            filters: Optional metadata filters
            with_vectors: If True, include embedding vectors in results (needed for multi-image aggregation)
            
        Returns:
            List of matching results with scores and metadata (and optionally vectors)
            
        Raises:
            ValueError: If query_embedding has wrong shape
            RuntimeError: If collection doesn't exist or search fails
            
        Example:
            >>> results = db.search_similar_faces(
            ...     embedding,
            ...     "missing_persons",
            ...     limit=5,
            ...     filters={'gender': 'male', 'age_range': [20, 40]},
            ...     with_vectors=True  # Include vectors for multi-image aggregation
            ... )
        """
        try:
            # Validate embedding
            if query_embedding is None or len(query_embedding) == 0:
                raise ValueError("Query embedding cannot be None or empty")
            
            if len(query_embedding) != 512:
                raise ValueError(f"Query embedding must be 512-dimensional, got {len(query_embedding)}")
            
            # Validate collection name
            if collection_name not in [self.missing_collection, self.found_collection]:
                raise ValueError(f"Invalid collection name: {collection_name}")
            
            # Build filters
            search_filter = None
            if filters:
                conditions = []
                
                for key, value in filters.items():
                    if key == 'age_range' and isinstance(value, list) and len(value) == 2:
                        # Age range filter
                        age_field = 'estimated_current_age' if collection_name == self.missing_collection else 'current_age_estimate'
                        conditions.append(
                            FieldCondition(
                                key=age_field,
                                range=models.Range(gte=value[0], lte=value[1])
                            )
                        )
                    elif isinstance(value, (str, int, float, bool)):
                        # Exact match filter
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                    elif isinstance(value, list):
                        # Multiple values filter (OR condition)
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(any=value)
                            )
                        )
                
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform search
            from qdrant_client.models import SearchRequest
            search_results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding.tolist(),
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_vectors=with_vectors  # Pass through with_vectors parameter
            ).points
            
            # Format results
            results = []
            for result in search_results:
                result_dict = {
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                }
                
                # Include vector if requested (needed for multi-image aggregation)
                if with_vectors and hasattr(result, 'vector') and result.vector is not None:
                    result_dict['vector'] = result.vector
                
                results.append(result_dict)
            
            logger.debug(f"Found {len(results)} similar faces in {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise RuntimeError(f"Search failed: {str(e)}")
    
    def search_by_metadata(self, collection_name: str, filters: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for points by metadata only (no vector search).
        
        Args:
            collection_name: Name of the collection
            filters: Dictionary of metadata filters
            limit: Maximum number of results
            
        Returns:
            List of matching points
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            
            search_filter = Filter(must=conditions) if conditions else None
            
            # Use scroll to get points by filter
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )
            
            # Format results
            results = []
            for point in scroll_result[0]:  # scroll returns (points, next_page_offset)
                results.append({
                    'id': str(point.id),
                    'score': 1.0,  # No similarity score for metadata-only search
                    'payload': point.payload,
                    'vector': point.vector
                })
            
            logger.debug(f"Found {len(results)} points matching filters in {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Metadata search failed: {str(e)}")
            raise RuntimeError(f"Metadata search failed: {str(e)}")
    
    def list_all_points(
        self, 
        collection_name: str, 
        limit: int = 100, 
        offset: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all points in a collection (paginated).
        
        Args:
            collection_name: Name of the collection
            limit: Maximum number of results to return
            offset: Optional offset for pagination (next_page_offset from previous scroll)
            
        Returns:
            Tuple of (list of points, next_page_offset or None)
        """
        try:
            # Validate collection name
            if collection_name not in [self.missing_collection, self.found_collection]:
                raise ValueError(f"Invalid collection name: {collection_name}")
            
            # Use scroll to get all points
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=None,  # No filter - get all points
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need vectors for listing
            )
            
            # Format results
            points = []
            for point in scroll_result[0]:  # scroll returns (points, next_page_offset)
                points.append({
                    'id': str(point.id),
                    'payload': point.payload
                })
            
            next_offset = scroll_result[1] if scroll_result[1] else None
            
            logger.debug(f"Listed {len(points)} points from {collection_name}")
            return points, next_offset
            
        except Exception as e:
            logger.error(f"List all points failed: {str(e)}")
            raise RuntimeError(f"List all points failed: {str(e)}")
    
    def delete_point(self, collection_name: str, point_id: str) -> bool:
        """
        Delete a point from the specified collection.
        
        Args:
            collection_name: Name of the collection
            point_id: ID of the point to delete
            
        Returns:
            True if deletion was successful
            
        Raises:
            ValueError: If collection name is invalid
            RuntimeError: If deletion fails
        """
        try:
            # Validate collection name
            if collection_name not in [self.missing_collection, self.found_collection]:
                raise ValueError(f"Invalid collection name: {collection_name}")
            
            # Delete point
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
            
            logger.info(f"Deleted point {point_id} from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete point {point_id}: {str(e)}")
            raise RuntimeError(f"Deletion failed: {str(e)}")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary containing collection statistics
            
        Raises:
            ValueError: If collection name is invalid
            RuntimeError: If stats retrieval fails
        """
        try:
            # Validate collection name
            if collection_name not in [self.missing_collection, self.found_collection]:
                raise ValueError(f"Invalid collection name: {collection_name}")
            
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            
            stats = {
                'collection_name': collection_name,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.value,
                'status': collection_info.status.value
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for {collection_name}: {str(e)}")
            raise RuntimeError(f"Stats retrieval failed: {str(e)}")
    
    def get_point_by_id(self, collection_name: str, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific point by ID.
        
        Args:
            collection_name: Name of the collection
            point_id: ID of the point to retrieve
            
        Returns:
            Point data or None if not found
            
        Raises:
            ValueError: If collection name is invalid
            RuntimeError: If retrieval fails
        """
        try:
            # Validate collection name
            if collection_name not in [self.missing_collection, self.found_collection]:
                raise ValueError(f"Invalid collection name: {collection_name}")
            
            # Retrieve point
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return None
            
            point = points[0]
            return {
                'id': point.id,
                'vector': point.vector,
                'payload': point.payload
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve point {point_id}: {str(e)}")
            raise RuntimeError(f"Point retrieval failed: {str(e)}")
    
    def update_point_metadata(
        self, 
        collection_name: str, 
        point_id: str, 
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a specific point.
        
        Args:
            collection_name: Name of the collection
            point_id: ID of the point to update
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            True if update was successful
            
        Raises:
            ValueError: If collection name is invalid
            RuntimeError: If update fails
        """
        try:
            # Validate collection name
            if collection_name not in [self.missing_collection, self.found_collection]:
                raise ValueError(f"Invalid collection name: {collection_name}")
            
            # Update point payload
            self.client.set_payload(
                collection_name=collection_name,
                payload=metadata_updates,
                points=[point_id]
            )
            
            logger.info(f"Updated metadata for point {point_id} in {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update point {point_id}: {str(e)}")
            raise RuntimeError(f"Update failed: {str(e)}")
    
    def insert_batch(
        self,
        collection_name: str,
        embeddings: List[Optional[np.ndarray]],
        payloads: List[Dict[str, Any]],
        point_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Insert multiple embeddings in a single batch operation.
        
        UPDATED: Now handles None embeddings for reference-only images.
        Reference images (embedding=None) are stored with metadata only.
        
        Args:
            collection_name: Name of the collection ('missing_persons' or 'found_persons')
            embeddings: List of 512-D face embeddings (may contain None for reference images)
            payloads: List of metadata dictionaries (one per embedding)
            point_ids: Optional list of point IDs (auto-generated if None)
            
        Returns:
            List of point IDs for the inserted records
            
        Raises:
            ValueError: If inputs are invalid or mismatched lengths
            RuntimeError: If batch insertion fails
            
        Example:
            >>> embeddings = [emb1, None, emb3]  # Image 2 is reference-only
            >>> payloads = [
            ...     {'case_id': 'MISS_001', 'is_valid_for_matching': True, ...},
            ...     {'case_id': 'MISS_001', 'is_valid_for_matching': False, ...},  # Reference
            ...     {'case_id': 'MISS_001', 'is_valid_for_matching': True, ...}
            ... ]
            >>> point_ids = db.insert_batch('missing_persons', embeddings, payloads)
        """
        try:
            # Validate inputs
            if not embeddings or not payloads:
                raise ValueError("embeddings and payloads cannot be empty")
            
            if len(embeddings) != len(payloads):
                raise ValueError(
                    f"Length mismatch: {len(embeddings)} embeddings vs {len(payloads)} payloads"
                )
            
            if point_ids is not None and len(point_ids) != len(embeddings):
                raise ValueError(
                    f"Length mismatch: {len(point_ids)} point_ids vs {len(embeddings)} embeddings"
                )
            
            # Validate collection name
            if collection_name not in [self.missing_collection, self.found_collection]:
                raise ValueError(f"Invalid collection name: {collection_name}")
            
            # Generate point IDs if not provided
            if point_ids is None:
                point_ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
            
            # ═══════════════════════════════════════════════════════════════════
            # NEW: Handle None embeddings (reference-only images)
            # ═══════════════════════════════════════════════════════════════════
            valid_points = []  # Points with embeddings
            reference_points = []  # Points without embeddings (metadata only)
            
            for idx, (embedding, payload) in enumerate(zip(embeddings, payloads)):
                # Add system metadata
                enhanced_payload = payload.copy()
                if 'upload_timestamp' not in enhanced_payload:
                    enhanced_payload['upload_timestamp'] = datetime.utcnow().isoformat()
                enhanced_payload['point_id'] = point_ids[idx]
                
                # Check if embedding is None (reference image)
                if embedding is None:
                    # Reference-only image - store metadata with dummy zero vector
                    logger.debug(f"Point {point_ids[idx]}: Reference image (no embedding), using zero vector")
                    reference_points.append(
                        PointStruct(
                            id=point_ids[idx],
                            vector=np.zeros(512).tolist(),  # Dummy zero vector
                            payload=enhanced_payload
                        )
                    )
                else:
                    # Validate embedding
                    if len(embedding) != 512:
                        raise ValueError(
                            f"Embedding at index {idx} has invalid shape: {embedding.shape}"
                        )
                    
                    # Valid image with embedding
                    valid_points.append(
                        PointStruct(
                            id=point_ids[idx],
                            vector=embedding.tolist(),
                            payload=enhanced_payload
                        )
                    )
            
            # Batch insert all points (valid + reference)
            all_points = valid_points + reference_points
            
            if all_points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=all_points
                )
                
                logger.info(
                    f"Batch inserted {len(all_points)} records to {collection_name}: "
                    f"{len(valid_points)} valid, {len(reference_points)} reference. "
                    f"Case: {payloads[0].get('case_id') or payloads[0].get('found_id', 'unknown')}"
                )
            
            return point_ids
            
        except Exception as e:
            logger.error(f"Batch insertion failed: {str(e)}")
            raise RuntimeError(f"Batch insertion failed: {str(e)}")
    
    def get_all_images_for_person(
        self,
        collection_name: str,
        case_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all images/points for a specific person.
        
        Args:
            collection_name: Name of the collection
            case_id: Case ID or Found ID of the person
            
        Returns:
            List of image dictionaries with id, vector, and payload
            
        Raises:
            ValueError: If collection name is invalid
            RuntimeError: If retrieval fails
            
        Example:
            >>> images = db.get_all_images_for_person('missing_persons', 'MISS_001')
            >>> print(f"Person has {len(images)} images")
            >>> for img in images:
            ...     print(f"  - {img['payload']['image_id']} at age {img['payload']['age_at_photo']}")
        """
        try:
            # Validate collection name
            if collection_name not in [self.missing_collection, self.found_collection]:
                raise ValueError(f"Invalid collection name: {collection_name}")
            
            # Build filter for case_id
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            scroll_filter = Filter(
                must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
            )
            
            # Scroll through all matching points
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=100,  # Should be enough for max 10 images per person
                with_payload=True,
                with_vectors=True
            )
            
            # Format results
            images = []
            for point in results[0]:  # scroll returns (points, next_page_offset)
                images.append({
                    'id': str(point.id),
                    'vector': point.vector,
                    'payload': point.payload
                })
            
            logger.debug(f"Retrieved {len(images)} images for person {case_id}")
            return images
            
        except Exception as e:
            logger.error(f"Failed to get images for person {case_id}: {str(e)}")
            raise RuntimeError(f"Retrieval failed: {str(e)}")
    
    def delete_person(
        self,
        collection_name: str,
        case_id: str
    ) -> bool:
        """
        Delete ALL images for a specific person.
        
        Args:
            collection_name: Name of the collection
            case_id: Case ID or Found ID of the person
            
        Returns:
            True if deletion was successful
            
        Raises:
            ValueError: If collection name is invalid
            RuntimeError: If deletion fails
            
        Example:
            >>> db.delete_person('missing_persons', 'MISS_001')
            >>> print("All images for MISS_001 deleted")
        """
        try:
            # Validate collection name
            if collection_name not in [self.missing_collection, self.found_collection]:
                raise ValueError(f"Invalid collection name: {collection_name}")
            
            # Build filter for case_id
            from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
            
            delete_filter = Filter(
                must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
            )
            
            # Delete all points matching the filter
            self.client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(filter=delete_filter)
            )
            
            logger.info(f"Deleted all images for person {case_id} from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete person {case_id}: {str(e)}")
            raise RuntimeError(f"Deletion failed: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the Qdrant connection and collections.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            # Test connection
            collections = self.client.get_collections()
            
            # Check collections
            missing_stats = self.get_collection_stats(self.missing_collection)
            found_stats = self.get_collection_stats(self.found_collection)
            
            health_info = {
                'status': 'healthy',
                'host': self.host,
                'port': self.port,
                'collections': {
                    'missing_persons': missing_stats,
                    'found_persons': found_stats
                },
                'total_collections': len(collections.collections),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return health_info
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize vector database service
        db = VectorDatabaseService(host="localhost", port=6333)
        
        # Initialize collections
        db.initialize_collections(vector_size=512)
        
        # Create dummy embedding and metadata for testing
        dummy_embedding = np.random.rand(512).astype(np.float32)
        
        # Test missing person insertion
        missing_metadata = {
            'case_id': 'MISS_TEST_001',
            'name': 'Test Person',
            'age_at_disappearance': 25,
            'year_disappeared': 2020,
            'gender': 'male',
            'location_last_seen': 'Test City',
            'contact': 'test@example.com'
        }
        
        point_id = db.insert_missing_person(dummy_embedding, missing_metadata)
        print(f"Inserted missing person with ID: {point_id}")
        
        # Test search
        search_results = db.search_similar_faces(
            dummy_embedding,
            "missing_persons",
            limit=5,
            score_threshold=0.5
        )
        print(f"Found {len(search_results)} similar faces")
        
        # Test collection stats
        stats = db.get_collection_stats("missing_persons")
        print(f"Collection stats: {stats}")
        
        # Test health check
        health = db.health_check()
        print(f"Health status: {health['status']}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")