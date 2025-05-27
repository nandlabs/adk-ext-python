import copy
import logging
import time
from typing import Any, Dict, List, Optional, Union
import uuid

from typing_extensions import override
from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentSnapshot
from google.cloud.firestore_v1.base_query import BaseQuery

from google.adk.events import Event
from google.adk.sessions import BaseSessionService, Session, State
from google.adk.sessions.base_session_service import (
    GetSessionConfig,
    ListSessionsResponse,
)

logger = logging.getLogger("adk_ext." + __name__)


class FirestoreSessionService(BaseSessionService):
    """A session service that uses Firestore as the backend."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        collection_name: str = "adk_sessions",
        events_collection_name: str = "adk_events",
    ):
        """Initialize the FirestoreSessionService.

        Args:
            project_id: The Google Cloud project ID. If None, the default project will be used.
            collection_name: The name of the Firestore collection to store sessions.
            events_collection_name: The name of the Firestore collection to store events.
        """
        self.db = firestore.Client(project=project_id)
        self.collection = self.db.collection(collection_name)
        self.events_collection_name = events_collection_name

    def _get_session_doc_ref(self, app_name: str, user_id: str, session_id: str):
        """Get the document reference for a session.

        Args:
            app_name: The name of the application.
            user_id: The ID of the user.
            session_id: The ID of the session.

        Returns:
            The document reference for the session.
        """
        return self.collection.document(f"{app_name}:{user_id}:{session_id}")

    def _get_events_collection_ref(self, app_name: str, user_id: str, session_id: str):
        """Get the collection reference for events in a session.

        Args:
            app_name: The name of the application.
            user_id: The ID of the user.
            session_id: The ID of the session.

        Returns:
            The collection reference for the events in the session.
        """
        return self.db.collection(
            f"{self.events_collection_name}/{app_name}:{user_id}:{session_id}/events"
        )

    def _doc_to_session(
        self, doc_snapshot: DocumentSnapshot, include_events: bool = False
    ) -> Session:
        """Convert a Firestore document to a Session object.

        Args:
            doc_snapshot: The Firestore document snapshot.
            include_events: Whether to include events in the session.

        Returns:
            The Session object.
        """
        if not doc_snapshot.exists:
            return None

        data = doc_snapshot.to_dict()
        session_id = data.get("session_id")
        app_name = data.get("app_name")
        user_id = data.get("user_id")
        state_data = data.get("state", {})
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")
        last_update_time = (updated_at or created_at) / 1000.0  # Convert to seconds

        # Create Session object directly with state_data dictionary
        try:
            session = Session(
                id=session_id,
                app_name=app_name,
                user_id=user_id,
                state=state_data,
                events=[],
                last_update_time=float(last_update_time),
            )
        except Exception as e:
            logger.error(f"Error creating session from document: {e}")
            return None

        # Include events if requested
        if include_events:
            events_ref = self._get_events_collection_ref(app_name, user_id, session_id)
            events_query = events_ref.order_by(
                "timestamp", direction=firestore.Query.ASCENDING
            )
            events_docs = events_query.stream()

            for event_doc in events_docs:
                event_data = event_doc.to_dict()
                event = Event(
                    **event_data,  # Unpack the event data
                )
                session.events.append(event)

        return session

    @override
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """Create a new session.

        Args:
            app_name: The name of the application.
            user_id: The ID of the user.
            state: Optional initial state data.
            session_id: Optional session ID. If not provided, a new ID will be generated.

        Returns:
            The created session.

        Raises:
            ValueError: If a session with the same ID already exists.
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Create timestamp
        timestamp = int(time.time() * 1000)  # milliseconds since epoch

        # Create session data
        session_data = {
            "session_id": session_id,
            "app_name": app_name,
            "user_id": user_id,
            "state": state or {},
            "created_at": timestamp,
            "updated_at": timestamp,
        }

        # Use a transaction to ensure atomicity
        doc_ref = self._get_session_doc_ref(app_name, user_id, session_id)

        @firestore.transactional
        def create_session_transaction(transaction, doc_ref, session_data):
            doc_snapshot = doc_ref.get(transaction=transaction)
            if doc_snapshot.exists:
                error_msg = f"Session already exists: {app_name}:{user_id}:{session_id}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            transaction.set(doc_ref, session_data)
            return True

        # Execute the transaction
        transaction = self.db.transaction()
        create_session_transaction(transaction, doc_ref, session_data)

        # Create a Session object directly with state dictionary
        session = Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=state or {},
            events=[],
            last_update_time=float(timestamp / 1000),  # Convert to seconds
        )

        logger.info(f"Created session: {app_name}:{user_id}:{session_id}")
        return session

    @override
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """Get a session.

        Args:
            app_name: The name of the application.
            user_id: The ID of the user.
            session_id: The ID of the session.
            config: Optional configuration for retrieving the session.

        Returns:
            The session if found, None otherwise.
        """
        # Validate inputs
        if (
            not app_name
            or not user_id
            or not session_id
            or app_name == ""
            or user_id == ""
            or session_id == ""
        ):
            error_msg = (
                "App name, user ID, and session ID must be provided to get a session."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get session document
        doc_ref = self._get_session_doc_ref(app_name, user_id, session_id)
        try:
            # Use a transaction for a consistent read of the session
            @firestore.transactional
            def get_session_transaction(transaction, doc_ref):
                doc_snapshot = doc_ref.get(transaction=transaction)
                if not doc_snapshot.exists:
                    return None

                # Convert document to Session object
                # Note: We can't use transaction for events query, so we'll handle it separately
                session = self._doc_to_session(doc_snapshot, include_events=False)
                return session

            # Execute the transaction to get the session
            transaction = self.db.transaction()
            session = get_session_transaction(transaction, doc_ref)

            if session is None:
                logger.warning(f"Session not found: {app_name}:{user_id}:{session_id}")
                return None

            # If session exists and we need to include events, fetch them separately
            # (can't be done within the same transaction due to Firestore limitations)
            include_events = True  # Default to always include events
            if include_events:
                events_ref = self._get_events_collection_ref(
                    app_name, user_id, session_id
                )
                events_query = events_ref.order_by(
                    "timestamp", direction=firestore.Query.ASCENDING
                )
                events_docs = events_query.stream()

                for event_doc in events_docs:
                    event_data = event_doc.to_dict()
                    event = Event(**event_data)
                    session.events.append(event)

            logger.debug(f"Retrieved session: {app_name}:{user_id}:{session_id}")
            return session

        except Exception as e:
            logger.error(
                f"Error getting session {app_name}:{user_id}:{session_id}: {e}"
            )
            return None

    @override
    async def list_sessions(
        self, *, app_name: str, user_id: str
    ) -> ListSessionsResponse:
        """List all sessions for an app and user.

        Args:
            app_name: The name of the application.
            user_id: The ID of the user.

        Returns:
            ListSessionsResponse containing the sessions.
        """
        # Create query to filter sessions by app_name and user_id
        if not app_name or not user_id:
            logger.error("App name and user ID must be provided to list sessions.")
            raise ValueError("App name and user ID must be provided to list sessions.")

        try:
            # For list operations, we need to use a snapshot to get a consistent view
            # of the collection at a single point in time
            query = self.collection.where(
                filter=firestore.FieldFilter("app_name", "==", app_name)
            ).where(filter=firestore.FieldFilter("user_id", "==", user_id))

            # Get a snapshot of the query for consistency
            query_snapshot = query.get()
            sessions = []

            for doc_snapshot in query_snapshot:
                session = self._doc_to_session(doc_snapshot, include_events=True)
                if session:
                    sessions.append(session)

            logger.debug(f"Listed {len(sessions)} sessions for {app_name}:{user_id}")
            return ListSessionsResponse(sessions=sessions)

        except Exception as e:
            logger.error(f"Error listing sessions for {app_name}:{user_id}: {e}")
            return ListSessionsResponse(sessions=[])

    @override
    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        """Delete a session and its associated events.

        Args:
            app_name: The name of the application.
            user_id: The ID of the user.
            session_id: The ID of the session.

        Raises:
            ValueError: If app_name, user_id, or session_id is empty or None.
        """
        # Validate inputs
        if (
            not app_name
            or not user_id
            or not session_id
            or app_name == ""
            or user_id == ""
            or session_id == ""
        ):
            error_msg = "App name, user ID, and session ID must be provided to delete a session."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get session document reference
        doc_ref = self._get_session_doc_ref(app_name, user_id, session_id)

        # Get events collection reference
        events_ref = self._get_events_collection_ref(app_name, user_id, session_id)

        try:
            # Use a transaction to mark the session for deletion
            # This ensures that the session is not modified during the deletion process
            @firestore.transactional
            def mark_session_for_deletion(transaction, doc_ref):
                # Verify the session exists
                session_snapshot = doc_ref.get(transaction=transaction)
                if not session_snapshot.exists:
                    return False

                # Mark the session as being deleted
                transaction.update(
                    doc_ref,
                    {
                        "status": "deleting",
                        "deletion_started_at": int(
                            time.time() * 1000
                        ),  # milliseconds since epoch
                    },
                )
                return True

            # Execute the transaction to mark the session for deletion
            transaction = self.db.transaction()
            session_exists = mark_session_for_deletion(transaction, doc_ref)

            if not session_exists:
                logger.warning(
                    f"Cannot delete nonexistent session: {app_name}:{user_id}:{session_id}"
                )
                return

            # Delete all events in batches
            batch_size = 500  # Maximum batch size for Firestore

            try:
                # Function to delete a batch of events
                def delete_event_batch():
                    batch = self.db.batch()
                    docs = events_ref.limit(batch_size).stream()
                    deleted = 0

                    for doc in docs:
                        batch.delete(doc.reference)
                        deleted += 1

                    if deleted > 0:
                        batch.commit()

                    return deleted

                # Delete events in batches until no more events are left
                deleted_count = delete_event_batch()
                total_deleted = deleted_count

                # Keep deleting in batches until there are no more events
                while deleted_count > 0:
                    deleted_count = delete_event_batch()
                    total_deleted += deleted_count

                # Now that events are deleted, delete the session document
                @firestore.transactional
                def delete_session_transaction(transaction, doc_ref):
                    # Verify the session is still marked for deletion
                    session_snapshot = doc_ref.get(transaction=transaction)
                    if not session_snapshot.exists:
                        return False

                    session_data = session_snapshot.to_dict()
                    if session_data.get("status") != "deleting":
                        # Someone else may have modified the session, which is unexpected
                        logger.warning(
                            f"Session no longer marked for deletion: {app_name}:{user_id}:{session_id}"
                        )
                        return False

                    # Delete the session document
                    transaction.delete(doc_ref)
                    return True

                # Execute the transaction to delete the session
                transaction = self.db.transaction()
                delete_session_transaction(transaction, doc_ref)

                logger.info(
                    f"Deleted session: {app_name}:{user_id}:{session_id} with {total_deleted} events"
                )

            except Exception as e:
                # If an error occurs during event deletion or session deletion,
                # log the error, remove the deletion marker, and raise the exception
                logger.error(
                    f"Error during event deletion, attempting to restore session state: {e}"
                )
                try:

                    @firestore.transactional
                    def restore_session_state(transaction, doc_ref):
                        session_snapshot = doc_ref.get(transaction=transaction)
                        if not session_snapshot.exists:
                            return False

                        session_data = session_snapshot.to_dict()
                        if session_data.get("status") == "deleting":
                            # Remove the deletion marker
                            transaction.update(
                                doc_ref,
                                {
                                    "status": "active",
                                    "deletion_started_at": firestore.DELETE_FIELD,
                                },
                            )
                        return True

                    # Execute the transaction to restore session state
                    transaction = self.db.transaction()
                    restore_session_state(transaction, doc_ref)
                    logger.info(
                        f"Restored session state for {app_name}:{user_id}:{session_id}"
                    )
                except Exception as restore_error:
                    logger.error(f"Failed to restore session state: {restore_error}")

                # Re-raise the original exception
                raise

        except Exception as e:
            logger.error(
                f"Error deleting session {app_name}:{user_id}:{session_id}: {e}"
            )
            raise

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        """Append an event to a session.

        Args:
            session: The session to append the event to.
            event: The event to append.

        Returns:
            The appended event.

        Raises:
            ValueError: If the session does not exist or inputs are invalid.
        """
        # Validate inputs first before any processing
        if not session:
            raise ValueError("Session must be provided to append an event.")
        if not event:
            raise ValueError("Event must be provided to append to session.")

        # If event doesn't have an ID, generate one
        if not event.id:
            event.id = str(uuid.uuid4())

        # If timestamp is not set, set it to current time
        if not event.timestamp:
            event.timestamp = int(time.time() * 1000)  # milliseconds since epoch

        # Prepare event data
        event_data = event.model_dump()

        try:
            # Get document references
            session_ref = self._get_session_doc_ref(
                session.app_name, session.user_id, session.id
            )
            event_doc_ref = self._get_events_collection_ref(
                session.app_name, session.user_id, session.id
            ).document(event.id)

            # Use a transaction to ensure both operations succeed or fail together
            @firestore.transactional
            def append_event_transaction(
                transaction, session_ref, event_doc_ref, event_data, timestamp
            ):
                # Verify the session exists
                session_snapshot = session_ref.get(transaction=transaction)
                if not session_snapshot.exists:
                    error_msg = f"Session does not exist: {session.app_name}:{session.user_id}:{session.id}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Add the event
                transaction.set(event_doc_ref, event_data)

                # Update the session's updated_at timestamp
                transaction.update(session_ref, {"updated_at": timestamp})
                return True

            # Execute the transaction
            transaction = self.db.transaction()
            append_event_transaction(
                transaction, session_ref, event_doc_ref, event_data, event.timestamp
            )

            # Update in-memory session object
            # Note: BaseSessionService.append_event is a coroutine, so we need to await it
            await super().append_event(session=session, event=event)

            logger.debug(f"Appended event {event.id} to session {session.id}")
            return event
        except Exception as e:
            if session:
                logger.error(f"Error appending event to session {session.id}: {e}")
            else:
                logger.error(f"Error appending event to session: {e}")
            raise
