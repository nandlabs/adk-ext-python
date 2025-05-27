"""Integration tests for FirestoreSessionService.

These tests are designed to run against a real Firestore instance.
They require Google Cloud credentials to be properly set up.

To run these tests:
1. Set up the GOOGLE_APPLICATION_CREDENTIALS environment variable
   pointing to your service account JSON file.
2. Create a test project in Google Cloud with Firestore enabled.
3. Run the tests with the following command:
   FIRESTORE_PROJECT_ID=your-test-project-id pytest -xvs test/integration/test_firestore_session_integration.py
"""

import asyncio
import os
import pytest
import time
import uuid
from typing import Optional, Generator

from google.adk.events import Event
from google.adk.sessions import Session, State

from adk.ext.sessions.firestore_session_svc import FirestoreSessionService


@pytest.fixture
def project_id() -> str:
    """Get the Firestore project ID from environment variables."""
    project_id = os.environ.get("FIRESTORE_PROJECT_ID")
    if not project_id:
        pytest.skip("FIRESTORE_PROJECT_ID environment variable not set")
    return project_id


@pytest.fixture
def collection_prefix() -> str:
    """Generate a unique collection prefix for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def service(
    project_id: str, collection_prefix: str
) -> Generator[FirestoreSessionService, None, None]:
    """Create a FirestoreSessionService instance for testing."""
    service = FirestoreSessionService(
        project_id=project_id,
        collection_name=f"{collection_prefix}_sessions",
        events_collection_name=f"{collection_prefix}_events",
    )
    yield service
    # No cleanup needed, collections are isolated by prefix


@pytest.fixture
def test_data() -> dict:
    """Create test data for sessions."""
    return {
        "app_name": "test-app",
        "user_id": f"test-user-{uuid.uuid4().hex[:8]}",  # Use unique user ID to avoid conflicts
        "state": {"key1": "value1", "key2": 123},
    }


class TestFirestoreSessionIntegration:
    """Integration tests for FirestoreSessionService."""

    @pytest.mark.asyncio
    async def test_create_get_delete_session(
        self, service: FirestoreSessionService, test_data: dict
    ):
        """Test creating, getting, and deleting a session."""
        # Create session
        created_session = await service.create_session(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
            state=test_data["state"],
        )
        assert created_session is not None
        assert created_session.id is not None
        assert created_session.app_name == test_data["app_name"]
        assert created_session.user_id == test_data["user_id"]
        assert created_session.state["key1"] == test_data["state"]["key1"]
        assert created_session.state["key2"] == test_data["state"]["key2"]

        # Get session
        retrieved_session = await service.get_session(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
            session_id=created_session.id,
        )
        assert retrieved_session is not None
        assert retrieved_session.id == created_session.id
        assert retrieved_session.app_name == created_session.app_name
        assert retrieved_session.user_id == created_session.user_id
        assert retrieved_session.state["key1"] == test_data["state"]["key1"]
        assert retrieved_session.state["key2"] == test_data["state"]["key2"]

        # Delete session
        await service.delete_session(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
            session_id=created_session.id,
        )

        # Verify deletion
        deleted_session = await service.get_session(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
            session_id=created_session.id,
        )
        assert deleted_session is None

    @pytest.mark.asyncio
    async def test_append_event(
        self, service: FirestoreSessionService, test_data: dict
    ):
        """Test appending an event to a session."""
        # Create session
        session = await service.create_session(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
            state=test_data["state"],
        )

        # Create an event
        event = Event(
            id=f"event-{uuid.uuid4().hex[:8]}",
            session_id=session.id,
            timestamp=int(time.time() * 1000),
            data={"event_key": "event_value"},
        )

        # Append the event
        appended_event = await service.append_event(
            session=session,
            event=event,
        )
        assert appended_event is not None
        assert appended_event.id == event.id

        # Get the session with event
        retrieved_session = await service.get_session(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
            session_id=session.id,
        )
        assert retrieved_session is not None
        assert len(retrieved_session.events) == 1
        assert retrieved_session.events[0].id == event.id
        assert retrieved_session.events[0].data["event_key"] == "event_value"

        # Clean up
        await service.delete_session(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
            session_id=session.id,
        )

    @pytest.mark.asyncio
    async def test_list_sessions(
        self, service: FirestoreSessionService, test_data: dict
    ):
        """Test listing sessions for a user."""
        # Create multiple sessions for the same user
        sessions = []
        num_sessions = 3

        for i in range(num_sessions):
            session = await service.create_session(
                app_name=test_data["app_name"],
                user_id=test_data["user_id"],
                state={"index": i, **test_data["state"]},
            )
            sessions.append(session)

        # List sessions
        list_response = await service.list_sessions(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
        )

        # Verify list response
        assert list_response is not None
        assert (
            len(list_response.sessions) >= num_sessions
        )  # There might be other sessions from previous tests

        # Verify our sessions are in the list
        session_ids = [s.id for s in list_response.sessions]
        for session in sessions:
            assert session.id in session_ids

        # Clean up
        for session in sessions:
            await service.delete_session(
                app_name=test_data["app_name"],
                user_id=test_data["user_id"],
                session_id=session.id,
            )

    @pytest.mark.asyncio
    async def test_multiple_events(
        self, service: FirestoreSessionService, test_data: dict
    ):
        """Test appending multiple events to a session."""
        # Create session
        session = await service.create_session(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
            state=test_data["state"],
        )

        # Append multiple events
        num_events = 10
        events = []

        for i in range(num_events):
            event = Event(
                id=f"event-{uuid.uuid4().hex[:8]}",
                session_id=session.id,
                timestamp=int(time.time() * 1000) + i,  # Ensure timestamp order
                data={"event_index": i},
            )
            appended_event = await service.append_event(session=session, event=event)
            events.append(appended_event)

        # Get the session with events
        retrieved_session = await service.get_session(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
            session_id=session.id,
        )

        # Verify events
        assert retrieved_session is not None
        assert len(retrieved_session.events) == num_events

        # Verify events are in timestamp order
        for i, event in enumerate(retrieved_session.events):
            assert event.data["event_index"] == i

        # Clean up
        await service.delete_session(
            app_name=test_data["app_name"],
            user_id=test_data["user_id"],
            session_id=session.id,
        )
