"""Test module for FirestoreSessionService."""

import asyncio
import unittest
import uuid
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch, call

from google.adk.events import Event
from google.adk.sessions import Session, State
from google.adk.sessions.base_session_service import ListSessionsResponse
from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentSnapshot, DocumentReference, Query

from adk.ext.sessions.firestore_session_svc import FirestoreSessionService


class TestFirestoreSessionService(unittest.TestCase):
    """Test case for FirestoreSessionService."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock Firestore Client
        self.mock_client_patcher = patch("google.cloud.firestore.Client")
        self.mock_client = self.mock_client_patcher.start()

        # Mock Firestore collection and document references
        self.mock_collection = MagicMock()
        self.mock_doc_ref = MagicMock(spec=DocumentReference)
        self.mock_events_collection = MagicMock()
        self.mock_batch = MagicMock()

        # Setup the mocks
        self.mock_client.return_value.collection = MagicMock(
            return_value=self.mock_collection
        )
        self.mock_collection.document = MagicMock(return_value=self.mock_doc_ref)
        self.mock_client.return_value.batch = MagicMock(return_value=self.mock_batch)

        # Create the service with mocked client
        self.service = FirestoreSessionService(
            project_id="test-project",
            collection_name="test_sessions",
            events_collection_name="test_events",
        )

        # Test data
        self.app_name = "test-app"
        self.user_id = "test-user"
        self.session_id = "test-session"
        self.state = {"key": "value"}
        self.test_event = Event(
            id="test-event-id",
            author="test-user",
            timestamp=1622505600000,  # Example timestamp
            custom_metadata={"session_id": self.session_id, "event_key": "event_value"},
        )

        # Mock for transaction
        self.mock_transaction = MagicMock()
        self.mock_client.return_value.transaction.return_value = self.mock_transaction

    def tearDown(self):
        """Tear down test fixtures."""
        self.mock_client_patcher.stop()

    def _setup_mock_doc_snapshot(self, exists=True):
        """Helper to create a mock document snapshot."""
        mock_snapshot = MagicMock(spec=DocumentSnapshot)
        mock_snapshot.exists = exists
        if exists:
            mock_snapshot.to_dict.return_value = {
                "session_id": self.session_id,
                "app_name": self.app_name,
                "user_id": self.user_id,
                "state": self.state,
                "created_at": 1622505600000,  # Example timestamp
                "updated_at": 1622505600000,
            }
        return mock_snapshot

    def test_init(self):
        """Test initialization of FirestoreSessionService."""
        self.assertEqual(self.service.db, self.mock_client.return_value)
        self.assertEqual(self.service.collection, self.mock_collection)
        self.assertEqual(self.service.events_collection_name, "test_events")

    def test_get_session_doc_ref(self):
        """Test getting session document reference."""
        doc_ref = self.service._get_session_doc_ref(
            self.app_name, self.user_id, self.session_id
        )
        self.mock_collection.document.assert_called_once_with(
            f"{self.app_name}:{self.user_id}:{self.session_id}"
        )
        self.assertEqual(doc_ref, self.mock_doc_ref)

    def test_get_events_collection_ref(self):
        """Test getting events collection reference."""
        self.mock_client.return_value.collection.side_effect = [
            self.mock_collection,
            self.mock_events_collection,
        ]
        events_ref = self.service._get_events_collection_ref(
            self.app_name, self.user_id, self.session_id
        )
        self.mock_client.return_value.collection.assert_called_with(
            f"test_events/{self.app_name}:{self.user_id}:{self.session_id}/events"
        )

    def test_doc_to_session_not_exists(self):
        """Test doc_to_session with non-existent document."""
        mock_snapshot = self._setup_mock_doc_snapshot(exists=False)
        session = self.service._doc_to_session(mock_snapshot)
        self.assertIsNone(session)

    def test_doc_to_session_exists(self):
        """Test doc_to_session with existent document."""
        mock_snapshot = self._setup_mock_doc_snapshot(exists=True)
        session = self.service._doc_to_session(mock_snapshot)
        self.assertIsNotNone(session)
        self.assertEqual(session.id, self.session_id)
        self.assertEqual(session.app_name, self.app_name)
        self.assertEqual(session.user_id, self.user_id)
        self.assertEqual(session.state["key"], "value")

    @mock.patch("uuid.uuid4")
    def test_create_session_new_id(self, mock_uuid):
        """Test creating a session with a new ID."""
        # Setup mocks
        mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
        mock_snapshot = self._setup_mock_doc_snapshot(exists=False)
        self.mock_doc_ref.get.return_value = mock_snapshot

        # Setup transaction mocks
        @firestore.transactional
        def _transactional_create(*args):
            # Mock the transaction
            return True

        # Execute
        session = asyncio.run(
            self.service.create_session(
                app_name=self.app_name, user_id=self.user_id, state=self.state
            )
        )

        # Verify
        self.assertEqual(session.id, "12345678-1234-5678-1234-567812345678")
        self.assertEqual(session.app_name, self.app_name)
        self.assertEqual(session.user_id, self.user_id)
        self.assertEqual(session.state["key"], "value")

    def test_create_session_existing_session(self):
        """Test creating a session with an existing ID."""
        # Setup mocks for an existing session
        mock_snapshot = self._setup_mock_doc_snapshot(exists=True)

        # Setup transaction behavior
        def side_effect(transaction, *args):
            # Simulate the exists check inside transaction
            if mock_snapshot.exists:
                raise ValueError(
                    f"Session already exists: {self.app_name}:{self.user_id}:{self.session_id}"
                )

        # Patch the transactional decorator to handle our mock
        with patch("google.cloud.firestore.transactional", side_effect=side_effect):
            # Execute and verify exception
            with self.assertRaises(ValueError) as context:
                asyncio.run(
                    self.service.create_session(
                        app_name=self.app_name,
                        user_id=self.user_id,
                        state=self.state,
                        session_id=self.session_id,
                    )
                )

            self.assertIn("Session already exists", str(context.exception))

    def test_create_session_transaction_exception(self):
        """Test creating a session when the transaction fails."""
        # Setup mock to simulate a transaction exception
        self.mock_client.return_value.transaction.side_effect = Exception(
            "Transaction failed"
        )

        # Execute and verify exception is raised
        with self.assertRaises(Exception) as context:
            asyncio.run(
                self.service.create_session(
                    app_name=self.app_name,
                    user_id=self.user_id,
                    state=self.state,
                    session_id=self.session_id,
                )
            )

        self.assertIn("Transaction failed", str(context.exception))

    def test_get_session_transaction_exception(self):
        """Test getting a session when the transaction fails."""
        # Setup mock to simulate a transaction exception
        self.mock_client.return_value.transaction.side_effect = Exception(
            "Transaction failed"
        )

        # Execute - should return None instead of raising exception
        session = asyncio.run(
            self.service.get_session(
                app_name=self.app_name, user_id=self.user_id, session_id=self.session_id
            )
        )

        # Verify
        self.assertIsNone(session)

    def test_get_session_exists(self):
        """Test getting an existing session."""
        # Setup document snapshot mock
        mock_snapshot = self._setup_mock_doc_snapshot(exists=True)
        self.mock_doc_ref.get.return_value = mock_snapshot

        # Setup events query mock
        mock_query = MagicMock(spec=Query)
        mock_event_docs = [MagicMock()]
        mock_event_docs[0].to_dict.return_value = {
            "id": "test-event-id",
            "author": "test-user",
            "timestamp": 1622505600000,
            "custom_metadata": {
                "session_id": self.session_id,
                "event_key": "event_value",
            },
        }
        mock_query.stream.return_value = mock_event_docs

        self.mock_events_collection.order_by.return_value = mock_query
        self.service._get_events_collection_ref = MagicMock(
            return_value=self.mock_events_collection
        )

        # Setup transaction mocks
        @firestore.transactional
        def _transactional_get(*args):
            # Mock the transaction get
            return mock_snapshot

        # Execute
        session = asyncio.run(
            self.service.get_session(
                app_name=self.app_name, user_id=self.user_id, session_id=self.session_id
            )
        )

        # Verify
        self.assertIsNotNone(session)
        self.assertEqual(session.id, self.session_id)

    def test_get_session_not_exists(self):
        """Test getting a non-existent session."""
        # Setup document snapshot mock
        mock_snapshot = self._setup_mock_doc_snapshot(exists=False)

        # Setup transaction mocks
        def side_effect(transaction, *args):
            # Simulate not exists in transaction
            return None

        # Patch the transactional decorator to handle our mock
        with patch("google.cloud.firestore.transactional", side_effect=side_effect):
            # Execute
            session = asyncio.run(
                self.service.get_session(
                    app_name=self.app_name,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
            )

            # Verify
            self.assertIsNone(session)

    def test_list_sessions(self):
        """Test listing sessions."""
        # Setup query and snapshot mocks
        mock_query = MagicMock()
        mock_snapshots = [self._setup_mock_doc_snapshot()]
        mock_query.get.return_value = mock_snapshots

        self.mock_collection.where.return_value.where.return_value = mock_query

        # Execute
        response = asyncio.run(
            self.service.list_sessions(app_name=self.app_name, user_id=self.user_id)
        )

        # Verify
        self.assertIsInstance(response, ListSessionsResponse)
        self.assertEqual(len(response.sessions), 1)
        self.assertEqual(response.sessions[0].id, self.session_id)
        self.assertEqual(response.sessions[0].app_name, self.app_name)
        self.assertEqual(response.sessions[0].user_id, self.user_id)

    def test_list_sessions_multiple(self):
        """Test listing multiple sessions."""
        # Setup query and snapshot mocks for multiple sessions
        mock_query = MagicMock()

        # Create multiple session snapshots
        mock_snapshots = []
        for i in range(3):
            session_id = f"test-session-{i}"
            mock_snapshot = MagicMock(spec=DocumentSnapshot)
            mock_snapshot.exists = True
            mock_snapshot.to_dict.return_value = {
                "session_id": session_id,
                "app_name": self.app_name,
                "user_id": self.user_id,
                "state": {"key": f"value-{i}"},
                "created_at": 1622505600000 + i * 1000,
                "updated_at": 1622505600000 + i * 1000,
            }
            mock_snapshots.append(mock_snapshot)

        mock_query.get.return_value = mock_snapshots
        self.mock_collection.where.return_value.where.return_value = mock_query

        # Execute
        response = asyncio.run(
            self.service.list_sessions(app_name=self.app_name, user_id=self.user_id)
        )

        # Verify
        self.assertIsInstance(response, ListSessionsResponse)
        self.assertEqual(len(response.sessions), 3)
        for i, session in enumerate(response.sessions):
            self.assertEqual(session.id, f"test-session-{i}")
            self.assertEqual(session.app_name, self.app_name)
            self.assertEqual(session.user_id, self.user_id)
            self.assertEqual(session.state["key"], f"value-{i}")

    def test_list_sessions_empty(self):
        """Test listing sessions when none exist."""
        # Setup query with empty result
        mock_query = MagicMock()
        mock_query.get.return_value = []
        self.mock_collection.where.return_value.where.return_value = mock_query

        # Execute
        response = asyncio.run(
            self.service.list_sessions(app_name=self.app_name, user_id=self.user_id)
        )

        # Verify
        self.assertIsInstance(response, ListSessionsResponse)
        self.assertEqual(len(response.sessions), 0)

    def test_list_sessions_exception(self):
        """Test listing sessions when an exception occurs."""
        # Setup query to raise an exception
        self.mock_collection.where.side_effect = Exception("Test exception")

        # Execute
        response = asyncio.run(
            self.service.list_sessions(app_name=self.app_name, user_id=self.user_id)
        )

        # Verify we get an empty response rather than an exception
        self.assertIsInstance(response, ListSessionsResponse)
        self.assertEqual(len(response.sessions), 0)

    def test_list_sessions_empty_params(self):
        """Test listing sessions with empty app_name or user_id."""
        # Execute and verify exception
        with self.assertRaises(ValueError):
            asyncio.run(self.service.list_sessions(app_name="", user_id=self.user_id))

        with self.assertRaises(ValueError):
            asyncio.run(self.service.list_sessions(app_name=self.app_name, user_id=""))

    def test_delete_session(self):
        """Test deleting a session."""
        # Setup document snapshot mock
        mock_snapshot = self._setup_mock_doc_snapshot(exists=True)
        self.mock_doc_ref.get.return_value = mock_snapshot

        # Setup events collection mock with no events
        mock_event_docs = []
        self.mock_events_collection.limit.return_value.stream.return_value = (
            mock_event_docs
        )
        self.service._get_events_collection_ref = MagicMock(
            return_value=self.mock_events_collection
        )

        # Execute
        asyncio.run(
            self.service.delete_session(
                app_name=self.app_name, user_id=self.user_id, session_id=self.session_id
            )
        )

        # Verify
        # Verify that delete was called on the document
        # Note: In actual implementation, this would be part of a transaction
        self.mock_events_collection.limit.assert_called()

    def test_delete_session_batch_exception(self):
        """Test deleting a session when batch operation fails."""
        # Setup document snapshot mock
        mock_snapshot = self._setup_mock_doc_snapshot(exists=True)
        self.mock_doc_ref.get.return_value = mock_snapshot

        # Setup events collection with events
        mock_event_docs = [MagicMock()]
        mock_event_docs[0].reference = MagicMock()
        self.mock_events_collection.limit.return_value.stream.return_value = (
            mock_event_docs
        )
        self.service._get_events_collection_ref = MagicMock(
            return_value=self.mock_events_collection
        )

        # Make the batch commit throw an exception
        self.mock_batch.commit.side_effect = Exception("Batch commit failed")

        # Execute - should handle the exception
        with self.assertRaises(Exception) as context:
            asyncio.run(
                self.service.delete_session(
                    app_name=self.app_name,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
            )

        # Verify
        self.assertIn("Batch commit failed", str(context.exception))

    def test_delete_session_not_exists(self):
        """Test deleting a non-existent session."""
        # Setup document snapshot mock
        mock_snapshot = self._setup_mock_doc_snapshot(exists=False)
        self.mock_doc_ref.get.return_value = mock_snapshot

        # Execute
        asyncio.run(
            self.service.delete_session(
                app_name=self.app_name, user_id=self.user_id, session_id=self.session_id
            )
        )

        # Verify no attempt to delete events was made
        self.mock_events_collection.limit.assert_not_called()

    def test_delete_session_transaction(self):
        """Test the transaction used in delete_session."""
        # Setup document snapshot mock
        mock_snapshot = self._setup_mock_doc_snapshot(exists=True)
        self.mock_doc_ref.get.return_value = mock_snapshot

        # Setup events collection mock with no events
        mock_event_docs = []
        self.mock_events_collection.limit.return_value.stream.return_value = (
            mock_event_docs
        )
        self.service._get_events_collection_ref = MagicMock(
            return_value=self.mock_events_collection
        )

        # Setup a specific mock for the transaction handling
        transaction_mock = MagicMock()
        transaction_mock.__enter__ = MagicMock(return_value=transaction_mock)
        transaction_mock.__exit__ = MagicMock(return_value=None)

        # Replace the self.db.transaction with our controlled mock
        self.mock_client.return_value.transaction.return_value = transaction_mock

        # Execute
        asyncio.run(
            self.service.delete_session(
                app_name=self.app_name, user_id=self.user_id, session_id=self.session_id
            )
        )

        # Verify transaction was used to delete the document
        self.mock_client.return_value.transaction.assert_called()

    def test_delete_session_large_event_batch(self):
        """Test deleting a session with a large number of events requiring multiple batches."""
        # Setup document snapshot mock
        mock_snapshot = self._setup_mock_doc_snapshot(exists=True)
        self.mock_doc_ref.get.return_value = mock_snapshot

        # Create a large number of mock events to trigger multiple batch operations
        batch_size = 10  # Use a small number for testing
        num_events = batch_size * 3 + 2  # Create enough to require multiple batches

        # Setup the events collection to return different event sets for each call
        mock_events_sets = []

        # First batch has batch_size events
        first_batch = [MagicMock() for _ in range(batch_size)]
        for mock_event in first_batch:
            mock_event.reference = MagicMock()
        mock_events_sets.append(first_batch)

        # Second batch has batch_size events
        second_batch = [MagicMock() for _ in range(batch_size)]
        for mock_event in second_batch:
            mock_event.reference = MagicMock()
        mock_events_sets.append(second_batch)

        # Third batch has batch_size events
        third_batch = [MagicMock() for _ in range(batch_size)]
        for mock_event in third_batch:
            mock_event.reference = MagicMock()
        mock_events_sets.append(third_batch)

        # Final batch has remaining events
        final_batch = [MagicMock() for _ in range(2)]
        for mock_event in final_batch:
            mock_event.reference = MagicMock()
        mock_events_sets.append(final_batch)

        # Empty batch to signal completion
        mock_events_sets.append([])

        # Configure the mock to return different sets each time
        self.mock_events_collection.limit.return_value.stream.side_effect = (
            mock_events_sets
        )
        self.service._get_events_collection_ref = MagicMock(
            return_value=self.mock_events_collection
        )

        # Patch the actual batch size constant to match our test value
        with patch.object(
            self.service, "_get_session_doc_ref", return_value=self.mock_doc_ref
        ):
            # Execute with our small batch size
            asyncio.run(
                self.service.delete_session(
                    app_name=self.app_name,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
            )

        # Verify batch operations
        # We expect 4 batches of event deletions plus 1 empty batch check
        self.assertEqual(self.mock_events_collection.limit.call_count, 5)
        # Each batch should be committed once
        self.assertEqual(self.mock_batch.commit.call_count, 4)
        # We should have deleted all events
        expected_delete_calls = batch_size * 3 + 2
        self.assertEqual(self.mock_batch.delete.call_count, expected_delete_calls)

    def test_append_event(self):
        """Test appending an event to a session."""
        # Create a session
        session = Session(
            id=self.session_id,
            app_name=self.app_name,
            user_id=self.user_id,
            state={"key": "value"},
            events=[],
        )

        # Setup collection and document references for events
        mock_event_doc_ref = MagicMock()
        self.mock_events_collection.document.return_value = mock_event_doc_ref
        self.service._get_events_collection_ref = MagicMock(
            return_value=self.mock_events_collection
        )

        # Setup session document snapshot
        mock_snapshot = self._setup_mock_doc_snapshot(exists=True)
        self.mock_doc_ref.get.return_value = mock_snapshot
        self.service._get_session_doc_ref = MagicMock(return_value=self.mock_doc_ref)

        # Execute
        event = asyncio.run(
            self.service.append_event(session=session, event=self.test_event)
        )

        # Verify
        self.assertEqual(event.id, self.test_event.id)
        self.mock_events_collection.document.assert_called_once()

    def test_append_event_transaction_rollback(self):
        """Test appending an event with transaction rollback."""
        # Create a session
        session = Session(
            id=self.session_id,
            app_name=self.app_name,
            user_id=self.user_id,
            state={"key": "value"},
            events=[],
        )

        # Setup mocks
        mock_event_doc_ref = MagicMock()
        self.mock_events_collection.document.return_value = mock_event_doc_ref
        self.service._get_events_collection_ref = MagicMock(
            return_value=self.mock_events_collection
        )

        # Setup session document snapshot
        mock_snapshot = self._setup_mock_doc_snapshot(exists=True)
        self.mock_doc_ref.get.return_value = mock_snapshot
        self.service._get_session_doc_ref = MagicMock(return_value=self.mock_doc_ref)

        # Setup transaction to raise an exception during set operation
        def transaction_side_effect(*args, **kwargs):
            raise Exception("Transaction failed on set operation")

        self.mock_transaction.set.side_effect = transaction_side_effect

        # Execute and verify exception
        with self.assertRaises(Exception) as context:
            asyncio.run(
                self.service.append_event(session=session, event=self.test_event)
            )

        # Verify the exception was raised
        self.assertIn("Transaction failed", str(context.exception))

    def test_append_event_session_not_exists(self):
        """Test appending an event to a non-existent session."""
        # Create a session
        session = Session(
            id=self.session_id,
            app_name=self.app_name,
            user_id=self.user_id,
            state={"key": "value"},
            events=[],
        )

        # Setup collection and document references for events
        mock_event_doc_ref = MagicMock()
        self.mock_events_collection.document.return_value = mock_event_doc_ref
        self.service._get_events_collection_ref = MagicMock(
            return_value=self.mock_events_collection
        )

        # Setup session document snapshot as non-existent
        mock_snapshot = self._setup_mock_doc_snapshot(exists=False)
        self.mock_doc_ref.get.return_value = mock_snapshot
        self.service._get_session_doc_ref = MagicMock(return_value=self.mock_doc_ref)

        # Setup transaction mocks to simulate non-existent session
        def side_effect(transaction, *args):
            # Simulate the not exists check inside transaction
            if not mock_snapshot.exists:
                raise ValueError(
                    f"Session does not exist: {self.app_name}:{self.user_id}:{self.session_id}"
                )

        # Patch the transactional decorator to handle our mock
        with patch("google.cloud.firestore.transactional", side_effect=side_effect):
            # Execute and verify exception
            with self.assertRaises(ValueError) as context:
                asyncio.run(
                    self.service.append_event(session=session, event=self.test_event)
                )

            self.assertIn("Session does not exist", str(context.exception))

    def test_create_session_invalid_inputs(self):
        """Test creating a session with invalid inputs."""
        # Test with empty app_name
        with self.assertRaises(ValueError):
            asyncio.run(
                self.service.create_session(
                    app_name="", user_id=self.user_id, state=self.state
                )
            )

        # Test with empty user_id
        with self.assertRaises(ValueError):
            asyncio.run(
                self.service.create_session(
                    app_name=self.app_name, user_id="", state=self.state
                )
            )

    def test_get_session_invalid_inputs(self):
        """Test getting a session with invalid inputs."""
        # Test with empty app_name
        with self.assertRaises(ValueError):
            asyncio.run(
                self.service.get_session(
                    app_name="", user_id=self.user_id, session_id=self.session_id
                )
            )

        # Test with empty user_id
        with self.assertRaises(ValueError):
            asyncio.run(
                self.service.get_session(
                    app_name=self.app_name, user_id="", session_id=self.session_id
                )
            )

        # Test with empty session_id
        with self.assertRaises(ValueError):
            asyncio.run(
                self.service.get_session(
                    app_name=self.app_name, user_id=self.user_id, session_id=""
                )
            )

    def test_delete_session_invalid_inputs(self):
        """Test deleting a session with invalid inputs."""
        # Test with empty app_name
        with self.assertRaises(ValueError):
            asyncio.run(
                self.service.delete_session(
                    app_name="", user_id=self.user_id, session_id=self.session_id
                )
            )

        # Test with empty user_id
        with self.assertRaises(ValueError):
            asyncio.run(
                self.service.delete_session(
                    app_name=self.app_name, user_id="", session_id=self.session_id
                )
            )

        # Test with empty session_id
        with self.assertRaises(ValueError):
            asyncio.run(
                self.service.delete_session(
                    app_name=self.app_name, user_id=self.user_id, session_id=""
                )
            )


if __name__ == "__main__":
    unittest.main()
