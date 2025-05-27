# adk-ext

[![Python Unit Tests](https://github.com/nandlabs/adk-ext-python/actions/workflows/unittest.yml/badge.svg)](https://github.com/nandlabs/adk-ext-python/actions/workflows/unittest.yml)
[![Python Versions](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.0.1-green)](https://github.com/nandlabs/adk-ext-python)

A comprehensive Python extension library for Google's Agent Development Kit (ADK).

## Overview

adk-ext provides a set of tools, utilities, and wrapper functions that extend the functionality of the standard ADK for Python developers. This library aims to simplify common ADK operations, improve development workflow, and offer additional features not available in the core ADK implementation.

## Features

- **Session Service**:  
   Session Service Implementations including:
  - Firestore: Persistent storage of sessions and events in Google Cloud Firestore
  - Redis (coming soon): High-performance in-memory session storage
  - MongoDB (coming soon): Document-oriented session storage

## Installation

```bash
pip install adk-ext
```

## Quick Start

### Using FirestoreSessionService

```python
from adk.ext.sessions.firestore_session_svc import FirestoreSessionService
from google.adk.runners import Runner
from google.adk.agents import LlmAgent

# Initialize the session service
session_service = FirestoreSessionService(project_id="your-gcp-project-id")

# Create Runner and execute the request
....
```

### Prerequisites for Firestore

- Set up a Google Cloud project with Firestore enabled
- Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your service account key file
- Install dependencies: `pip install google-cloud-firestore`

## Testing

### Unit Tests

To run the unit tests:

```bash
pytest -xvs test/unittests/
```

### Integration Tests

To run the integration tests for Firestore:

1. Set up your Google Cloud credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
```

2. Specify the Google Cloud project ID:

```bash
export FIRESTORE_PROJECT_ID=your-test-project-id
```

3. Run the integration tests:

```bash
pytest -xvs test/integration/
```

Note: Integration tests will create temporary collections in your Firestore database with random prefixes to avoid conflicts.

## Documentation

For detailed documentation, examples and API reference, please visit the [documentation site](https://adk-ext.readthedocs.io).

## Requirements

- Python 3.8+
- ADK core library

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Releasing

This project uses GitHub Actions to automatically publish to PyPI when a new release is tagged with a semantic version number.

1. Update the version in `pyproject.toml`
2. Commit and push the changes
3. Create and push a new tag with a semantic version format:

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

4. The GitHub Action will automatically build and publish the package to PyPI

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- ADK core developers
- Contributors to this extension library
- All users providing valuable feedback and feature requests
