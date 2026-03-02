# Project Structure Overview

 High-level explanation of the project's directory layout and key components.

```
open-source-resume-screening/          # Root of workspace
├── pyproject.toml                    # Project metadata and build configuration
├── requirements.txt                  # Python dependencies
├── osrs/                             # Virtual environment for Python
│   ├── Include/                       # Python include files
│   ├── Lib/                           # Installed packages
│   ├── Scripts/                       # Activation scripts and utilities
│   └── share/                         # Shared resources
├── scripts/                          # Custom utility scripts (if any)
├── src/                              # Source code for the application
│   ├── __init__.py                   # Package marker
│   ├── main.py                       # Entry point for CLI or service startup
│   ├── agents/                       # Autonomous agents handling 
│   │   ├── __init__.py
│   │   ├── base.py                   # Core agent abstractions
│   │   ├── data_fetching_agent.py    # Retrieves data from external sources
│   │   ├── explanation_agent.py      # Generates explanations/
│   │   ├── fairness_agent.py         # Conducts fairness analyses
│   │   ├── orchestrator.py           # Coordinates multiple agents
│   │   ├── ranking_agent.py          # Handles ranking logic
│   │   ├── screening_agent.py        # Applies screening logic
│   │   └── validation_agent.py       # Ensures inputs meet criteria
│   ├── api/                          # Web API module
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI or similar application setup
│   │   ├── routes/                   # HTTP route definitions
│   │   └── templates/                # HTML templates if used
│   ├── config/                       # Configuration modules
│   │   ├── __init__.py
│   │   ├── logging_config.py         # Logging setup
│   │   └── settings.py               # Application settings and constants
│   ├── metrics/                      # Tracking and fairness metrics
│   │   ├── __init__.py
│   │   ├── fairness.py
│   │   └── tracking.py
│   ├── models/                       # Data models and schemas
│   │   ├── __init__.py
│   │   ├── schemas.py                # Pydantic or dataclass definitions
│   │   └── scoring.py                # Scoring logic and evaluation models
│   ├── services/                     # Business logic services
│   │   ├── __init__.py
│   │   ├── document_parser.py        # Parses resumes/documents
│   │   ├── embedding_service.py      # Embedding generation
│   │   ├── llm_service.py            # LLM interaction wrapper
│   │   ├── mongo_service.py          # MongoDB abstraction
│   │   ├── screening_service.py      # Candidate screening logic
│   │   └── talentmatch_client.py     # External API client
│   └── utils/                        # Utility helpers
│       ├── __init__.py
│       ├── batch_processing.py       # Batch job utilities
│       └── text_processing.py        # Text parsing and cleaning
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest configuration
│   ├── integration/                  # Integration tests
│   │   └── __init__.py
│   └── unit/                         # Unit tests
│       ├── __init__.py
│       └── test_ranking_agent.py     # Example unit test
```  