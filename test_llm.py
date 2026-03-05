"""Quick test of LLM status and model selection."""

from src.services.llm_service import LLMService

def test_llm():
    """Test LLM initialization and model selection."""
    print("Testing LLM Service...")
    llm = LLMService.get_instance()

    print(f"Initial model: {llm.model_name}")

    try:
        print("Initializing LLM...")
        llm.initialize()
        print("✓ Initialization successful")

        info = llm.get_model_info()
        print(f"Status: {info.get('status')}")
        print(f"Model: {info.get('model')}")
        print(f"Available models: {info.get('available_models', [])}")

        if info.get('status') == 'ready':
            print("✓ LLM is ready for use")
        else:
            print("✗ LLM is not ready")

    except Exception as e:
        print(f"✗ LLM initialization failed: {e}")

if __name__ == "__main__":
    test_llm()