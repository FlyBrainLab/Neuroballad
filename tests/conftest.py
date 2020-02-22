import numpy as np
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--mode", action="store", default="w",
        help="mode for testing opening documents: r, r+, w, w-, x, a"
    )


@pytest.fixture
def mode(request):
    return request.config.getoption("--mode")
