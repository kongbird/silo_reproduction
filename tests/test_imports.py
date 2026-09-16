def test_basic_imports():
    import torch
    import numpy
    import gymnasium

    assert torch.__version__ is not None
    assert numpy.__version__ is not None
    assert gymnasium.__version__ is not None
