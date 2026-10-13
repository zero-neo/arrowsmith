import arrowsmith


def test_version_is_string():
    assert isinstance(arrowsmith.__version__, str)
