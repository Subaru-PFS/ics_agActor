def test_version_import():
    # Import the package version exposed by agActor
    from agActor.version import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
