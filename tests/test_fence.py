"""The library's test session is fenced from credentials and service managers."""
import os
import subprocess

import pytest

from fence import CREDENTIAL, LiveAccess


def test_no_credential_and_no_service_manager():
    assert not [n for n in os.environ if CREDENTIAL.search(n)]
    with pytest.raises(LiveAccess):
        subprocess.run(["launchctl", "list"])
