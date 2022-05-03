import os
import tempfile
from typing import Any

import nox
from nox.sessions import Session

nox.options.sessions = "black", "flake8", "mypy"
locations = "src", "noxfile.py"


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    req_path = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
    session.run(
        "poetry",
        "export",
        "--dev",
        "--format=requirements.txt",
        "--without-hashes",
        f"--output={req_path}",
        external=True,
    )
    session.install(f"--constraint={req_path}", *args, **kwargs)
    os.unlink(req_path)


@nox.session(python="3.9")
def black(session: Session) -> None:
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python="3.9")
def flake8(session: Session) -> None:
    args = session.posargs or locations
    install_with_constraints(session, "flake8")
    session.run("flake8", *args)


@nox.session(python="3.9")
def mypy(session: Session) -> None:
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)
