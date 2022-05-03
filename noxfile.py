import tempfile
from typing import Any

import nox
from nox.sessions import Session

nox.options.sessions = "black", "flake8", "mypy", "tests"
locations = "src", "noxfile.py"


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


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


# @nox.session(python="3.9")
# def tests(session: Session) -> None:
#     """Run the test suite."""
#     args = session.posargs
#     session.run("poetry", "install", "--no-dev", external=True)
#     install_with_constraints(session, "pytest")
#     session.run("pytest", *args)
