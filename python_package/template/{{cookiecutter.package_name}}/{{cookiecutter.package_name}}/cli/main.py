"""Main CLI module for {{ cookiecutter.package_name }}."""

import typer
from {{ cookiecutter.package_name }} import __version__
from {{ cookiecutter.package_name }}.utils import get_logger

app = typer.Typer()

@app.command()
def main():
    """Main entrypoint for the {{ cookiecutter.package_name }} CLI."""
    logger = get_logger()
    logger.info("Starting {{ cookiecutter.package_name }} CLI")
    typer.echo("Hello, I am the {{ cookiecutter.package_name }} CLI!")
    logger.info("{{ cookiecutter.package_name }} CLI executed successfully")


@app.command()
def version():
    """Show the version of {{ cookiecutter.package_name }}."""
    typer.echo(f"{{ cookiecutter.package_name }} v{__version__}")


if __name__ == "__main__":
    app()