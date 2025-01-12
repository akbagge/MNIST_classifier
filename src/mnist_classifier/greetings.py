import typer

# Create a Typer app
app = typer.Typer()

@app.command()
def hello(count: int = 1, name: str = "World"):
    """
    Print a greeting message.

    Parameters:
    ----------
    count : int
        The number of times to print the message (default is 1).
    name : str
        The name to include in the greeting (default is "World").
    """
    for _ in range(count):
        typer.echo(f"Hello {name}!")

if __name__ == "__main__":
    app()
