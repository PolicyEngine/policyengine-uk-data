from pathlib import Path

data_folder = Path(__file__).parent.parent.parent / "data"

def main() -> None:
    """Print a hello message from the policyengine-uk-data package."""
    print("Hello from policyengine-uk-data!")
