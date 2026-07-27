from cli_app import run_cli
from gui_app import run_gui
import sys


def main():
    print("Select a mode:")
    print("1. GUI")
    print("2. CLI")

    choice = input("Enter your choice: ").strip()

    if choice == "1":
        run_gui()

    elif choice == "2":

        try:
            run_cli()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(0)

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()