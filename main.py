from dotenv import load_dotenv

load_dotenv()

from register import register_user
from recognize import run_recognition
from database_manager import init_db

if __name__ == "__main__":
    init_db()
    while True:
        cmd = input("1. Register\n2. Recognize\n3. Exit\nChoice: ").strip()
        if cmd == "1":
            name = input("Name: ").strip()
            if not name:
                print("Name cannot be empty.")
                continue
            register_user(name)
        elif cmd == "2":
            run_recognition()
        elif cmd == "3":
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
