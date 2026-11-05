"""Download one complete folder from the SSH server into this project."""

import os
import stat
from pathlib import Path, PurePosixPath

import paramiko
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


FOLDER_NAME = "20261105_141925_266808_e100_lr1_human_studio"



REMOTE_FOLDER = f"~/3d_project/pipeline/outputs/{FOLDER_NAME}/snapshots"
LOCAL_DOWNLOADS = PROJECT_ROOT / f"downloads/{FOLDER_NAME}"


load_dotenv(PROJECT_ROOT / ".env")


def download_tree(sftp, remote_dir, local_dir):
    """Copy a remote directory and everything inside it."""
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = str(PurePosixPath(remote_dir) / entry.filename)
        local_path = local_dir / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            download_tree(sftp, remote_path, local_path)
        else:
            sftp.get(remote_path, str(local_path))
            print(f"Downloaded {remote_path}")


def main():
    host = os.getenv("SSH_HOST")
    username = os.getenv("SSH_USERNAME")
    password = os.getenv("SSH_PASSWORD")
    port = int(os.getenv("SSH_PORT", "22"))

    if not all((host, username, password)):
        raise SystemExit("Set SSH_HOST, SSH_USERNAME, and SSH_PASSWORD in the project-root .env file.")

    local_dir = LOCAL_DOWNLOADS / Path(REMOTE_FOLDER.rstrip("/")).name

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())

    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            look_for_keys=False,
            allow_agent=False,
            timeout=20,
        )
        with client.open_sftp() as sftp:
            remote_home = sftp.normalize(".")
            if REMOTE_FOLDER == "~":
                remote_dir = remote_home
            elif REMOTE_FOLDER.startswith("~/"):
                remote_dir = str(PurePosixPath(remote_home) / REMOTE_FOLDER[2:])
            else:
                remote_dir = REMOTE_FOLDER

            if not stat.S_ISDIR(sftp.stat(remote_dir).st_mode):
                raise SystemExit(f"Remote path is not a directory: {remote_dir}")

            download_tree(sftp, remote_dir, local_dir)
    finally:
        client.close()

    print(f"Download complete: {local_dir}")


if __name__ == "__main__":
    main()
