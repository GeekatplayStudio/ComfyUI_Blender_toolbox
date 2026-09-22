"""Credential management backed by the operating-system keyring."""

import base64
import json
import os
import re
from itertools import cycle

import keyring
from aiohttp import web
from keyring.errors import KeyringError
from server import PromptServer


KEYRING_SERVICE = "ComfyUI-Blender-Toolbox"
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
KEY_INDEX_PATH = os.path.join(ROOT_DIR, "geekatplay_keystore.json")
LEGACY_STORE_PATH = os.path.join(ROOT_DIR, "geekatplay_keystore.enc")
LEGACY_OBFUSCATION_KEY = "GeekatplayStudio_Secure_Key_Salt_2026"
MAX_NAME_LENGTH = 80
MAX_SECRET_LENGTH = 8192


def _validate_name(name):
    normalized = str(name or "").strip()
    if not normalized or len(normalized) > MAX_NAME_LENGTH:
        raise ValueError(f"Credential name must contain 1-{MAX_NAME_LENGTH} characters")
    if not re.fullmatch(r"[A-Za-z0-9_. @+-]+", normalized):
        raise ValueError("Credential name contains unsupported characters")
    return normalized


def _read_index():
    try:
        with open(KEY_INDEX_PATH, "r", encoding="utf-8") as index:
            data = json.load(index)
        return sorted({_validate_name(item) for item in data.get("keys", [])})
    except FileNotFoundError:
        return []
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[Geekatplay KeyManager] Could not read credential index: {exc}")
        return []


def _write_index(names):
    temporary_path = KEY_INDEX_PATH + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as index:
        json.dump({"version": 2, "keys": sorted(set(names))}, index, indent=2)
    os.replace(temporary_path, KEY_INDEX_PATH)


def _decode_legacy_store(encoded):
    decoded = base64.b64decode(encoded).decode("utf-8")
    return "".join(
        chr(ord(value) ^ ord(key))
        for value, key in zip(decoded, cycle(LEGACY_OBFUSCATION_KEY))
    )


def migrate_legacy_store():
    """Move legacy XOR-obfuscated credentials into the OS keyring once."""
    if not os.path.exists(LEGACY_STORE_PATH):
        return
    try:
        with open(LEGACY_STORE_PATH, "r", encoding="utf-8") as legacy:
            keys = json.loads(_decode_legacy_store(legacy.read()))
        for name, value in keys.items():
            save_key(name, value)
        os.replace(LEGACY_STORE_PATH, LEGACY_STORE_PATH + ".migrated")
        print("[Geekatplay KeyManager] Migrated legacy credentials to the OS keyring.")
    except Exception as exc:
        print(f"[Geekatplay KeyManager] Legacy credential migration failed: {exc}")


def list_keys():
    migrate_legacy_store()
    return _read_index()


def load_keys():
    """Compatibility helper. Secrets are fetched from the keyring on demand."""
    result = {}
    for name in list_keys():
        try:
            secret = keyring.get_password(KEYRING_SERVICE, name)
        except KeyringError as exc:
            raise RuntimeError(f"OS credential vault is unavailable: {exc}") from exc
        if secret is not None:
            result[name] = secret
    return result


def save_key(name, value):
    name = _validate_name(name)
    secret = str(value or "").strip()
    if not secret or len(secret) > MAX_SECRET_LENGTH:
        raise ValueError(f"Credential value must contain 1-{MAX_SECRET_LENGTH} characters")
    try:
        keyring.set_password(KEYRING_SERVICE, name, secret)
    except KeyringError as exc:
        raise RuntimeError(f"Could not save credential in the OS vault: {exc}") from exc
    names = _read_index()
    if name not in names:
        names.append(name)
        _write_index(names)


def delete_key(name):
    name = _validate_name(name)
    try:
        keyring.delete_password(KEYRING_SERVICE, name)
    except keyring.errors.PasswordDeleteError:
        pass
    except KeyringError as exc:
        raise RuntimeError(f"Could not delete credential from the OS vault: {exc}") from exc
    _write_index(item for item in _read_index() if item != name)


def get_key(name):
    """Read a secret by credential name.

    The lookup is case-insensitive: a key saved as "Tripo3d" is found by "Tripo3D" or "tripo3d",
    so nodes resolve credentials regardless of how the name was capitalised when it was saved.
    """
    if not name or name == "None":
        return ""
    name = _validate_name(name)
    try:
        secret = keyring.get_password(KEYRING_SERVICE, name)
        if secret:
            return secret
        wanted = name.casefold()
        for stored in _read_index():
            if stored != name and stored.casefold() == wanted:
                secret = keyring.get_password(KEYRING_SERVICE, stored)
                if secret:
                    return secret
        return ""
    except KeyringError as exc:
        raise RuntimeError(f"Could not read credential from the OS vault: {exc}") from exc


if hasattr(PromptServer, "instance"):
    routes = PromptServer.instance.routes

    @routes.post("/geekatplay/credentials")
    async def save_key_endpoint(request):
        try:
            payload = await request.json()
            save_key(payload.get("name"), payload.get("value"))
            return web.json_response({"status": "success", "keys": list_keys()})
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"status": "error", "message": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"status": "error", "message": str(exc)}, status=500)

    @routes.delete("/geekatplay/credentials/{name}")
    async def delete_key_endpoint(request):
        try:
            delete_key(request.match_info["name"])
            return web.json_response({"status": "success", "keys": list_keys()})
        except ValueError as exc:
            return web.json_response({"status": "error", "message": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"status": "error", "message": str(exc)}, status=500)

    @routes.get("/geekatplay/credentials")
    async def list_keys_endpoint(request):
        return web.json_response({"keys": list_keys(), "storage": "os-keyring"})

    # Backward-compatible endpoints for older cached frontend files.
    @routes.post("/geekatplay/save_key")
    async def legacy_save_key_endpoint(request):
        return await save_key_endpoint(request)

    @routes.post("/geekatplay/delete_key")
    async def legacy_delete_key_endpoint(request):
        try:
            payload = await request.json()
            delete_key(payload.get("name"))
            return web.json_response({"status": "success", "keys": list_keys()})
        except Exception as exc:
            return web.json_response({"status": "error", "message": str(exc)}, status=400)

    @routes.get("/geekatplay/list_keys")
    async def legacy_list_keys_endpoint(request):
        return await list_keys_endpoint(request)


class Geekatplay_ApiKey_Manager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "service_name": (["None", *list_keys()],),
                "mode": (
                    ["Read/Select", "SAVE New Key", "DELETE Selected"],
                    {"default": "Read/Select"},
                ),
            },
            "optional": {
                "new_key_name": ("STRING", {"default": "Meshy"}),
                "new_key_value": (
                    "STRING",
                    {"default": "", "multiline": False, "password": True},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "manage_keys"
    CATEGORY = "Geekatplay/Authentication"
    OUTPUT_NODE = True

    def manage_keys(self, service_name, mode, new_key_name="Meshy", new_key_value=""):
        if mode == "SAVE New Key":
            save_key(new_key_name, new_key_value)
            return (new_key_value.strip(),)
        if mode == "DELETE Selected":
            if service_name != "None":
                delete_key(service_name)
            return ("",)
        direct_value = new_key_value.strip()
        return (get_key(service_name) or direct_value,)

    @classmethod
    def IS_CHANGED(cls, service_name, mode, new_key_name="", new_key_value=""):
        if mode != "Read/Select":
            return float("nan")
        # Never include secret material in ComfyUI's change fingerprint.
        return service_name


NODE_CLASS_MAPPINGS = {"Geekatplay_ApiKey_Manager": Geekatplay_ApiKey_Manager}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Geekatplay_ApiKey_Manager": "Credential Manager (OS Keyring)"
}
