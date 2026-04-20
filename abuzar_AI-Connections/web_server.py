"""Small Python web server for the multi-agent puzzle MVP."""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agents.claude_client import ClaudeError
from agents.group_agents import GroupGenerationFactory, generate_fresh_puzzle_batch
from agents.group_bank import GROUP_BANK_STRATEGIES, add_groups_to_bank, assemble_puzzle_batch, load_group_bank
from agents.nyt_guard import blocklist_status
from agents.puzzle_agents import MultiAgentPuzzleFactory
from agents.puzzle_store import add_puzzles, load_latest_run, load_puzzles, save_latest_run


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


class PuzzleRequestHandler(BaseHTTPRequestHandler):
    server_version = "ConnectionForge/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/puzzles":
            puzzles = load_puzzles()
            self.send_json({"count": len(puzzles), "group_count": len(load_group_bank())})
            return

        if parsed.path == "/api/puzzles/random":
            puzzles = load_puzzles()
            if not puzzles:
                self.send_json({"error": "No puzzles found in data/puzzles.json."}, HTTPStatus.NOT_FOUND)
                return
            self.send_json({"puzzle": random.choice(puzzles)})
            return

        if parsed.path == "/api/agent-runs/latest":
            self.send_json(load_latest_run())
            return

        if parsed.path == "/api/nyt-guard":
            self.send_json(blocklist_status())
            return

        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/generate-groups":
            self.handle_generate_groups()
            return

        if parsed.path != "/api/generate":
            self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return

        factory = None
        try:
            body = self.read_json_body()
            count = max(1, min(int(body.get("count", 1)), 12))
            difficulty = str(body.get("difficulty", "mixed"))
            theme = str(body.get("theme", ""))
            strategy = str(body.get("strategy", "standard"))
            model = str(body.get("model", "")).strip() or None
            generator_model = str(body.get("generator_model", "")).strip() or None
            reviewer_model = str(body.get("reviewer_model", "")).strip() or None
            max_review_rounds = max(1, min(int(body.get("max_review_rounds", 2)), 3))
            save = bool(body.get("save", True))

            if strategy == "fresh-puzzle":
                result = generate_fresh_puzzle_batch(
                    target_count=count,
                    existing_groups=load_group_bank(),
                    existing_puzzles=load_puzzles(),
                    theme=theme,
                    difficulty=difficulty,
                    model=model,
                    generator_model=generator_model,
                    reviewer_model=reviewer_model,
                )
            elif strategy in GROUP_BANK_STRATEGIES:
                result = assemble_puzzle_batch(
                    target_count=count,
                    existing_puzzles=load_puzzles(),
                    difficulty=difficulty,
                )
            else:
                factory = MultiAgentPuzzleFactory(
                    model=model,
                    generator_model=generator_model,
                    reviewer_model=reviewer_model,
                    existing_puzzles=load_puzzles(),
                )
                result = factory.generate_batch(
                    target_count=count,
                    difficulty=difficulty,
                    theme=theme,
                    strategy=strategy,
                    max_review_rounds=max_review_rounds,
                )
            saved_payload = {"accepted": [], "rejected": [], "total": len(load_puzzles())}
            group_saved_payload = {"accepted": [], "rejected": [], "total": len(load_group_bank())}

            if save and result["accepted"]:
                saved_payload = add_puzzles(result["accepted"])
                if saved_payload["accepted"] and result.get("generated_groups"):
                    group_saved_payload = add_groups_to_bank(result["generated_groups"])

            response = {
                "accepted": result["accepted"],
                "rejected": (
                    result["rejected"]
                    + saved_payload.get("rejected", [])
                    + group_saved_payload.get("rejected", [])
                ),
                "trace": result["trace"],
                "saved": save,
                "bank_total": saved_payload.get("total", len(load_puzzles())),
                "group_bank_total": group_saved_payload.get("total", len(load_group_bank())),
                "generated_groups": group_saved_payload.get("accepted", []),
            }
            save_latest_run(response)
            self.send_json(response)
        except ClaudeError as error:
            if factory is not None:
                save_latest_run(
                    {
                        "accepted": [],
                        "rejected": [{"errors": [str(error)], "stage": "claude_error"}],
                        "trace": [event.to_dict() for event in factory.trace],
                        "saved": False,
                        "bank_total": len(load_puzzles()),
                    }
                )
            self.send_json({"error": str(error)}, HTTPStatus.BAD_GATEWAY)
        except Exception as error:
            if factory is not None:
                save_latest_run(
                    {
                        "accepted": [],
                        "rejected": [{"errors": [str(error)], "stage": "generation_error"}],
                        "trace": [event.to_dict() for event in factory.trace],
                        "saved": False,
                        "bank_total": len(load_puzzles()),
                    }
                )
            self.send_json({"error": f"Generation failed: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_generate_groups(self) -> None:
        factory = None
        try:
            body = self.read_json_body()
            count = max(1, min(int(body.get("count", 1)), 24))
            difficulty = str(body.get("difficulty", "mixed"))
            theme = str(body.get("theme", ""))
            save = bool(body.get("save", True))
            model = str(body.get("model", "")).strip() or None
            generator_model = str(body.get("generator_model", "")).strip() or None
            reviewer_model = str(body.get("reviewer_model", "")).strip() or None

            factory = GroupGenerationFactory(
                existing_groups=load_group_bank(),
                existing_puzzles=load_puzzles(),
                model=model,
                generator_model=generator_model,
                reviewer_model=reviewer_model,
            )
            response = factory.generate_groups(
                target_count=count,
                difficulty=difficulty,
                theme=theme,
                save=save,
            )
            response["group_bank_total"] = response.pop("bank_total")
            response["bank_total"] = len(load_puzzles())
            save_latest_run(response)
            self.send_json(response)
        except ClaudeError as error:
            if factory is not None:
                save_latest_run(
                    {
                        "accepted": [],
                        "rejected": [{"errors": [str(error)], "stage": "claude_error"}],
                        "trace": [event.to_dict() for event in factory.trace],
                        "saved": False,
                        "bank_total": len(load_puzzles()),
                        "group_bank_total": len(load_group_bank()),
                    }
                )
            self.send_json({"error": str(error)}, HTTPStatus.BAD_GATEWAY)
        except Exception as error:
            if factory is not None:
                save_latest_run(
                    {
                        "accepted": [],
                        "rejected": [{"errors": [str(error)], "stage": "generation_error"}],
                        "trace": [event.to_dict() for event in factory.trace],
                        "saved": False,
                        "bank_total": len(load_puzzles()),
                        "group_bank_total": len(load_group_bank()),
                    }
                )
            self.send_json({"error": f"Group generation failed: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw or "{}")

    def serve_static(self, request_path: str) -> None:
        if request_path in ("", "/"):
            path = WEB_DIR / "index.html"
        else:
            relative = unquote(request_path).lstrip("/")
            path = WEB_DIR / relative

        try:
            resolved = path.resolve()
            resolved.relative_to(WEB_DIR.resolve())
        except ValueError:
            self.send_json({"error": "Invalid path."}, HTTPStatus.BAD_REQUEST)
            return

        if not resolved.exists() or not resolved.is_file():
            self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json_bytes(payload)
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Connection Forge web MVP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), PuzzleRequestHandler)
    print(f"Connection Forge running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
