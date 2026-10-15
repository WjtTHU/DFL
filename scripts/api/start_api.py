#!/usr/bin/env python
import argparse
import os

import uvicorn

from wip_analysis.utils.runtime_env import load_runtime_env

load_runtime_env()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the API server")
    default_reload = os.environ.get("RELOAD", "false").lower() == "true"
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("WIP_ANALYSIS_LOG_LEVEL", "info").lower(),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Set the logging level (default: from WIP_ANALYSIS_LOG_LEVEL env or 'info')",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WIP_ANALYSIS_PORT", "8724")),
        help="Set the port to listen on (default: from PORT/WIP_ANALYSIS_PORT env or 8724)",
    )
    parser.set_defaults(reload=default_reload)
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        help="Enable auto-reload",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable auto-reload",
    )
    args = parser.parse_args()

    # Set environment variable so FastAPI app can access it
    os.environ["WIP_ANALYSIS_LOG_LEVEL"] = args.log_level.upper()
    os.environ["WIP_ANALYSIS_PORT"] = str(args.port)

    uvicorn.run(
        "wip_api.main:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        reload_dirs=["src"],
        log_level=args.log_level,
    )
