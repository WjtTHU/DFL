from typing import Any, NoReturn

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder


def build_error_detail(reason: str, **context: Any) -> dict[str, Any]:
    detail: dict[str, Any] = {"reason": reason}
    sanitized_context = {k: v for k, v in context.items() if v is not None}
    if sanitized_context:
        detail["context"] = jsonable_encoder(sanitized_context)
    return jsonable_encoder(detail)


def raise_api_error(status_code: int, reason: str, **context: Any) -> NoReturn:
    raise HTTPException(status_code=status_code, detail=build_error_detail(reason, **context))
