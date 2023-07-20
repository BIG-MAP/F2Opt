"""Schemas for the server API."""

from typing import Any, Dict, List

from pydantic import BaseModel


# FINALES2.server.schemas


class Request(BaseModel):
    quantity: str
    methods: List[str]
    parameters: Dict[str, Dict[str, Any]]
    tenant_uuid: str


class Result(BaseModel):
    data: Dict[str, Any]
    quantity: str
    method: List[str]
    parameters: Dict[str, Dict[str, Any]]
    tenant_uuid: str
    request_uuid: str
