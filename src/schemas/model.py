from typing_extensions import TypedDict
from typing_extensions import Literal
from pydantic import BaseModel, Field

class State(TypedDict):
    input: str
    decision: str
    output: str

class Route(BaseModel):
    step: Literal["SALES_RECO_NODE", "COMPLIANCE_CHECK_NODE", "VENDOR_ONBOARDING_NODE", "OPS_STOCK_NODE", "GENERAL_KB_NODE", "DEFAULT_NODE"] = Field(
        None, description="The next step in the routing process"
    )